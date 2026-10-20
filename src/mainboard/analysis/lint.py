"""Checking and fixing one project with the SQL rules.

A check extracts the fact families the selected rules' SQL names, then runs every rule's query at
once on parallel cursors (DuckDB plans them concurrently and parallelizes each). A fix runs every
fixable rule's edit query the same way, groups the edit rows by the finding they repair and fixes
the project to its fixpoint (fixing.py). A test runs every rule's examples through the same check
and fix.
"""

import difflib
import json
import os
import pkgutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the frozen mcmr CLI with fixed argv since=2026-10-13
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Self

from cyclopts import App

from ..log import logger
from .catalog import Example, Family, Repair, Rule, define, install, needed
from .config import ProjectConfig
from .consumers import DECLARED, Consumers
from .edits import Edit, Import, Remove, Rename, Replace
from .facts import Budget, Facts, Query, listed, snake, vendored
from .findings import Hit, Span
from .fixing import Fixpoint
from .providers.ducks import Ducks
from .providers.external import Tool
from .rules.core import Core

if TYPE_CHECKING:
    from collections.abc import Iterable

app = App(name="lint", help="Check and fix a project with the SQL rules.")
_FIXABLE = {"safe": ("safe",), "review": ("safe", "review")}
# The type checkers' lane: the hook runs ty as a step of its own and the fixpoint judges every fix
# by ty's diagnostics, so no check reads them.
_TYPE_CHECKED = "PY-TY0001"
_SKIPPED = "skipped: it reads facts no SQL family derives"


def load_catalog() -> None:
    """Import every rule module, registering its rules and relations; a module that fails to
    import (one being written) is reported and skipped."""
    package = import_module("mainboard.analysis.rules")
    for module in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        try:
            import_module(module.name)
        except (SyntaxError, ImportError, NameError, AttributeError, TypeError) as error:
            print(f"skipped {module.name}: {type(error).__name__}: {error}")


class Lint:
    """The selected rules, configured for one project."""

    def __init__(
        self,
        root: Path,
        rules: Iterable[type[Rule]],
        *,
        memory: str = "2GB",
        families: tuple[str, ...] = (),
        consumers: Consumers | None = None,
        fixing: bool = True,
    ) -> None:
        self.root, self.budget, self.fixing = root, Budget(memory=memory), fixing
        self.consumers = consumers or Consumers(root)
        self.current: Facts | None = None
        self.derived = {
            name for name, family in Family.defined.items() if family.verified or name in families
        }
        self.ducks = Ducks(budget=self.budget)
        self.project = ProjectConfig.of(root)
        present = {self.ducks.language(PurePosixPath(path).suffix) for path in listed(root)}
        self.rules = [
            rule
            for rule in (
                cls(**self.project.rule(cls.id).settings)
                for cls in rules
                if cls.id != _TYPE_CHECKED and self.project.rule(cls.id).enabled
            )
            if rule.language is None or rule.language in present
        ]
        queries = self.queries = [query for rule in self.rules for query in self._queries(rule)]
        wanted = [*queries, *(Query(sql=r.query) for r in needed(queries))]
        self.families = self._families(wanted)
        views = {view for query in wanted for view in query.views()}
        derived_queries = [query for name in self.families for query in self._sql_of(name)]
        self.core = Core.reading(views | {view for q in derived_queries for view in q.views()})
        self.reads_trees = bool(self.core) or any(
            view.startswith("ast_")
            for view in views | {view for query in derived_queries for view in query.views()}
        )

    def facts(self, root: Path) -> Facts:
        """The fact base of `root`, with every relation and empty view the rules need: the syntax
        trees are read and the derived families computed, each after those it reads. It replaces
        the previous fact base, whose database is closed and deleted."""
        derived = [Family.defined[name] for name in self.families]
        if self.current:
            self.current.close()
        db = self.current = Facts(root, self.budget)
        if derived or self.reads_trees:
            if self.reads_trees:
                self.ducks.register(db, root)
            for family in derived:
                family.provide(db, root)
            define(
                db,
                [sql for family in derived for sql in family.relations.values()]
                + list(self.core.values()),
            )
            if self.core:
                db.adopt(Core.family, self.core)
            for family in _ordered(derived):
                db.adopt(family.family, family.relations, empty=not family.grounded(db))
        Tool.register_all(db, root, self.queries)
        self.rules = [rule for rule in self.rules if self._served(db, rule)]
        install(db, self.applicable(db), fixes=self.fixing)
        for rule in self.rules:
            for query in self._queries(rule):
                db.ensure(query)
        return db

    def _queries(self, rule: Rule) -> list[Query]:
        """The SQL `rule` runs: its findings and, when fixing, its fix."""
        return [query for query in (rule.rows(), rule.edits() if self.fixing else None) if query]

    def _served(self, db: Facts, rule: Rule) -> bool:
        """Whether the fact base holds every view `rule` and its relations read: a rule of a
        family no SQL derives (Rust surfaces, TypeScript modules) is skipped with a warning."""
        queries = self._queries(rule)
        try:
            for query in [*queries, *(Query(sql=r.query) for r in needed(queries))]:
                db.ensure(query)
        except ValueError as error:
            logger.warning("{} is skipped: {}", rule.id, error)
            return False
        return True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        """Remove what this run cached and spilled."""
        if self.current:
            self.current.close()
        self.ducks.close()

    def _families(self, queries: list[Query]) -> list[str]:
        """The derived families `queries` read, with those the derived ones read in turn. A
        family no SQL derives is left out, and `facts` skips the rules that read it."""
        found = {name for name in families_read(queries) if name in self.derived}
        while more := {
            family for name in found for family in self._read_by(name) if family not in found
        }:
            found |= more
        return sorted(found)

    def _read_by(self, name: str) -> set[str]:
        """The families the derived family `name` reads, which must be derived too."""
        read = families_read(self._sql_of(name)) - {name}
        if stranger := next((f for f in sorted(read) if f not in self.derived), None):
            raise ValueError(
                f"derived family {name} reads {stranger}, which is not derived (no verified SQL "
                f"family of that name; pass --family {stranger} if one is being ported)"
            )
        return read

    @staticmethod
    def _sql_of(name: str) -> list[Query]:
        """The SQL of the derived family `name`: its relations and the shared ones they call."""
        own = Family.defined[name].queries()
        return [*own, *(Query(sql=relation.query) for relation in needed(own))]

    def applicable(self, db: Facts) -> list[Rule]:
        """The rules whose language the project holds."""
        return [
            rule for rule in self.rules if rule.language is None or rule.language in db.languages
        ]

    def findings(self, db: Facts) -> dict[str, list[tuple]]:
        """`(path, start_line, start_column, message)` rows per rule; paths a rule excludes and
        vendored paths (asked of the project, since `db` may be a fix's mirror) dropped."""

        def run(rule: Rule) -> list[tuple]:
            accepts = self.project.rule(rule.id).accepts
            declared = [DECLARED[rule.id]] if rule.id in DECLARED else []
            rows = db.cursor(rule.language).execute(
                rule.statement(*declared), rule.rows().parameters
            )
            kept = [row for row in rows.fetchall() if accepts(row[0])]
            return self._unconsumed(rule, kept) if declared else kept

        rules = self.applicable(db)
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            found = dict(zip([rule.id for rule in rules], pool.map(run, rules), strict=True))
        skipped = vendored(self.root, {row[0] for rows in found.values() for row in rows})
        return {
            rule: [row for row in rows if row[0] not in skipped] for rule, rows in found.items()
        }

    def _unconsumed(self, rule: Rule, rows: list[tuple]) -> list[tuple]:
        """`rows` (the usual four columns and the name the finding declares) as findings, without
        those of a dead-code rule whose declaration another repository of the workspace imports:
        this project's facts cannot see that use (consumers.py)."""
        kept = [
            row[:4]
            for row in rows
            if row[4] is None or self.consumers.used(rule.id, row[0], row[4]) is None
        ]
        if len(kept) < len(rows):
            logger.debug(
                "{}: {} of {} findings dropped, other repositories import their names",
                rule.id,
                len(rows) - len(kept),
                len(rows),
            )
        return kept

    def repairs(self, db: Facts, applicability: tuple[str, ...]) -> dict[str, list[Repair]]:
        """Per fixable rule, the edits of each finding it repairs, as one group saying what it
        changed."""
        if not self.fixing:
            raise ValueError("this Lint was built without fixes (`fixing=False`)")

        def run(rule: Rule) -> list[Repair]:
            accepts = self.project.rule(rule.id).accepts
            cursor = db.cursor(rule.language)
            cursor.execute(rule.finding_rows(), rule.rows().parameters)
            places = {
                key: (path, line or 1, change)
                for key, path, line, change in cursor.execute(rule.changes()).fetchall()
            }
            if (query := rule.edits("finding_rows")) is None:
                result = cursor.execute("SELECT * FROM finding_rows ORDER BY finding_key")
            else:
                result = cursor.execute(query.sql, query.parameters)
            columns = [column for column, *_ in result.description]
            grouped: defaultdict[object, list[Edit]] = defaultdict(list)
            for number, row in enumerate(result.fetchall()):
                record = dict(zip(columns, row, strict=True))
                if query is None:
                    key = record.pop("finding_key")
                    grouped[key].extend(rule.repairs(Hit(**record)))
                else:
                    grouped[record.get("finding_key", ("row", number))].extend(_edits(record))
            return [
                Repair(group, **_where(rule, group, places.get(key)))
                for key, group in grouped.items()
                if group and all(accepts(edit.file()) for edit in group)
            ]

        rules = [
            rule
            for rule in self.applicable(db)
            if rule.fixes() != "none" and rule.applicability in applicability
        ]
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            found = dict(zip([rule.id for rule in rules], pool.map(run, rules), strict=True))
        files = {edit.file() for groups in found.values() for group in groups for edit in group}
        skipped = vendored(self.root, files)
        return {
            rule: [group for group in groups if not any(e.file() in skipped for e in group)]
            for rule, groups in found.items()
        }

    def flat(self, db: Facts) -> set[tuple]:
        """Every finding as `(path, line, column, rule, message)`, the shape a proof compares."""
        return {
            (row[0], row[1], row[2], rule, row[3])
            for rule, rows in self.findings(db).items()
            for row in rows
        }


def baseline_findings(report: dict, rule: str) -> set[tuple]:
    """A rule's findings from a stored full mcmr report."""
    return {
        (f["span"]["path"], f["span"]["start_line"], f["span"]["start_column"], f["message"])
        for failure in report["failures"]
        if failure["rule"] == rule
        for f in failure["findings"]
    }


def old_findings(root: Path, rule: str) -> set[tuple]:
    """mcmr's findings for one rule, run live; a rule mcmr's deterministic lanes do not run
    (an external one) has none."""
    run = subprocess.run(
        [
            "mcmr",
            "check",
            str(root),
            "--select",
            rule,
            "--format",
            "json",
            "--limit",
            "1000000",
            "--deterministic",
            "--no-contextual",
            "--no-external",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return baseline_findings(json.loads(run.stdout), rule)
    except json.JSONDecodeError:
        return set()


def _where(rule: Rule, group: list[Edit], place: tuple[str, int, str | None] | None) -> dict:
    """The path, line and change text of a repair: the finding's own and the rule's `change`
    template filled from its row; a group of no finding (a fix whose edit rows carry no
    `finding_key`) is placed at its first edit, and a rule without a template says what the
    edits did."""
    path, line, change = place or (group[0].file(), 1, None)
    return {"path": path, "line": line, "change": change or rule.describe(group)}


def _edits(row: dict) -> list[Edit]:
    """The edits one row of a fix query states: a removal or a rename (to `target`), a replaced
    span, a needed import."""
    made: list[Edit] = []
    match row.get("operation"):
        case "remove":
            made.append(Remove(path=row["path"]))
        case "rename":
            made.append(Rename(path=row["path"], target=row["target"]))
    if row.get("text") is not None:
        line, column = row["start_line"], row["start_column"]
        made.append(
            Replace(
                span=Span(
                    path=row["path"],
                    start_line=line,
                    start_column=column,
                    end_line=row.get("end_line", line),
                    end_column=row.get("end_column", column),
                ),
                text=row["text"],
            )
        )
    if row.get("import"):
        made.append(Import(path=row["path"], statement=row["import"]))
    return made


def families_read(queries: Iterable[Query]) -> set[str]:
    """The defined families whose views `queries` name, a view naming its family as its prefix;
    `Core`, built from the tree itself, is not one."""
    known = {snake(name): name for name in Family.defined if name != Core.family}
    return {
        known[max((p for p in known if view.startswith(f"{p}_")), key=len)]
        for query in queries
        for view in query.views()
        if any(view.startswith(f"{p}_") for p in known)
    }


def _ordered(families: list[type[Family]]) -> list[type[Family]]:
    """Derived families, each after the derived families its SQL reads."""
    names = {family.family for family in families}
    reads = {
        family.family: families_read(family.queries()) & names - {family.family}
        for family in families
    }
    done: list[type[Family]] = []
    while len(done) < len(families):
        ready = [
            f for f in families if f not in done and reads[f.family] <= {d.family for d in done}
        ]
        if not ready:
            raise ValueError(f"derived families read each other in a cycle: {sorted(names)}")
        done += ready
    return done


def _select(rule: tuple[str, ...]) -> list[type[Rule]]:
    load_catalog()
    if rule:
        return [Rule.defined[r] for r in rule]
    return [spec for name, spec in Rule.defined.items() if name != _TYPE_CHECKED]


@app.command
def check(
    root: Path,
    *,
    rule: tuple[str, ...] = (),
    memory: str = "2GB",
    family: tuple[str, ...] = (),
) -> None:
    """Every finding of the selected rules (all when none), with the time of each stage.

    root: the project to check.
    rule: rule ids, every SQL rule when empty.
    memory: a DuckDB memory limit such as `4GB`.
    family: mcmr families to derive in SQL though not yet verified (`CallFact`).
    """
    start = time.perf_counter()
    with Lint(root, _select(rule), memory=memory, families=family, fixing=False) as lint:
        db = lint.facts(root)
        extracted = time.perf_counter()
        found = lint.findings(db)
    ran = time.perf_counter()
    for rule_id, rows in sorted(found.items()):
        for path, line, column, message in sorted(rows, key=str):
            print(f"{path}:{line}:{column}\t{rule_id}\t{message}")
    print(
        f"{sum(map(len, found.values()))} findings from {len(found)} rules; facts "
        f"{extracted - start:.2f} s, rules {ran - extracted:.3f} s"
    )


@app.command
def compare(
    root: Path,
    *,
    rule: tuple[str, ...] = (),
    baseline: Path | None = None,
    memory: str = "2GB",
    family: tuple[str, ...] = (),
) -> None:
    """Each selected rule's findings against mcmr's (frozen): equal, missing, extra.

    root: the project both engines analyze.
    rule: rule ids, every SQL rule when empty.
    baseline: a stored full `mcmr check --format json` report; without it mcmr runs per rule.
    memory: a DuckDB memory limit such as `4GB`.
    family: mcmr families to derive in SQL though not yet verified (`CallFact`).
    """
    report = json.loads(baseline.read_text()) if baseline else None
    with Lint(root, _select(rule), memory=memory, families=family, fixing=False) as lint:
        found = lint.findings(lint.facts(root))
    for spec in lint.rules:
        old = baseline_findings(report, spec.id) if report else old_findings(root, spec.id)
        skipped = vendored(root, {finding[0] for finding in old})
        old = {finding for finding in old if finding[0] not in skipped}
        new = set(found.get(spec.id, []))
        print(
            f"{spec.id}\t{old == new}\t{len(old)}\t{len(new)}\tmissing {len(old - new)}\t"
            f"extra {len(new - old)}",
            flush=True,
        )
        for sample in sorted(old ^ new, key=str)[:3]:
            print("   ", "missing" if sample in old else "extra", sample)


@app.command
def fix(
    root: Path,
    *,
    rule: tuple[str, ...] = (),
    write: bool = False,
    unsafe: bool = False,
    passes: int = 100,
    memory: str = "2GB",
    family: tuple[str, ...] = (),
) -> None:
    """Fix every fixable finding, pass after pass, until nothing fixable is left; each state is
    proven on a mirror and only the last proven one is written (with `--write`).

    root: the project to fix.
    rule: rule ids whose fixes run, every fixable SQL rule when empty.
    write: write the last proven state into the project; without it the run only reports.
    unsafe: include fixes marked for review, not only safe ones.
    passes: the most passes before stopping.
    memory: a DuckDB memory limit such as `4GB`.
    family: mcmr families to derive in SQL though not yet verified (`CallFact`).
    """
    applicability = _FIXABLE["review" if unsafe else "safe"]
    with Lint(root, _select(rule), memory=memory, families=family) as lint:
        for report in Fixpoint(lint, root, applicability, limit=passes).run(write=write):
            print(report, flush=True)


@app.command
def test(*, rule: tuple[str, ...] = (), memory: str = "2GB", family: tuple[str, ...] = ()) -> None:
    """Run the selected rules' examples (every rule's when none): each `bad` must be flagged,
    the rule's fix must turn it into `good`, and `good` must not be flagged.

    rule: rule ids.
    memory: a DuckDB memory limit for each example's fact base.
    family: mcmr families to derive in SQL though not yet verified (`CallFact`).
    """
    failed = 0
    for spec in _select(rule):
        for number, example in enumerate(spec.examples, 1):
            verdict = _trial(spec, example, memory, family)
            failed += verdict not in {"pass", _SKIPPED}
            print(f"{spec.id} example {number}: {verdict}", flush=True)
    print(f"{failed} examples failed")
    if failed:
        raise SystemExit(1)


def _trial(spec: type[Rule], example: Example, memory: str, family: tuple[str, ...]) -> str:
    """One example's verdict: `pass`, or what went wrong. The snippet lives in `example.path`,
    by default a TOML snippet in `pyproject.toml` and code in a module of the rule's language
    (a test module when it defines a test); a Python rule's project always holds a module, so it
    holds Python whatever the snippet is."""
    with tempfile.TemporaryDirectory(prefix="mb-trial-") as scratch:
        root = Path(scratch)
        language = spec().language
        path = root / (example.path or _example_path(language, example.bad))
        for name, text in {**example.context, str(path.relative_to(root)): example.bad}.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
        if language == "python" and not any(root.rglob("*.py")):
            (root / "anchor.py").write_text("")
        with Lint(root, [spec], memory=memory, families=family) as lint:
            return _verdict(spec, example, lint, root, path)


def _verdict(spec: type[Rule], example: Example, lint: Lint, root: Path, path: Path) -> str:
    """Whether `bad` is flagged, the fix turns it into `good`, and `good` is not flagged."""
    db = lint.facts(root)
    if not lint.rules:
        return _SKIPPED
    if not lint.findings(db).get(spec.id):
        return "bad is not flagged"
    if example.good == "":
        return "pass"
    if spec.fixes() != "none":
        report = "\n".join(Fixpoint(lint, root, _FIXABLE["review"]).run(write=True))
        if example.good is None:
            return "pass" if not path.exists() else f"the fix keeps the file\n{report}"
        fixed_text = path.read_text(encoding="utf-8")
        if fixed_text != example.good:
            diff = difflib.unified_diff(
                example.good.splitlines(keepends=True),
                fixed_text.splitlines(keepends=True),
                "good",
                "fixed",
            )
            return f"the fix does not give good\n{''.join(diff)}{report}"
    path.write_text(example.good or "", encoding="utf-8")
    if lint.findings(lint.facts(root)).get(spec.id):
        return "good is flagged"
    return "pass"


def _example_path(language: str | None, text: str) -> str:
    if text.lstrip().startswith("["):
        return "pyproject.toml"
    if language in (None, "python"):
        return "test_module.py" if "def test_" in text else "module.py"
    return {
        "rust": "lib.rs",
        "cuda": "kernel.cu",
        "typescript": "module.ts",
        "swift": "View.swift",
    }.get(language, f"module.{language}")


from .commands import catalog as catalog  # noqa: E402
from .commands import diagram as diagram  # noqa: E402
from .commands import evidence as evidence  # noqa: E402
from .commands import impact as impact  # noqa: E402
from .commands import matrix as matrix  # noqa: E402
from .commands import prose as prose  # noqa: E402
from .commands import swift_inventory as swift_inventory  # noqa: E402
from .commands import uses as uses  # noqa: E402

if __name__ == "__main__":
    app()
