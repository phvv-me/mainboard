"""Who uses a project's names from outside it.

One repository's run sees the imports inside the project, never the ones in the rest of the
monorepo, so a rule that calls a public name dead (the reach rules) or a fix that renames or
removes one would break the projects that import it. The consumers of a project are the other
repositories its workspace owns: the topmost working tree's `[lint].owners` minus its
`[lint].exclude` (forks) and the `linguist-vendored` trees, and minus any owner holding a package
of the same name (a copy or the original of the project itself).

A consumer's imports are found by text first (`rg` lists the files that import one of the project's
top-level packages), then only those files are parsed, with the engine's own syntax-tree provider,
and read as the pairs `(module, name)` they import or reach through an alias. Names count by the
module's path inside the project, so `from patos.sql import Column` uses the `Column` declared in
`src/patos/sql/columns.py` (a re-export is followed no further: a name counts as used when any
module on its way is imported). The index is built on first use, once per run.
"""

import hashlib
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs ripgrep with fixed argv since=2026-10-16
import threading
import time
import tomllib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence, Set
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from ...log import logger
from ..engine.facts import Budget, Facts
from ..providers.ducks import Ducks
from .surface import positional, surface
from .workspace import Workspace, module_key

if TYPE_CHECKING:
    from ..engine.apply import Plan


_MODULES = frozenset({"ALL-ARCH0005"})
_EXPORTS = frozenset({"PY-MODU0005", "PY-RUFF0001", "PY-RUFF0002"})
_IMPORTS = """
    WITH statements AS (
      SELECT fact_id, path, node_id, type, name FROM ast_nodes
      WHERE type IN ('import_statement', 'import_from_statement')
    ), keyword AS (
      SELECT n.fact_id, n.parent_id AS statement, min(n.sibling_index) AS position
      FROM ast_nodes n JOIN statements s ON s.fact_id = n.fact_id AND s.node_id = n.parent_id
      WHERE n.type = 'import' GROUP BY ALL
    ), bound AS (
      SELECT s.path, s.type AS kind, s.name AS module, p.type AS part, p.name AS spelled,
             coalesce(d.name, p.name) AS imported
      FROM statements s
      JOIN keyword k ON k.fact_id = s.fact_id AND k.statement = s.node_id
      JOIN ast_nodes p ON p.fact_id = s.fact_id AND p.parent_id = s.node_id
        AND p.sibling_index > k.position
        AND p.type IN ('dotted_name', 'aliased_import', 'wildcard_import')
      LEFT JOIN ast_nodes d ON d.fact_id = p.fact_id AND d.parent_id = p.node_id
        AND p.type = 'aliased_import' AND d.type = 'dotted_name'
      WHERE s.type = 'import_statement' OR NOT starts_with(s.name, '.')
    ), imports AS (
      SELECT path,
             CASE kind WHEN 'import_statement' THEN imported ELSE module END AS module,
             CASE WHEN kind = 'import_statement' THEN ''
                  WHEN part = 'wildcard_import' THEN '*' ELSE imported END AS name,
             CASE part WHEN 'wildcard_import' THEN NULL
                       WHEN 'aliased_import' THEN spelled
                       WHEN 'dotted_name' THEN CASE kind WHEN 'import_statement'
                         THEN split_part(imported, '.', 1) ELSE imported END END AS local,
             CASE kind WHEN 'import_statement'
                       THEN CASE part WHEN 'aliased_import' THEN imported
                                      ELSE split_part(imported, '.', 1) END
                       ELSE module || '.' || imported END AS target
      FROM bound
    )
"""
# Ordered, so the consumer named for a module (the first one seen) is the same every run.
_PAIRS = (
    _IMPORTS
    + """
    SELECT module, name, min(path) AS consumer FROM imports GROUP BY ALL ORDER BY ALL
"""
)
_CHAINS = (
    _IMPORTS
    + """
    , chains AS (
      SELECT path, regexp_extract(name, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)*') AS dotted
      FROM ast_nodes WHERE type = 'attribute'
    )
    SELECT DISTINCT i.target || substr(a.dotted, length(i.local) + 1) AS chain, min(a.path)
    FROM chains a
    JOIN (SELECT DISTINCT path, local, target FROM imports WHERE local IS NOT NULL) i
      ON i.path = a.path AND split_part(a.dotted, '.', 1) = i.local
    WHERE contains(a.dotted, '.')
    GROUP BY ALL ORDER BY ALL
"""
)


class Index:
    """The `(module, name)` pairs a project's consumers import, with a consumer file for each.

    submodules: every trailing part of the paths of the project's modules (`log`, `mainboard/log`
        ...), so `mainboard.log` reads as a module imported, not as a name used.
    """

    def __init__(self, submodules: frozenset[str] = frozenset()) -> None:
        self.names: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
        self.modules: dict[str, str] = {}
        self.submodules = submodules

    def add_chain(self, chain: str, *, consumer: str) -> None:
        """Every module and name a dotted path reaches: `a.b.c` is `c` of `a.b` and `b` of `a`."""
        parts = chain.split(".")
        self.add_module(parts[0], consumer=consumer)
        for end in range(1, len(parts)):
            self.add_name(".".join(parts[:end]), name=parts[end], consumer=consumer)
            self.add_module(".".join(parts[: end + 1]), consumer=consumer)

    def add_import(self, module: str, *, name: str, consumer: str) -> None:
        """One `from module import name`, or `import module` when `name` is empty."""
        if not name:
            self.add_chain(module, consumer=consumer)
            return
        self.add_module(module, consumer=consumer)
        self.add_name(module, name=name, consumer=consumer)
        if name != "*":
            self.add_module(f"{module}.{name}", consumer=consumer)

    def add_module(self, module: str, *, consumer: str) -> None:
        self.modules.setdefault(module.replace(".", "/"), consumer)

    def add_name(self, module: str, *, name: str, consumer: str) -> None:
        path = module.replace(".", "/")
        if f"{path}/{name}" not in self.submodules:
            self.names[name].append((path, consumer))

    def user(self, path: str, *, name: str, exact: bool = False) -> str | None:
        """A consumer file importing `name` from the module `path`.

        Unless `exact`, a package holding the module counts too, re-exporting it as far as the text
        tells.
        """
        inside = f"/{module_key(path)}/"
        return next(
            (
                consumer
                for wanted in (name, "*")
                for module, consumer in self.names.get(wanted, ())
                if (inside.endswith(f"/{module}/") if exact else f"/{module}/" in inside)
            ),
            None,
        )

    def user_of_module(self, path: str) -> str | None:
        """A consumer file importing the module `path` itself, whole or by a name."""
        key = module_key(path)
        parent, _, stem = key.rpartition("/")
        for module, consumer in self.modules.items():
            if f"/{key}".endswith(f"/{module}"):
                return consumer
        return next(
            (
                consumer
                for module, consumer in self.names.get(stem, ())
                if f"/{parent}".endswith(f"/{module}")
            ),
            None,
        )


class Consumers:
    """The imports of a project's consumers, built on first use.

    root: the project; its consumers are found through its workspace.
    memory: the DuckDB limit for parsing the consumer files.
    """

    _index: Index | None

    @property
    def index(self) -> Index:
        """The consumers' imports, built by the first caller while the others wait."""
        with self._lock:
            if self._index is None:
                self._index = self._build()
            return self._index

    @staticmethod
    def _excluded(top: Path) -> list[str]:
        """The workspace's `[lint].exclude` patterns, which no consumer file may sit under."""
        try:
            return list(tomllib.loads((top / "mainboard.toml").read_text())["lint"]["exclude"])
        except FileNotFoundError, KeyError:
            return []

    @staticmethod
    def _importers(
        top: Path, owners: list[str], *, packages: list[str], exclude: Sequence[str]
    ) -> list[str]:
        """The Python files of `owners` importing one of `packages` by text, relative to `top`."""
        alternatives = "|".join(packages)
        pattern = rf"^\s*(from\s+({alternatives})(\.|\s)|import\s+[\w., ]*\b({alternatives})\b)"
        found = subprocess.run(
            [
                str(shutil.which("rg")),
                "--files-with-matches",
                "--hidden",
                "--no-messages",
                "--type",
                "py",
                "--glob",
                "!.git",
                *(f"--glob=!{glob}" for glob in exclude),
                "--regexp",
                pattern,
                "--",
                *owners,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=top,
        )
        if found.returncode > 1:
            raise RuntimeError(f"ripgrep failed: {found.stderr.strip()}")
        return sorted(found.stdout.splitlines())

    def __init__(self, root: Path, *, memory: str = "1GB") -> None:
        self.root, self.memory = root, memory
        self._lock = threading.Lock()
        self._index = None

    def _broken(
        self, plan: Plan, edited: Mapping[str, bytes], gone: Sequence[str]
    ) -> Iterator[str]:
        for path, data in edited.items():
            yield from self._narrowed(path, before=plan.source(path).data, after=data)
        for path in gone:
            if path.endswith(".py") and (consumer := self.index.user_of_module(path)):
                yield f"{path}: the fix removes the module, which {consumer} imports"

    def _build(self) -> Index:
        """Search, parse and read the consumers."""
        workspace = Workspace(self.root)
        index = Index(workspace.submodules())
        packages = workspace.packages()
        owners = workspace.consumers(packages) if packages else []
        if owners and workspace.top is not None:
            self._indexed(index, top=workspace.top, owners=owners, packages=packages)
        return index

    def _cache(self) -> Path:
        """The syntax trees of this project's consumers, kept between runs by file content.

        A later run parses only the consumer files that changed.
        """
        digest = hashlib.blake2b(str(self.root).encode(), digest_size=8).hexdigest()
        return Path.home() / f".cache/mainboard/analysis/consumers-{digest}.duckdb"

    def _indexed(self, index: Index, *, top: Path, owners: list[str], packages: list[str]) -> None:
        """Add to `index` what the files of `owners` import of `packages`, the search timed."""
        started = time.perf_counter()
        own = f"{self.root.resolve().relative_to(top).as_posix()}/**"
        files = Consumers._importers(
            top, owners, packages=packages, exclude=[*Consumers._excluded(top), own]
        )
        searched = time.perf_counter()
        if files:
            self._read(index, top, files, set(packages))
        logger.debug(
            "{} consumer files in {} repositories import {}: search {:.2f} s, read {:.2f} s",
            len(files),
            len(owners),
            packages,
            searched - started,
            time.perf_counter() - searched,
        )

    def _narrowed(self, path: str, *, before: bytes, after: bytes) -> Iterator[str]:
        """What the edit of `path` takes from its consumers.

        That is names they import and positions they may pass; nothing for a file that does not
        parse either way.
        """
        try:
            lost, was, now = (
                surface(before) - surface(after),
                positional(before),
                positional(after),
            )
        except SyntaxError:
            return
        narrowed = {key[0] for key, count in was.items() if now.get(key, count) < count}
        reasons = [
            *(
                (name, "the fix removes `{name}`, which {consumer} imports")
                for name in sorted(lost)
            ),
            *(
                (
                    name,
                    "the fix makes parameters of `{name}` keyword-only, and {consumer} imports "
                    "it, so its callers may pass them by position",
                )
                for name in sorted(narrowed)
            ),
        ]
        for name, reason in reasons:
            if consumer := self.index.user(path, name=name):
                yield f"{path}: {reason.format(name=name, consumer=consumer)}"

    def _read(self, index: Index, top: Path, files: Sequence[str], packages: Set[str]) -> None:
        """Parse `files` and add what they import of `packages`.

        The files are linked into a folder of their own, which the parser reads.
        """
        with closing(Budget(memory=self.memory)) as budget:
            mirror = budget.folder / "consumers"
            for file in files:
                (mirror / file).parent.mkdir(parents=True, exist_ok=True)
                (mirror / file).symlink_to(top / file)
            with closing(Facts(mirror, budget)) as facts:
                ducks = Ducks(
                    budget=budget, cache=self._cache(), exclusive=True, report_gaps=False
                )
                ducks.register(facts, mirror)
                pairs = facts.db.execute(_PAIRS).fetchall()
                chains = facts.db.execute(_CHAINS).fetchall()
        for module, name, consumer in (row for row in pairs if _owned(row[0], packages)):
            index.add_import(module, name=name, consumer=consumer)
        for chain, consumer in (row for row in chains if _owned(row[0], packages)):
            index.add_chain(chain, consumer=consumer)

    def broken(self, plan: Plan) -> list[str]:
        """`path: reason` for each file whose fixes would break a consumer's import.

        That is a file whose edits remove or rename a name a consumer imports, or one the plan
        removes or renames while a consumer imports its module.
        """
        edited = {path: data for path, data in plan.written().items() if path.endswith(".py")}
        gone = [*plan.removed, *plan.renamed]
        if not edited and not any(path.endswith(".py") for path in gone):
            return []
        return list(self._broken(plan, edited, gone))

    def used(self, rule: str, *, path: str, declared: str) -> str | None:
        """A consumer file using what a finding of `rule` declares, if any."""
        if rule in _MODULES:
            return self.index.user_of_module(path)
        return self.index.user(path, name=declared, exact=rule in _EXPORTS)


def _owned(dotted: str, packages: Set[str]) -> bool:
    """Whether the dotted module path `dotted` starts at one of `packages`."""
    return dotted.split(".")[0] in packages
