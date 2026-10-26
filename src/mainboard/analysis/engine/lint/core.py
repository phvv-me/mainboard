"""The selected rules, configured for one project: its facts, findings and repairs."""

import os
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from graphlib import TopologicalSorter
from importlib import import_module
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Self

import duckdb

from ....core.execution.modules import submodules
from ....log import logger
from ...catalog import Family, Repair, Rule, define, install, needed
from ...constants import DECLARED
from ...project.config import ProjectConfig
from ...project.consumers import Consumers
from ...providers.ducks import Ducks
from ...providers.external import Tool
from ...rules.core import Core
from ..facts import Budget, Facts, Query, listed, snake, vendored
from .repairs import grouped, placed

_DELETE_CONSUMED_FINDING_ROWS_SQL = (
    "DELETE FROM finding_rows WHERE list_contains($keys, finding_key)"
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from duckdb import DuckDBPyConnection

# The type checkers' lane: the hook runs ty as a step of its own and the fixpoint judges every fix
# by ty's diagnostics, so no check reads them.
_TYPE_CHECKED = "PY-TY0001"


def load_catalog() -> None:
    """Import every fact provider and rule module, registering its families, rules and relations.

    A module that fails to import (one being written) is reported and skipped.
    """
    for name in [
        *submodules("mainboard.analysis.providers"),
        *submodules("mainboard.analysis.rules"),
    ]:
        try:
            import_module(name)
        except (SyntaxError, ImportError, NameError, AttributeError, TypeError) as error:
            logger.result(f"skipped {name}: {type(error).__name__}: {error}")


class Lint:
    """The selected rules, configured for one project."""

    @staticmethod
    def _adopt(base: Facts, family: type[Family], *, empty: bool) -> None:
        """Adopt one family on a cursor of its own."""
        with base.db.cursor() as cursor:
            base.adopt(family.family, family.relations, empty=empty, cursor=cursor)

    @staticmethod
    def _adopted(db: Facts, derived: Sequence[type[Family]]) -> None:
        """Adopt the derived families, each on a cursor of its own once every family it reads is.

        Whether a family holds evidence, and the project's sources, are read first: both read the
        database, which the families then write from several threads.
        """
        empty = {family.family: not family.grounded(db) for family in derived}
        if any("source_files" in query.views() for f in derived for query in f.queries()):
            db.ensure(Query(sql="SELECT path FROM source_files"))
        named = {family.family: family for family in derived}
        order = TopologicalSorter(
            {
                name: _families_read(f.queries()) & named.keys() - {name}
                for name, f in named.items()
            }
        )
        order.prepare()
        with ThreadPoolExecutor(os.process_cpu_count()) as pool:
            running: dict[Future[None], str] = {}
            while order.is_active():
                for name in order.get_ready():
                    adopting = pool.submit(Lint._adopt, db, named[name], empty=empty[name])
                    running[adopting] = name
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                    order.done(running.pop(future))

    @staticmethod
    def _sql_of(name: str) -> list[Query]:
        """The SQL of the derived family `name`: its relations and the shared ones they call."""
        own = Family.defined[name].queries()
        return [*own, *(Query(sql=relation.query) for relation in needed(own))]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        raised: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Remove what this run cached and spilled."""
        if self.current:
            self.current.close()
        self.ducks.close()

    def __init__(
        self,
        root: Path,
        rules: Iterable[type[Rule]],
        *,
        memory: str = "2GB",
        families: Sequence[str] = (),
        fixing: bool = True,
    ) -> None:
        self.root, self.budget, self.fixing = root, Budget(memory=memory), fixing
        self.consumers = Consumers(root)
        self.current: Facts | None = None
        self.derived = {
            name for name, family in Family.defined.items() if family.verified or name in families
        }
        self.ducks = Ducks(budget=self.budget)
        self.project = ProjectConfig.of(root)
        self.rules = self._configured(rules)
        self.queries = [query for rule in self.rules for query in self._queries(rule)]
        self._planned()

    def _configured(self, rules: Iterable[type[Rule]]) -> list[Rule]:
        """The enabled rules, configured as the project says, for languages it holds."""
        present = {self.ducks.language(PurePosixPath(path).suffix) for path in listed(self.root)}
        return [
            rule
            for rule in (
                cls(**self.project.rule(cls.id).settings)
                for cls in rules
                if cls.id != _TYPE_CHECKED and self.project.rule(cls.id).enabled
            )
            if rule.language is None or rule.language in present
        ]

    def _consumed(self, rule: Rule, path: str, declared: str | None) -> bool:
        """Whether another repository of the workspace uses what a finding of `rule` declares.

        This project's facts cannot see such a use (consumers.py).
        """
        return declared is not None and (
            self.consumers.used(rule.id, path=path, declared=declared) is not None
        )

    def _derive(self, base: Facts, root: Path, derived: Sequence[type[Family]]) -> None:
        """Read the syntax trees and compute the derived families, each after those it reads."""
        if self.reads_trees:
            self.ducks.register(base, root)
        for family in derived:
            family.provide(base, root)
        define(
            base,
            [sql for family in derived for sql in family.relations.values()]
            + list(self.core.values()),
        )
        if self.core:
            base.adopt(Core.family, self.core)
        Lint._adopted(base, derived)

    def _families(self, queries: list[Query]) -> list[str]:
        """The derived families `queries` read, with those they read in turn.

        A family no SQL derives is left out, and `facts` skips the rules that read it.
        """
        found = {name for name in _families_read(queries) if name in self.derived}
        while more := {
            family for name in found for family in self._read_by(name) if family not in found
        }:
            found |= more
        return sorted(found)

    def _planned(self) -> None:
        """Plan the derived families, the core relations and whether the syntax trees are read."""
        wanted = [*self.queries, *(Query(sql=r.query) for r in needed(self.queries))]
        self.families = self._families(wanted)
        derived = [query for name in self.families for query in self._sql_of(name)]
        views = {view for query in [*wanted, *derived] for view in query.views()}
        self.core = Core.reading(views)
        self.reads_trees = bool(self.core) or any(view.startswith("ast_") for view in views)

    def _queries(self, rule: Rule) -> list[Query]:
        """The SQL `rule` runs: its findings and, when fixing, its fix."""
        return [query for query in (rule.rows(), rule.edits() if self.fixing else None) if query]

    def _read_by(self, name: str) -> set[str]:
        """The families the derived family `name` reads, which must be derived too."""
        read = _families_read(self._sql_of(name)) - {name}
        if stranger := next((f for f in sorted(read) if f not in self.derived), None):
            raise ValueError(
                f"derived family {name} reads {stranger}, which is not derived (no verified SQL "
                f"family of that name; pass --family {stranger} if one is being ported)"
            )
        return read

    def _repaired(self, base: Facts, rule: Rule) -> list[Repair]:
        """`rule`'s repairs: the edits of each finding, one group per finding it fixes."""
        accepts = self.project.rule(rule.id).accepts
        cursor = base.cursor(rule.language)
        cursor.execute(rule.finding_rows(), rule.rows().parameters)
        self._withheld(cursor, rule)
        places = {
            key: (path, line or 1, change)
            for key, path, line, change in cursor.execute(rule.changes()).fetchall()
        }
        return [
            Repair(group, **placed(rule, group, places.get(key)))
            for key, group in grouped(cursor, rule).items()
            if group and all(accepts(edit.file()) for edit in group)
        ]

    def _served(self, base: Facts, rule: Rule) -> bool:
        """Whether the fact base holds every view `rule` and its relations read.

        A rule of a family no SQL derives (Rust surfaces, TypeScript modules) is skipped with a
        warning.
        """
        try:
            _ensured(base, self._queries(rule))
        except ValueError as error:
            logger.warning("{} is skipped: {}", rule.id, error)
            return False
        return True

    def _unconsumed(self, rule: Rule, rows: Sequence[tuple]) -> list[tuple]:
        """`rows` as findings, less dead code that another repository of the workspace uses.

        The rows carry the usual four columns and the name the finding declares.
        """
        kept = [row[:4] for row in rows if not self._consumed(rule, row[0], row[4])]
        if len(kept) < len(rows):
            logger.debug(
                "{}: {} of {} findings dropped, other repositories import their names",
                rule.id,
                len(rows) - len(kept),
                len(rows),
            )
        return kept

    def _withheld(self, cursor: DuckDBPyConnection, rule: Rule) -> None:
        """Draw up no fix for a finding `findings` does not report.

        That is dead code another repository of the workspace uses.
        """
        if rule.id not in DECLARED:
            return
        named = f"SELECT finding_key, path, {DECLARED[rule.id]} FROM finding_rows"
        consumed = [
            key
            for key, path, declared in cursor.execute(named).fetchall()
            if self._consumed(rule, path, declared)
        ]
        cursor.execute(_DELETE_CONSUMED_FINDING_ROWS_SQL, {"keys": consumed})

    def applicable(self, db: Facts) -> list[Rule]:
        """The rules whose language the project holds."""
        return [
            rule for rule in self.rules if rule.language is None or rule.language in db.languages
        ]

    def facts(self, root: Path) -> Facts:
        """The fact base of `root`, with every relation and empty view the rules need.

        The syntax trees are read and the derived families computed, each after those it reads. It
        replaces the previous fact base, whose database is closed and deleted.
        """
        derived = [Family.defined[name] for name in self.families]
        if self.current:
            self.current.close()
        base = self.current = Facts(root, self.budget)
        if derived or self.reads_trees:
            self._derive(base, root, derived)
        Tool.register_all(base, root, self.queries)
        self.rules = [rule for rule in self.rules if self._served(base, rule)]
        install(base, self.applicable(base), fixes=self.fixing)
        _ensured(base, [query for rule in self.rules for query in self._queries(rule)])
        return base

    def findings(self, db: Facts) -> dict[str, list[tuple]]:
        """`(path, start_line, start_column, message)` rows per rule.

        Paths a rule excludes and vendored paths are dropped, asked of the project since `db` may
        be a fix's mirror.
        """

        def run(rule: Rule) -> list[tuple]:
            accepts = self.project.rule(rule.id).accepts
            declared = [DECLARED[rule.id]] if rule.id in DECLARED else []
            cursor = db.cursor(rule.language)
            try:
                rows = cursor.execute(rule.statement(*declared), rule.rows().parameters)
            except duckdb.Error as error:
                error.add_note(f"in the query of rule {rule.id}")
                raise
            kept = [row for row in rows.fetchall() if accepts(row[0])]
            return self._unconsumed(rule, kept) if declared else kept

        rules = self.applicable(db)
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            found = dict(zip([rule.id for rule in rules], pool.map(run, rules), strict=True))
        skipped = vendored(self.root, {row[0] for rows in found.values() for row in rows})
        return {
            rule: [row for row in rows if row[0] not in skipped] for rule, rows in found.items()
        }

    def flat(self, db: Facts) -> set[tuple]:
        """Every finding as `(path, line, column, rule, message)`, the shape a proof compares."""
        return {
            (row[0], row[1], row[2], rule, row[3])
            for rule, rows in self.findings(db).items()
            for row in rows
        }

    def repairs(self, db: Facts, applicability: Sequence[str]) -> dict[str, list[Repair]]:
        """Per fixable rule, the edits of each finding it repairs.

        Each finding's edits are one group saying what it changed.
        """
        if not self.fixing:
            raise ValueError("this Lint was built without fixes (`fixing=False`)")
        rules = [
            rule
            for rule in self.applicable(db)
            if rule.fixes() != "none" and rule.applicability in applicability
        ]
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            repaired = pool.map(lambda rule: self._repaired(db, rule), rules)
            found = dict(zip([rule.id for rule in rules], repaired, strict=True))
        files = {edit.file() for group in chain.from_iterable(found.values()) for edit in group}
        skipped = vendored(self.root, files)
        return {
            rule: [group for group in groups if not any(e.file() in skipped for e in group)]
            for rule, groups in found.items()
        }


def _families_read(queries: Iterable[Query]) -> set[str]:
    """The defined families whose views `queries` name.

    A view names its family as its prefix; `Core`, built from the tree itself, is not one.
    """
    known = {snake(name): name for name in Family.defined if name != Core.family}
    return {
        known[max((p for p in known if view.startswith(f"{p}_")), key=len)]
        for query in queries
        for view in query.views()
        if any(view.startswith(f"{p}_") for p in known)
    }


def selected(rule: Sequence[str]) -> list[type[Rule]]:
    load_catalog()
    if rule:
        return [Rule.defined[r] for r in rule]
    return [spec for name, spec in Rule.defined.items() if name != _TYPE_CHECKED]


def _ensured(base: Facts, queries: Sequence[Query]) -> None:
    """Make sure the fact base holds every view `queries` and the relations they call read."""
    for query in [*queries, *(Query(sql=r.query) for r in needed(queries))]:
        base.ensure(query)
