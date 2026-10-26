"""The shared SQL a rule reads (relations, functions, fact families), installed in a fact base."""

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence, Set
from functools import cache
from typing import TYPE_CHECKING, ClassVar

import duckdb
from patos import FrozenModel

from ..engine.facts import Query, Setting, snake

if TYPE_CHECKING:
    from pathlib import Path

    from ..engine.facts import Facts
    from .rules import Rule


class Evidenced(FrozenModel):
    """What a project must hold for a family's or a stored relation's rows to exist at all: a
    project without it has none, so the engine builds them empty (the columns, no row) without
    running their SQL.

    `evidence` names the languages (as `ast_files` lists them) of the files the rows come from,
    `evidence_imports` the modules (`cupy`, `patos.cuda`; a submodule counts) some Python file
    must import. Both empty: any project. When both are given, both must hold.
    """

    evidence: ClassVar[frozenset[str]] = frozenset()
    evidence_imports: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def grounded(cls, db: Facts) -> bool:
        """Whether the project in `db` holds the evidence."""
        return (not cls.evidence or bool(cls.evidence & db.held)) and (
            not cls.evidence_imports or db.imports(cls.evidence_imports)
        )


class Relation(Evidenced):
    """A shared relation over the fact views, the table macro `snake(ClassName)(fields...)`.

    A `stored` relation (no fields) is computed once per fact base, into each language's schema
    whose rules read it, and its macro reads those rows: for relations several rules share or a
    recursive query revisits. A class without a `query` is a base the relations of one kind share.
    """

    query: ClassVar[str]
    stored: ClassVar[bool] = False
    defined: ClassVar[dict[str, type[Relation]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        if "query" in cls.__dict__:
            Relation.defined[cls.name()] = cls

    @classmethod
    def _defined(cls, db: Facts, *, languages: Set[str | None]) -> None:
        """Create the macro, after a stored relation's rows (see `install`)."""
        if cls.stored:
            cls._stored(db, languages=languages)
        db.db.execute(cls.definition())

    @classmethod
    def _stored(cls, db: Facts, *, languages: Set[str | None]) -> None:
        """The rows of a stored relation in each schema of `languages`, empty without evidence.

        A schema-only copy in `main` lets the macro bind there when no reader is unlabeled.
        """
        body = cls.query if cls.grounded(db) else f"SELECT * FROM ({cls.query}) WHERE false"
        table = f"{cls.name()}__rows"
        for language in sorted(languages, key=str):
            db.cursor(language).execute(
                f"CREATE OR REPLACE TABLE {language or 'main'}.{table} AS {body}"
            )
        if None not in languages:
            db.db.execute(
                f"CREATE OR REPLACE TABLE main.{table} AS "
                f"SELECT * FROM {next(iter(languages))}.{table} LIMIT 0"
            )

    @classmethod
    def definition(cls) -> str:
        """The statement creating the macro."""
        body = f"SELECT * FROM {cls.name()}__rows" if cls.stored else cls.query
        arguments = ", ".join(cls.model_fields)
        return f"CREATE OR REPLACE MACRO {cls.name()}({arguments}) AS TABLE {body}"

    @classmethod
    def install(cls, db: Facts, *, languages: Set[str | None]) -> None:
        """Define the relation in `db` with the empty views its body names.

        A stored one's rows are computed first in every language of `languages` that reads it.
        """
        db.ensure(Query(sql=cls.query))
        try:
            cls._defined(db, languages=languages)
        except duckdb.Error as error:
            raise ValueError(f"relation {cls.__name__}: {error}") from error

    @classmethod
    def name(cls) -> str:
        return snake(cls.__name__)


class Function(FrozenModel):
    """A shared SQL expression, the scalar macro `snake(ClassName)(fields...)`: `class
    Plural(Function)` with fields `amount, noun` is called `plural(n, 'line')`."""

    expression: ClassVar[str]
    defined: ClassVar[dict[str, type[Function]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        if "expression" in cls.__dict__:
            Function.defined[snake(cls.__name__)] = cls

    @classmethod
    def create(cls, db: duckdb.DuckDBPyConnection) -> None:
        """Define the function in `db`."""
        db.execute(cls.definition())

    @classmethod
    def definition(cls) -> str:
        """The statement creating the macro."""
        arguments = ", ".join(cls.model_fields)
        return f"CREATE OR REPLACE MACRO {snake(cls.__name__)}({arguments}) AS {cls.expression}"


class Family(Evidenced):
    """One of mcmr's fact families derived in SQL instead of by its kernel: `relations` maps each
    relation its rules read, by its mcmr name and the identity relation first, to SQL over the
    providers' views (`ast_files` and `ast_nodes` from sitting_duck, other families' views,
    `source_files`) and the family's earlier relations by their view names, calling any shared
    `Function`. The engine derives a family that is `verified` (every rule reading it gives
    findings equal to mcmr's, or differences judged right; a family still being ported says it
    is not) and one a run names while it is ported. A family reads only families that are
    derived too: reading one that is not is an error, not a fallback to the kernel.

    A project without the family's evidence (see `Evidenced`) has no row in any relation."""

    family: ClassVar[str]
    relations: ClassVar[dict[str, str]]
    verified: ClassVar[bool] = True
    defined: ClassVar[dict[str, type[Family]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        if "family" in cls.__dict__:
            Family.defined[cls.family] = cls

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Create the tables the relations read beyond the syntax trees; by default none.

        Such tables are the directory tree or git history. Several families may share one, so a
        provider creates it once.
        """

    @classmethod
    def queries(cls) -> list[Query]:
        return [Query(sql=sql) for sql in cls.relations.values()]


def needed(queries: Iterable[Query]) -> list[type[Relation]]:
    """The relations `queries` call, directly or through other relations, each after its own."""
    found: dict[str, type[Relation]] = {}

    def visit(name: str) -> None:
        if name in found or name not in Relation.defined:
            return
        relation = Relation.defined[name]
        for inner in Query(sql=relation.query).macros():
            visit(inner)
        found[name] = relation

    for query in queries:
        for name in query.macros():
            visit(name)
    return list(found.values())


@cache
def _calls(text: str) -> frozenset[str]:
    """The names `text` calls, each as `name(` after a word boundary."""
    return frozenset(re.findall(r"\b(\w+)\s*\(", text))


def _called(sql: Sequence[str]) -> list[type[Function]]:
    """The shared functions `sql` calls, each after the functions its own body calls.

    Functions reached through other functions count too.
    """
    found: dict[str, type[Function]] = {}

    def visit(text: str) -> None:
        called = _calls(text)
        for name, function in Function.defined.items():
            if name not in found and name in called:
                found[name] = function
                visit(function.definition())
                found[name] = found.pop(name)

    for text in sql:
        visit(text)
    return list(found.values())


def define(db: Facts, sql: list[str]) -> None:
    """Create the shared functions `sql` calls, unless the fact base already has them.

    Each is created after those its body calls.
    """
    for function in _called(sql):
        if function not in db.defined:
            function.create(db.db)
            db.defined.add(function)


def install(db: Facts, rules: Iterable[Rule], *, fixes: bool = True) -> None:
    """Create in `db` the shared functions and relations `rules` call.

    Their fixes' calls count too with `fixes`, and each relation comes after those it calls, with
    the empty views their bodies name. A stored relation's rows are computed in every language that
    reads it (a schema-only copy in `main` lets its macro bind there), or built empty where the
    project lacks its evidence.
    """
    readers: defaultdict[type[Relation], set[str | None]] = defaultdict(set)
    queries, statements = [], []
    for rule in rules:
        mine = [query for query in (rule.rows(), rule.edits() if fixes else None) if query]
        queries += mine
        statements.append(rule.statement())
        for relation in needed(mine):
            readers[relation].add(rule.language)
    define(db, [*statements, *(q.sql for q in queries), *(r.query for r in needed(queries))])
    for relation in needed(queries):
        relation.install(db, languages=readers[relation])
