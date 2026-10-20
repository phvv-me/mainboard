"""What a rule is: a named SQL relation over the fact views whose rows are findings, with its
settings as typed fields and its fix as SQL over those findings.

```python
class PackageDepth(Rule, id="ALL-FILE0002"):
    \"\"\"Directory levels between a directory and the source root above it.\"\"\"

    maximum: int = 5
    framework_roots: tuple[str, ...] = ()

    query = \"\"\"
        SELECT *, source_depth AS value FROM directory_facts WHERE source_depth > $maximum
    \"\"\"
    message = "package depth is {value} for `{fact_id}`"
```

Fields are settings: a project overrides them, pydantic validates them, and the query reads them
as `$name`. `query` yields one row per finding with `path`, `start_line`, `start_column` (and
`end_line`, `end_column` when the fix edits the finding's own span), plus the columns `message`
templates, or a `message` column itself.

`fix` is SQL over `finding`, the query's rows, yielding one row per edit (`Edit`): a span and the
`text` replacing it (empty text deletes, an empty span inserts), an `import` the text needs, or a
file `operation`. A fix the facts cannot compute in SQL is a method `repair(self, hit)` returning
edits instead. `Relation`s are the shared building blocks: `class PlacedSymbols(Relation)` is the
table macro `placed_symbols()`, whose views resolve in the calling rule's language and whose
fields are its parameters.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
from functools import cache
from string import Formatter
from typing import TYPE_CHECKING, ClassVar, Literal

import duckdb
from patos import FrozenModel

from .edits import Edit
from .facts import Query, Setting, snake
from .findings import Hit

if TYPE_CHECKING:
    from pathlib import Path

    from .facts import Facts

type Applicability = Literal["safe", "review", "display", "none"]

LANGUAGES = {
    "PY": "python",
    "RS": "rust",
    "TS": "typescript",
    "C": "c",
    "CPP": "cpp",
    "CU": "cuda",
    "SW": "swift",
}


_KINDS = {
    "replace": ("rewrote", "span"),
    "insert": ("inserted", "block"),
    "delete": ("deleted", "span"),
    "move": ("moved", "block"),
    "remove": ("removed", "file"),
    "rename": ("renamed", "path"),
    "import": ("added", "import"),
}


class Repair(list[Edit]):
    """The edits one finding's fix makes, with where the finding stood and what the fix changed
    (`change`, a line an agent can read to pick up from the fix)."""

    def __init__(self, edits: Iterable[Edit], *, path: str, line: int, change: str) -> None:
        super().__init__(edits)
        self.path, self.line, self.change = path, line, change


class Example(FrozenModel):
    """A snippet the rule must flag (`bad`) and what its fix must turn it into (`good`; None when
    the fix removes the file), held in the project file `path` (`lint test` picks one from the
    rule's language when empty) beside the project's other files, `context` (path: text)."""

    bad: str
    good: str | None = ""
    path: str = ""
    context: dict[str, str] = {}


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
    def name(cls) -> str:
        return snake(cls.__name__)

    @classmethod
    def definition(cls) -> str:
        """The statement creating the macro."""
        body = f"SELECT * FROM {cls.name()}__rows" if cls.stored else cls.query
        arguments = ", ".join(cls.model_fields)
        return f"CREATE OR REPLACE MACRO {cls.name()}({arguments}) AS TABLE {body}"


class Function(FrozenModel):
    """A shared SQL expression, the scalar macro `snake(ClassName)(fields...)`: `class
    Plural(Function)` with fields `amount, noun` is called `plural(n, 'line')`."""

    expression: ClassVar[str]
    defined: ClassVar[dict[str, type[Function]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        Function.defined[snake(cls.__name__)] = cls

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
    findings equal to mcmr's, or differences judged right) and one a run names while it is
    ported. A family reads only families that are derived too: reading one that is not is an
    error, not a fallback to the kernel.

    A project without the family's evidence (see `Evidenced`) has no row in any relation."""

    family: ClassVar[str]
    relations: ClassVar[dict[str, str]]
    verified: ClassVar[bool] = False
    defined: ClassVar[dict[str, type[Family]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        if "family" in cls.__dict__:
            Family.defined[cls.family] = cls

    @classmethod
    def queries(cls) -> list[Query]:
        return [Query(sql=sql) for sql in cls.relations.values()]

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Create the tables the relations read beyond the syntax trees (the directory tree, git
        history); by default there are none. Several families may share one, so a provider
        creates it once."""


class RuleProvenance(FrozenModel):
    """A dated guidance review, separate from the toolchain actually used for analysis."""

    reviewed_on: date
    method: str
    review_context: str
    inspected: tuple[str, ...]
    limits: str


class Rule(FrozenModel):
    """A relation whose rows are findings; its fields are the settings `query` reads as `$name`."""

    id: ClassVar[str]
    applicability: ClassVar[Applicability]
    query: ClassVar[str]
    message: ClassVar[str | None] = None
    fix: ClassVar[str | None] = None
    change: ClassVar[str | None] = None
    examples: ClassVar[tuple[Example, ...]] = ()
    guidance: ClassVar[str] = ""
    references: ClassVar[tuple[str, ...]] = ()
    provenance: ClassVar[RuleProvenance | None] = None
    defined: ClassVar[dict[str, type[Rule]]] = {}

    def __init_subclass__(cls, *, id: str = "", applicability: Applicability = "none") -> None:
        """A subclass without an `id` is a base its family's rules share, not a rule."""
        super().__init_subclass__()
        if id:
            cls.id, cls.applicability = id, applicability

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        if "id" in cls.__dict__:
            Rule.defined[cls.id] = cls

    @classmethod
    def summary(cls) -> str:
        """The docstring's first line."""
        return (cls.__doc__ or "").strip().splitlines()[0]

    @property
    def language(self) -> str | None:
        """The language the id's prefix names (`PY-` Python ...); `ALL-` rules read every one."""
        return LANGUAGES.get(self.id.split("-", 1)[0])

    def sql(self) -> str:
        """The finding query; a rule composing its SQL from its settings overrides this."""
        return self.query

    def parameters(self) -> dict[str, Setting]:
        """What the SQL reads as `$name`: the settings, and what a rule derives from them or
        keeps as a constant (it extends this)."""
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in self.model_dump().items()
        }

    def rows(self) -> Query:
        """The finding query with the parameters it names bound."""
        sql = self.sql()
        return Query(
            sql=sql,
            parameters={
                name: value
                for name, value in self.parameters().items()
                if re.search(rf"\${name}\b", sql)
            },
        )

    def statement(self, *columns: str) -> str:
        """The query the check runs: placement and message, one row per finding, then `columns`
        (SQL expressions over the finding's row)."""
        extra = "".join(f", {column}" for column in columns)
        return (
            f"SELECT path, start_line, start_column, {self._message()} AS message{extra} "
            f"FROM ({self.sql()})"
        )

    def edits(self, found: str | None = None) -> Query | None:
        """The fix's edit rows over this rule's findings, when the fix is SQL; every `finding`
        row carries a `finding_key`, and edit rows sharing one are applied together or not at
        all. `found` names a table `finding_rows()` filled, which `finding` then reads instead
        of running the query again."""
        if (declared := type(self).fix) is None:
            return None
        fix = declared.strip()
        finding = (
            f"finding AS (SELECT * FROM {found})"
            if found
            else f"finding AS (SELECT *, row_number() OVER () AS finding_key FROM ({self.sql()}))"
        )
        # A fix with its own common table expressions joins `finding` to them.
        opening = re.match(r"(?is)with(\s+recursive)?\s+", fix)
        sql = (
            f"WITH{opening.group(1) or ''} {finding}, {fix[opening.end() :]}"
            if opening
            else f"WITH {finding} {fix}"
        )
        return Query(
            sql=sql,
            parameters={
                name: value
                for name, value in self.parameters().items()
                if re.search(rf"\${name}\b", sql)
            },
        )

    def finding_rows(self) -> str:
        """The statement filling the temporary table `finding_rows` with the numbered findings,
        which `edits("finding_rows")` and `changes()` both read."""
        return (
            "CREATE OR REPLACE TEMP TABLE finding_rows AS "
            f"SELECT *, row_number() OVER () AS finding_key FROM ({self.sql()})"
        )

    def changes(self) -> str:
        """The statement giving each numbered finding's key, place and change text (NULL when the
        rule states none)."""
        return (
            f"SELECT finding_key, path, start_line, {self._change()} AS change FROM finding_rows"
        )

    @classmethod
    def fixes(cls) -> Literal["none", "sql", "method"]:
        """How the rule fixes: SQL over its findings, a `repair` method, or not at all."""
        if cls.fix is not None:
            return "sql"
        return "method" if cls.repair is not Rule.repair else "none"

    def repair(self, hit: Hit) -> Edit | Iterable[Edit] | None:
        """The edits a Python fix makes for one finding; a rule fixing in Python overrides it."""
        return None

    def repairs(self, hit: Hit) -> list[Edit]:
        """The edits `repair` makes for one finding."""
        made = self.repair(hit)
        return [made] if isinstance(made, Edit) else list(made or ())

    def describe(self, edits: Iterable[Edit]) -> str:
        """What a fix without a `change` template did: the kinds of edit it made, and the rule
        by its name in words."""
        kinds = Counter(_KINDS[edit.kind] for edit in edits)
        made = ", ".join(f"{verb} {n} {noun}{'s' * (n > 1)}" for (verb, noun), n in kinds.items())
        return f"{made} ({snake(type(self).__name__).replace('_', ' ')})"

    def _change(self) -> str:
        """The change template as one SQL expression, a missing value read as empty text."""
        if self.change is None:
            return "CAST(NULL AS VARCHAR)"
        parts = self._parts(self.change)
        return f"concat({', '.join(parts)})" if parts else "''"

    def _message(self) -> str:
        """The template as one SQL expression: `{column}` reads the row, `{setting}` is the
        setting's value."""
        if self.message is None:
            return "message"
        return " || ".join(self._parts(self.message)) or "''"

    def _parts(self, template: str) -> list[str]:
        """The SQL expressions whose concatenation is `template`: `{column}` reads the row,
        `{setting}` is the setting's value."""
        settings = self.model_dump()

        def quoted(text: str) -> str:
            return "'" + text.replace("'", "''") + "'"

        return [
            part
            for literal, field, _, _ in Formatter().parse(template)
            for part in (
                quoted(literal) if literal else "",
                quoted(str(settings[field]))
                if field in settings
                else f'CAST("{field}" AS VARCHAR)'
                if field
                else "",
            )
            if part
        ]


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


def _called(sql: list[str]) -> list[type[Function]]:
    """The shared functions `sql` calls, directly or through other functions, each after the
    functions its own body calls."""
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
    """Create the shared functions `sql` calls, each after those its body calls, unless the fact
    base has them."""
    for function in _called(sql):
        if function not in db.defined:
            db.db.execute(function.definition())
            db.defined.add(function)


def install(db: Facts, rules: Iterable[Rule], *, fixes: bool = True) -> None:
    """Create in `db` the shared functions and relations `rules` call (their fixes' too, with
    `fixes`), each relation after those it calls, with the empty views their bodies name; a
    stored relation's rows are computed in every language that reads it (a schema-only copy in
    `main` lets its macro bind there), or built empty where the project lacks its evidence."""
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
        db.ensure(Query(sql=relation.query))
        try:
            if relation.stored:
                body = (
                    relation.query
                    if relation.grounded(db)
                    else f"SELECT * FROM ({relation.query}) WHERE false"
                )
                for language in sorted(readers[relation], key=str):
                    db.cursor(language).execute(
                        f"CREATE OR REPLACE TABLE {language or 'main'}."
                        f"{relation.name()}__rows AS {body}"
                    )
                if None not in readers[relation]:
                    db.db.execute(
                        f"CREATE OR REPLACE TABLE main.{relation.name()}__rows AS "
                        f"SELECT * FROM {next(iter(readers[relation]))}."
                        f"{relation.name()}__rows LIMIT 0"
                    )
            db.db.execute(relation.definition())
        except duckdb.Error as error:
            raise ValueError(f"relation {relation.__name__}: {error}") from error
