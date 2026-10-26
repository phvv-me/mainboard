"""The fact base rules query: one DuckDB database holding every fact family as plainly named views,
and each language's narrowing of them.

A family arrives as SQL relations over what the database already reads (`adopt`), its identity
relation first (one row per fact). Every relation becomes a view `<family>_<relation>` in snake
case, without the `Fact` suffix and with dots as underscores: `symbol_facts`, `symbol_symbols`,
`symbol_symbols_reference_references`, `syntax_nodes`. Every language present gets a schema of
the same views narrowed to its facts, so a cursor whose `search_path` is `python, main` reads only
Python facts with the same SQL. Shared helpers are table macros, which resolve view names in the
caller's language.
"""

import threading
from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

import duckdb
from patos import FrozenModel

from .budget import Budget
from .files import listed
from .sql import Query, snake

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from ...catalog import Function


# A key column naming its parent's column of another name.
_KEYS = {"function_id": "entity_id"}


class _Narrowing(FrozenModel):
    """One family's relations as one language sees them, each narrowed to that language's facts.

    The identity narrows by its `language`, a relation with `fact_id` by it, and any other
    through the key it shares with a relation already narrowed (`call_id`, `class_id`,
    `function_id` ...).

    prefix: the family's view prefix (`call_fact` for `CallFact`).
    alone: whether the family holds this one language and no unlabeled fact, so nothing narrows.
    columns: each relation's columns.
    """

    language: str
    prefix: str
    identity: str
    alone: bool
    columns: dict[str, frozenset[str]]

    def clause(self, table: str, *, relation: str, narrowed: Mapping[str, str]) -> str | None:
        """The `WHERE` narrowing `relation`, None while its parent is pending.

        narrowed: the relations already narrowed, by name, with their tables.
        """
        if self.alone:
            return ""
        if table == self.identity:
            return f"WHERE language = '{self.language}'"
        if "fact_id" in self.columns[relation]:
            return (
                f"WHERE fact_id IN (SELECT fact_id FROM {self.identity} "
                f"WHERE language = '{self.language}')"
            )
        # A relation keyed by `file_id` narrows through it, before any other `*_id` column,
        # which may hold node ids (`class_id`, `parent_id`) that no parent relation shares.
        keys = sorted(
            (c for c in self.columns[relation] if c.endswith("_id")),
            key=lambda c: (c != "file_id", c),
        )
        found = (
            (key, parent, target)
            for key in keys
            for parent in narrowed
            if (target := key if key in self.columns[parent] else _KEYS.get(key))
            in self.columns[parent]
        )
        return next(
            (
                f"WHERE {key} IN (SELECT {target} "
                f"FROM {self.language}.{self.prefix}_{snake(parent)})"
                for key, parent, target in found
            ),
            None,
        )

    def views(self, tables: Mapping[str, str]) -> Iterator[tuple[str, str, str]]:
        """Each relation, its table and its `WHERE`, in an order where a parent comes first.

        Each round takes the first pending relation whose parent is narrowed; once none is, the
        rest are not narrowed at all.
        """
        pending = dict(tables)
        narrowed: dict[str, str] = {}
        while pending:
            ready = next(
                (
                    (relation, table, clause)
                    for relation, table in pending.items()
                    if (clause := self.clause(table, relation=relation, narrowed=narrowed))
                    is not None
                ),
                None,
            )
            if ready is None:
                yield from ((relation, table, "") for relation, table in pending.items())
                return
            relation, table, _ = ready
            yield ready
            narrowed[relation] = table
            del pending[relation]


class Facts:
    """The views rules read, over one DuckDB database of one project's facts.

    `languages` are those the facts name and the project has source files in (a family's facts
    may name a language its project merely configures). `source_files` (path, content) is read
    from `root` the first time a query names it.
    """

    SUFFIXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "python": (".py", ".pyi"),
        "rust": (".rs",),
        "swift": (".swift",),
        "objective_c": (".m",),
        "objective_cpp": (".mm",),
        "typescript": (".ts", ".tsx", ".mts", ".cts"),
        "c": (".c", ".h"),
        "cpp": (".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".h"),
        "cuda": (".cu", ".cuh"),
    }

    @cached_property
    def held(self) -> frozenset[str]:
        """The languages `ast_files` lists a file of."""
        rows = self.db.execute("SELECT DISTINCT language FROM ast_files").fetchall()
        return frozenset(language for (language,) in rows)

    @cached_property
    def imported(self) -> frozenset[str]:
        """The dotted names of the modules the project imports by absolute name, as written.

        Both `import a.b` and `from a.b import c` count. A name `from m import n as x` binds is
        counted too, so the set holds every module imported and possibly more.
        """
        rows = self.db.execute(
            """
            SELECT name FROM ast_nodes
            WHERE type = 'import_from_statement' AND NOT starts_with(name, '.')
            UNION
            SELECT d.name FROM ast_nodes d
            JOIN ast_nodes p ON p.fact_id = d.fact_id AND p.node_id = d.parent_id
            WHERE d.type = 'dotted_name' AND p.type IN ('import_statement', 'aliased_import')
            """
        ).fetchall()
        return frozenset(name for (name,) in rows)

    @cached_property
    def present(self) -> set[str]:
        """The languages the project holds source files in."""
        if self.root is None:
            return set()
        names = listed(self.root)
        return {
            language
            for language, suffixes in self.SUFFIXES.items()
            if any(name.lower().endswith(suffixes) for name in names)
        }

    def __init__(self, root: Path | None = None, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()
        self.file = self.budget.folder / f"facts-{uuid4().hex[:12]}.duckdb"
        self.db = self.budget.connect(self.file)
        self.root = root
        self.families: dict[str, str] = {}
        self.identities: list[str] = []
        self.languages: set[str] = set()
        self.known: set[str] = set()
        self.defined: set[type[Function]] = set()
        # Families adopted at once share the catalog bookkeeping below; their tables are not.
        self.lock = threading.RLock()

    def _identify(self, family: str, tables: dict[str, str]) -> None:
        """Record the family and its identity relation, then narrow each relation per language.

        The identity is `facts`, else the first relation. Naming it lets a derived identity come
        after the relations it reads.
        """
        prefix = snake(family)
        self.families[prefix] = family
        identity = tables.get("facts", tables[next(iter(tables))])
        self.identities.append(identity)
        if self.db.execute(
            f"SELECT 1 FROM (DESCRIBE {identity}) WHERE column_name = 'language'"
        ).fetchone():
            self._narrow(prefix, identity=identity, tables=tables)

    def _narrow(self, prefix: str, *, identity: str, tables: dict[str, str]) -> None:
        """Give each language's schema every relation narrowed to that language's facts.

        The identity narrows by its `language`, a relation with `fact_id` by it, and any other
        through the key it shares with a relation already narrowed (`call_id`, `class_id`,
        `function_id` ...). A family holding one language and no unlabeled fact needs no narrowing
        at all.
        """
        present = self.db.execute(
            f"SELECT language, list_distinct(list(lower(regexp_extract(path, '\\.[^./]+$')))) "
            f"FROM {identity} GROUP BY language"
        ).fetchall()
        self.languages |= {
            language
            for language, suffixes in present
            if language is not None
            and (set(suffixes) & set(self.SUFFIXES.get(language, ())) or language in self.present)
        }
        columns = {
            relation: frozenset(
                c
                for (c,) in self.db.execute(
                    f"SELECT column_name FROM (DESCRIBE {table})"
                ).fetchall()
            )
            for relation, table in tables.items()
        }
        for language in [language for language, _ in present if language is not None]:
            self.db.execute(f"CREATE SCHEMA IF NOT EXISTS {language}")
            narrowing = _Narrowing(
                language=language,
                prefix=prefix,
                identity=identity,
                alone=len(present) == 1,
                columns=columns,
            )
            for relation, table, clause in narrowing.views(tables):
                self.db.execute(
                    f"CREATE VIEW {language}.{prefix}_{snake(relation)} "
                    f"AS SELECT * FROM {table} {clause}"
                )

    def _read_sources(self) -> None:
        """`source_files` (path, content): every project file a fact names, read from the root.

        A fact names a file in any `path` or `*_path` column.
        """
        if self.root is None:
            raise ValueError("these facts were registered without a root to read sources from")
        columns = self.db.execute(
            "SELECT table_name, column_name FROM duckdb_columns() WHERE schema_name = 'main' "
            "AND data_type = 'VARCHAR' AND (column_name = 'path' OR suffix(column_name, '_path')) "
            "AND table_name IN (SELECT table_name FROM duckdb_tables())"
        ).fetchall()
        named = " UNION ".join(
            f'SELECT "{column}" AS path FROM "{table}"' for table, column in columns
        )
        paths = sorted(
            path
            for (path,) in self.db.execute(
                f"SELECT DISTINCT path FROM ({named}) WHERE path IS NOT NULL"
            ).fetchall()
            if (self.root / path).is_file()
        )
        self.db.execute(
            "CREATE TABLE main.source_files AS SELECT path, content FROM "
            "(SELECT unnest($paths) AS path, unnest($files) AS file) "
            "JOIN read_text($files) r ON r.filename = file",
            {"paths": paths, "files": [str(self.root / path) for path in paths]},
        )

    def _view(self, family: str, *, relation: str, table: str) -> None:
        """Name `table` `<family>_<relation>`."""
        view = f"{snake(family)}_{snake(relation)}"
        self.db.execute(f"CREATE VIEW main.{view} AS SELECT * FROM {table}")
        self.known.add(view)

    def adopt(
        self,
        family: str,
        relations: Mapping[str, str],
        *,
        empty: bool = False,
        cursor: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Store one family of relations, SQL over what the database reads, identity first.

        The database reads tables a provider loaded, other families' views, and this family's
        earlier relations by their view names. An `empty` family keeps the columns of its relations
        and no row. The tables are computed on `cursor` (the database's own connection when None),
        so families no other one reads can be adopted at once, each on its own cursor.
        """
        tables = {}
        for name, select in relations.items():
            with self.lock:
                self.ensure(Query(sql=select))
            tables[name] = f'"{family}/{name}"'
            body = f"SELECT * FROM ({select}) WHERE false" if empty else select
            (cursor or self.db).execute(f"CREATE TABLE {tables[name]} AS {body}")
            with self.lock:
                self._view(family, relation=name, table=tables[name])
        with self.lock:
            self._identify(family, tables)

    def close(self) -> None:
        """Close the database and delete its file."""
        self.db.close()
        for file in (self.file, self.file.with_suffix(".duckdb.wal")):
            file.unlink(missing_ok=True)

    def cursor(self, language: str | None) -> duckdb.DuckDBPyConnection:
        """A cursor reading `language`'s facts through the plain view names, all facts for None."""
        cursor = self.db.cursor()
        if language is not None:
            cursor.execute(f"SET search_path = '{language},main'")
        return cursor

    def ensure(self, query: Query) -> None:
        """Create, empty, every view `query` names that no fact filled in this project.

        `source_files` is read when the query names it. Names are only ever added to the fact base,
        so the catalog is listed again only when a query names one the last listing lacked.
        """
        wanted = query.views()
        if wanted <= self.known:
            return
        self.known = {
            name
            for (name,) in self.db.execute(
                "SELECT view_name FROM duckdb_views() UNION SELECT table_name FROM duckdb_tables()"
                " UNION SELECT function_name FROM duckdb_functions()"
            ).fetchall()
        }
        for view in wanted - self.known:
            self.known.add(view)
            if view == "source_files":
                self._read_sources()
                continue
            prefixes = [p for p in self.families if view.startswith(f"{p}_")]
            if not prefixes:
                raise ValueError(f"`{view}` is no view of a registered fact family")
            self.db.execute(
                f"CREATE VIEW main.{view} AS SELECT * EXCLUDE (relation) "
                f'FROM "{self.families[max(prefixes, key=len)]}/records" WHERE false'
            )

    def imports(self, modules: Iterable[str]) -> bool:
        """Whether the project imports one of `modules` or a submodule of one."""
        return any(
            name == module or name.startswith(f"{module}.")
            for module in modules
            for name in self.imported
        )
