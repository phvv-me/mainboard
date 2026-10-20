"""The fact base rules query: one DuckDB database holding every fact family as plainly named views,
and each language's narrowing of them.

A family arrives as Arrow relations: its identity relation first (one row per fact), a generic
`records` relation whose `relation` column names each nested record path, and any specialized
ones. Every relation becomes a view `<family>_<relation>` in snake case, without the `Fact`
suffix and with dots as underscores: `symbol_facts`, `symbol_symbols`,
`symbol_symbols_reference_references`, `syntax_nodes`. Every language present gets a schema of
the same views narrowed to its facts, so a cursor whose `search_path` is `python, main` reads only
Python facts with the same SQL. Shared helpers are table macros, which resolve view names in the
caller's language.
"""

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=lists the project's files with git's fixed argv since=2026-10-13
import tempfile
from collections.abc import Mapping, Sequence
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

import duckdb
import sqlglot
from pathspec import GitIgnoreSpec
from patos import FrozenModel
from pydantic import Field
from sqlglot import exp

from ..core.errors import NoWorkspace
from ..core.project import Project
from ..manifest.loading import load

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pyarrow as pa

    from .catalog import Function


ANALYSIS_TOOL = "mb"

UNSIGNED: dict[str, str | bool | int | float | list[str]] = {"allow_unsigned_extensions": "true"}

# Sequences and mappings, which accept a `list[str]` where a list of settings is read.
type Setting = str | int | float | bool | None | Sequence[Setting] | Mapping[str, Setting]


def listed(root: Path) -> list[str]:
    """The project's files git does not ignore, submodules' files included, relative to `root`;
    every file under a root outside git (a scratch mirror). Files the workspace's `[lint]` table
    excludes (references, venue style bundles, datasets) and vendored ones are left out."""
    present = project_files(root)
    skipped = vendored(root, present)
    excluded = lint_excludes(root)
    return [name for name in present if name not in skipped and not excluded(name)]


def project_files(root: Path) -> list[str]:
    """The same source listing before lint/vendor exclusions, for inventories that account for
    excluded code too. Ignored untracked/build files are not part of the git listing."""
    # The user's global excludes file is left out, so every machine lists the same files.
    git = ["git", "-C", str(root), "-c", "core.excludesFile=/dev/null", "ls-files", "-z"]
    try:
        names = [
            name
            for flags in (["--cached", "--recurse-submodules"], ["--others", "--exclude-standard"])
            for name in subprocess.run([*git, *flags], capture_output=True, check=True)
            .stdout.decode()
            .split("\0")
        ]
    except subprocess.CalledProcessError:
        names = []
        for folder, folders, files in os.walk(root):
            folders[:] = [
                name for name in folders if name not in {".git", "__pycache__", ".mainboard"}
            ]
            names += [os.path.relpath(os.path.join(folder, name), root) for name in files]
    return [
        name
        for name in dict.fromkeys(names)
        if name and os.path.lexists(root / name) and not (root / name).is_dir()
    ]


@cache
def lint_excludes(root: Path) -> Callable[[str], bool]:
    """Whether a file, named relative to `root`, is one the workspace's `[lint]` table leaves
    alone: `exclude`, the analysis tool's own `exclude` and the generated-state folders, the very
    patterns `mainboard lint` reads, from the topmost workspace above `root` (a project with a
    manifest of its own is linted under its parent's). Nothing is excluded outside a workspace."""
    project = Project()
    workspace = None
    while True:
        try:
            workspace = project.find_root(workspace.parent if workspace else root.resolve())
        except NoWorkspace:
            break
    if workspace is None:
        return lambda name: False
    table = load(project.manifest(workspace)).lint
    tool = table.tools.get(ANALYSIS_TOOL)
    generated = (f"/{name}/" for name in project.out_dirs)
    spec = GitIgnoreSpec.from_lines([*generated, *table.exclude, *(tool.exclude if tool else ())])
    prefix = root.resolve().relative_to(workspace).as_posix()
    return lambda name: spec.match_file(name if prefix == "." else f"{prefix}/{name}")


def vendored(root: Path, paths: Iterable[str]) -> set[str]:
    """The `paths` (files or folders, relative to `root`) that git marks `linguist-vendored`, the
    standard mark for third-party code: analysis neither reads nor reports them. Outside git,
    none."""
    return marked(root, paths, "linguist-vendored")


def marked(root: Path, paths: Iterable[str], attribute: str) -> set[str]:
    """The paths git positively marks with ATTRIBUTE; none outside git."""
    if not (paths := list(paths)):
        return set()
    run = subprocess.run(
        ["git", "-C", str(root), "check-attr", "-z", "--stdin", attribute],
        input="".join(f"{path}\0" for path in paths).encode(),
        capture_output=True,
        check=False,
    )
    if run.returncode:
        return set()
    fields = run.stdout.decode().split("\0")
    return {
        path
        for path, value in zip(fields[0::3], fields[2::3], strict=False)
        if value not in ("unspecified", "unset", "false")
    }


class SqlNames:
    """The views and the table macros each SQL text names: the views without the text's own common
    table expressions, the macros as `name()` in a `FROM`.

    Parsing is the cost, and the answer depends only on the text and sqlglot's version, so every
    text is parsed once per process and once per machine: the answers are kept in a file keyed by
    the text's digest, which a changed text simply misses.
    """

    FILE: ClassVar[Path] = (
        Path.home() / f".cache/mainboard/analysis/sql-names-sqlglot-{sqlglot.__version__}.json"
    )

    def __init__(self) -> None:
        self.stored = self._read()
        self.parsed: dict[str, list[list[str]]] = {}

    @classmethod
    def _read(cls) -> dict[str, list[list[str]]]:
        try:
            return json.loads(cls.FILE.read_text())
        except OSError, json.JSONDecodeError:
            return {}

    def of(self, sql: str) -> tuple[frozenset[str], frozenset[str]]:
        """The views and the macros `sql` names."""
        key = hashlib.blake2b(sql.encode(), digest_size=16).hexdigest()
        if (names := self.stored.get(key)) is None:
            if not self.parsed:
                atexit.register(self.save)
            names = self.stored[key] = self.parsed[key] = self._parse(sql)
        views, macros = names
        return frozenset(views), frozenset(macros)

    @staticmethod
    def _parse(sql: str) -> list[list[str]]:
        tree = sqlglot.parse_one(sql, read="duckdb")
        local = {cte.alias for cte in tree.find_all(exp.CTE)}
        tables = list(tree.find_all(exp.Table))
        return [
            sorted({table.name for table in tables if table.name} - local),
            sorted({table.this.name for table in tables if isinstance(table.this, exp.Anonymous)}),
        ]

    def save(self) -> None:
        """Add this process's parses to the file, whole or not at all (processes may share it)."""
        self.FILE.parent.mkdir(parents=True, exist_ok=True)
        merged = self._read() | self.parsed
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.FILE.parent, suffix=".tmp", delete=False
        ) as handle:
            json.dump(merged, handle)
        Path(handle.name).replace(self.FILE)


@cache
def _names(sql: str) -> tuple[frozenset[str], frozenset[str]]:
    return NAMES.of(sql)


NAMES = SqlNames()


def snake(name: str) -> str:
    """`ImportBindingFact` -> `import_binding`, `symbols.reference` -> `symbols_reference`."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return words.removesuffix("_Fact").replace(".", "_").lower()


class Query(FrozenModel):
    """A rule's SQL over the fact views, its settings bound as `$name` parameters."""

    sql: str
    parameters: dict[str, Setting] = {}

    def views(self) -> frozenset[str]:
        """The views the SQL names, its own common table expressions and macro calls aside."""
        return _names(self.sql)[0]

    def macros(self) -> frozenset[str]:
        """The table macros the SQL calls, `name()` in a `FROM`."""
        return _names(self.sql)[1]


class Budget(FrozenModel):
    """What one run's DuckDB work may take, in a folder of its own that goes with the run.

    DuckDB's defaults (80% of memory, a spill of up to 90% of free disk into `.tmp` under the
    working directory, the monorepo root under `mainboard run`) let a few concurrent runs swap
    the machine to a halt. Every setting goes in by `SET`: a cap given to `connect(config=)` is
    reported but not enforced (duckdb/duckdb-python#667). Values measured in
    scratchpad/mcmr-mb/DUCKDB-RESOURCES.md and since: at 2 GB a full check of
    research/reproducibility peaks under 6 GB of process memory (most of it outside DuckDB's
    buffer manager: mcmr's kernel and Arrow copies), at about 3% of the speed of 3 GB.
    """

    memory: str = "2GB"
    spill: str = "2GB"
    threads: int = 8
    folder: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / f"mb-run-{uuid4().hex[:12]}"
    )

    def connect(self, file: Path | None = None) -> duckdb.DuckDBPyConnection:
        """A fact base in `file` (in memory for None): its tables stay compressed on disk and
        leave memory without spilling (an in-memory base of 4.7 GB is a 1 GB file), at about 5% of
        the speed. Unsigned extensions load, which a self-built `sitting_duck` is until the
        community repository ships it for this DuckDB; this is the one flag `connect` can take."""
        if file:
            file.parent.mkdir(parents=True, exist_ok=True)
        db = duckdb.connect(str(file or ":memory:"), config=UNSIGNED)
        for statement in self._statements(memory_limit=self.memory, threads=self.threads):
            db.execute(statement)
        return db

    def script(self) -> str:
        """SET statements for the sitting_duck parse in the DuckDB command line."""
        return "".join(
            f"{statement};\n"
            for statement in self._statements(memory_limit="1GB", max_temp_directory_size="1GB")
        )

    def _statements(self, **settings: str | int) -> list[str]:
        """`settings` with this run's spill folder (created: DuckDB makes only its last
        component) and spill cap."""
        (self.folder / "spill").mkdir(parents=True, exist_ok=True)
        merged = {
            "max_temp_directory_size": self.spill,
            **settings,
            "temp_directory": str(self.folder / "spill"),
        }
        return [f"SET {name} = '{value}'" for name, value in merged.items()]

    def close(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)


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
    KEYS: ClassVar[dict[str, str]] = {"function_id": "entity_id"}

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

    def close(self) -> None:
        """Close the database and delete its file."""
        self.db.close()
        for file in (self.file, self.file.with_suffix(".duckdb.wal")):
            file.unlink(missing_ok=True)

    def register(self, family: str, relations: Mapping[str, pa.Table]) -> None:
        """Store one family's Arrow relations as tables and name each through a view."""
        tables = {}
        for name, arrow in relations.items():
            self.db.register(
                "arrival",
                arrow.rename_columns([column.replace(".", "_") for column in arrow.column_names]),
            )
            self.db.execute(f'CREATE TABLE "{family}/{name}" AS SELECT * FROM arrival')
            self.db.unregister("arrival")
            tables[name] = f'"{family}/{name}"'
        if "records" in tables:
            for (path,) in self.db.execute(
                f"SELECT DISTINCT relation FROM {tables['records']}"
            ).fetchall():
                self.db.execute(
                    f'CREATE TABLE "{family}/{path}" AS SELECT * EXCLUDE (relation) '
                    f"FROM {tables['records']} WHERE relation = '{path}'"
                )
                tables[path] = f'"{family}/{path}"'
        for relation, table in tables.items():
            self._view(family, relation, table)
        self._identify(family, tables)

    def adopt(self, family: str, relations: Mapping[str, str], *, empty: bool = False) -> None:
        """Store one family whose relations are SQL over what the database already reads
        (tables a provider loaded, other families' views, and this family's
        earlier relations by their view names), its identity relation first. An `empty` family
        keeps the columns of its relations and no row."""
        tables = {}
        for name, select in relations.items():
            self.ensure(Query(sql=select))
            tables[name] = f'"{family}/{name}"'
            body = f"SELECT * FROM ({select}) WHERE false" if empty else select
            self.db.execute(f"CREATE TABLE {tables[name]} AS {body}")
            self._view(family, name, tables[name])
        self._identify(family, tables)

    def _view(self, family: str, relation: str, table: str) -> None:
        """Name `table` `<family>_<relation>`."""
        view = f"{snake(family)}_{snake(relation)}"
        self.db.execute(f"CREATE VIEW main.{view} AS SELECT * FROM {table}")
        self.known.add(view)

    def _identify(self, family: str, tables: dict[str, str]) -> None:
        """Record the family and its identity relation (`facts`, else the first), then narrow
        each relation to each language. Naming it lets a derived identity come after the
        relations it reads."""
        prefix = snake(family)
        self.families[prefix] = family
        identity = tables.get("facts", tables[next(iter(tables))])
        self.identities.append(identity)
        if self.db.execute(
            f"SELECT 1 FROM (DESCRIBE {identity}) WHERE column_name = 'language'"
        ).fetchone():
            self._narrow(prefix, identity, tables)

    def ensure(self, query: Query) -> None:
        """Create, empty, every view `query` names that no fact filled in this project (and read
        `source_files` when it names them). Names are only ever added to the fact base, so the
        catalog is listed again only when a query names one the last listing lacked."""
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

    def cursor(self, language: str | None) -> duckdb.DuckDBPyConnection:
        """A cursor reading `language`'s facts through the plain view names, all facts for None."""
        cursor = self.db.cursor()
        if language is not None:
            cursor.execute(f"SET search_path = '{language},main'")
        return cursor

    def _narrow(self, prefix: str, identity: str, tables: dict[str, str]) -> None:
        """Each language's schema gets every relation narrowed to that language's facts: the
        identity by its `language`, a relation with `fact_id` by it, any other through the key it
        shares with a relation already narrowed (`call_id`, `class_id`, `function_id` ...). A
        family holding one language and no unlabeled fact needs no narrowing at all."""
        present = self.db.execute(
            f"SELECT language, list_distinct(list(lower(regexp_extract(path, '\\.[^./]+$')))) "
            f"FROM {identity} GROUP BY language"
        ).fetchall()
        for language, suffixes in present:
            if language is not None and (
                set(suffixes) & set(self.SUFFIXES.get(language, ())) or language in self.present
            ):
                self.languages.add(language)
        alone = len(present) == 1 and present[0][0] is not None
        columns = {
            relation: {
                c
                for (c,) in self.db.execute(
                    f"SELECT column_name FROM (DESCRIBE {table})"
                ).fetchall()
            }
            for relation, table in tables.items()
        }
        for language in [language for language, _ in present if language is not None]:
            self.db.execute(f"CREATE SCHEMA IF NOT EXISTS {language}")
            pending = dict(tables)
            narrowed: dict[str, str] = {}
            while pending:
                for relation, table in list(pending.items()):
                    clause = self._clause(
                        language,
                        identity,
                        table,
                        columns[relation],
                        {r: columns[r] for r in narrowed},
                        prefix,
                        alone,
                    )
                    if clause is None:
                        continue
                    self.db.execute(
                        f"CREATE VIEW {language}.{prefix}_{snake(relation)} "
                        f"AS SELECT * FROM {table} {clause}"
                    )
                    narrowed[relation] = table
                    del pending[relation]
                    break
                else:
                    for relation, table in pending.items():
                        self.db.execute(
                            f"CREATE VIEW {language}.{prefix}_{snake(relation)} "
                            f"AS SELECT * FROM {table}"
                        )
                    break

    def _clause(
        self,
        language: str,
        identity: str,
        table: str,
        columns: set[str],
        narrowed: dict[str, set[str]],
        prefix: str,
        alone: bool,
    ) -> str | None:
        """The `WHERE` narrowing one relation to `language`, None while its parent is pending."""
        if alone:
            return ""
        if table == identity:
            return f"WHERE language = '{language}'"
        if "fact_id" in columns:
            return (
                f"WHERE fact_id IN (SELECT fact_id FROM {identity} WHERE language = '{language}')"
            )
        # A relation keyed by `file_id` narrows through it, before any other `*_id` column,
        # which may hold node ids (`class_id`, `parent_id`) that no parent relation shares.
        keys = sorted((c for c in columns if c.endswith("_id")), key=lambda c: (c != "file_id", c))
        for key in keys:
            for parent, parent_columns in narrowed.items():
                target = key if key in parent_columns else self.KEYS.get(key)
                if target in parent_columns:
                    return (
                        f"WHERE {key} IN (SELECT {target} "
                        f"FROM {language}.{prefix}_{snake(parent)})"
                    )
        return None

    @cached_property
    def held(self) -> frozenset[str]:
        """The languages `ast_files` lists a file of."""
        rows = self.db.execute("SELECT DISTINCT language FROM ast_files").fetchall()
        return frozenset(language for (language,) in rows)

    @cached_property
    def imported(self) -> frozenset[str]:
        """The dotted names of the modules the project's Python files import by absolute name,
        as written (`import a.b`, `from a.b import c`). A name `from m import n as x` binds is
        counted too, so the set holds every module imported and possibly more."""
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

    def imports(self, modules: Iterable[str]) -> bool:
        """Whether the project imports one of `modules` or a submodule of one."""
        return any(
            name == module or name.startswith(f"{module}.")
            for module in modules
            for name in self.imported
        )

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

    def _read_sources(self) -> None:
        """`source_files` (path, content): every project file a fact names, in any `path` or
        `*_path` column, read from the root."""
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
