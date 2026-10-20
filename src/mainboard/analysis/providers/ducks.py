"""Facts from DuckDB community extensions: `sitting_duck` syntax trees, kept per file content.

The trees live in the tables `astcache.nodes` (one row per node, keyed by the git blob and
language of the file) and `astcache.keys` (the contents parsed, also those without a node), so a
fix's later passes parse only the files an edit changed. `astcache` is a DuckDB file attached to
the fact base: a run's own (in the run's budget folder) or a persistent one a command passes as
`cache`. A file another process holds attached cannot be shared, so the run keeps the tables in
its own fact base instead.

Parsing runs in this process when the engine's DuckDB loads `sitting_duck` (decided once, by
`in_process`) and otherwise through a DuckDB 1.5 command-line binary (`DuckCli`, fetched with the
extension on first use), which writes the nodes to a Parquet file this process inserts.

The extension is not in the community repository for DuckDB 2.0.0-alpha43385 (the engine's wheel
2.0.0.dev2609250715, commit ca15f79c32); its first signed build arrives with 2.0 GA. Until then
v1.16.0 is built from source against that exact DuckDB (scratchpad `sd116/build`, output
`sd116/out43385/sitting_duck.duckdb_extension`) and installed unsigned at
`~/.duckdb/extensions/v2.0.0-alpha43385/osx_arm64/sitting_duck.duckdb_extension`, where DuckDB
looks; `Budget.connect` allows unsigned extensions. After GA, `INSTALL sitting_duck FROM
community` fills the same kind of folder for v2.0.0.
"""

import ast
import hashlib
import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs git and the pinned DuckDB CLI with fixed argv since=2026-10-13
import tempfile
from collections import Counter
from contextlib import closing
from functools import cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar

import duckdb
from patos import FrozenModel
from pydantic import Field

from ...log import logger
from ..facts import UNSIGNED, Budget, listed
from .duckdb_cli import DuckCli

if TYPE_CHECKING:
    from ..facts import Facts


@cache
def in_process() -> bool:
    """Whether this process's DuckDB loads `sitting_duck`, tried once: where it does not (another
    machine, a job host), the command-line binary parses."""
    try:
        with closing(duckdb.connect(config=UNSIGNED)) as db:
            db.execute("LOAD sitting_duck")
    except duckdb.Error as error:
        logger.info("sitting_duck does not load here, parsing through the CLI: {}", error)
        return False
    return True


class Source(FrozenModel):
    """One file to analyze: its path in the project, its language and its content's git blob."""

    path: str
    language: str
    blob: str


class Ducks(FrozenModel):
    """Syntax trees from `sitting_duck`, through a DuckDB that has the extension."""

    cli: DuckCli = Field(default_factory=DuckCli)
    budget: Budget = Field(default_factory=Budget)
    cache: Path = Field(default_factory=lambda data: data["budget"].folder / "ast.duckdb")
    # The cache serves this project alone, so `register` drops the trees of contents it lacks.
    exclusive: bool = False
    # Whether `register` warns about valid files the grammar misreads. The files of other
    # repositories (a consumer's imports are read, nothing else) are not this run's to report.
    report_gaps: bool = True

    # The grammar each language parses with (CUDA is C++ to tree-sitter) and its suffixes, the
    # grammar's own first.
    LANGUAGES: ClassVar[dict[str, tuple[str, tuple[str, ...]]]] = {
        "python": ("python", (".py", ".pyi")),
        "cuda": ("cpp", (".cu", ".cuh")),
        "cpp": ("cpp", (".cpp", ".cc", ".cxx", ".c++", ".hh", ".hpp", ".hxx")),
        "c": ("c", (".c", ".h")),
        "rust": ("rust", (".rs",)),
        "swift": ("swift", (".swift",)),
        "typescript": ("typescript", (".ts", ".tsx", ".mts", ".cts")),
        "javascript": ("javascript", (".js", ".jsx", ".mjs", ".cjs")),
        "go": ("go", (".go",)),
        "toml": ("toml", (".toml",)),
        "markdown": ("markdown", (".md",)),
        "bash": ("bash", (".sh",)),
    }
    # Languages listed in `ast_files` (their families read the text) but never parsed.
    TEXT: ClassVar[dict[str, tuple[str, ...]]] = {"latex": (".tex",)}
    EMPTY: ClassVar[str] = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"

    def close(self) -> None:
        """Remove this run's cache and spill."""
        self.budget.close()

    def language(self, suffix: str) -> str | None:
        """The language a file suffix holds, None when none here parses it."""
        return next(
            (language for language, (_, ends) in self.LANGUAGES.items() if suffix.lower() in ends),
            None,
        )

    def errors(self, texts: dict[str, bytes]) -> Counter[str]:
        """How many syntax errors tree-sitter finds in each text (error nodes and the zero-width
        tokens it inserts for missing ones), by the language of its key's suffix."""
        with tempfile.TemporaryDirectory() as scratch:
            named = {}
            for number, (key, data) in enumerate(texts.items()):
                if (language := self.language(PurePosixPath(key).suffix)) is None:
                    continue
                grammar, _ = self.LANGUAGES[language]
                file = Path(scratch) / grammar / f"{number}{self.LANGUAGES[grammar][1][0]}"
                file.parent.mkdir(exist_ok=True)
                file.write_bytes(data)
                named[str(file)] = key
            counted = f"""
                SELECT file_path, count(*) FILTER (type = 'ERROR' OR (children_count = 0
                        AND start_line = end_line AND start_column = end_column)) AS errors
                FROM ({self._read(scratch)}) GROUP BY ALL"""
            return Counter({named[file]: errors for file, errors in self._rows(counted, scratch)})

    def sources(self, root: Path) -> list[Source]:
        """The project's files git does not ignore, in a language we parse or list, with their
        blobs."""
        suffixes = {
            suffix: language for language, (_, ends) in self.LANGUAGES.items() for suffix in ends
        } | {suffix: language for language, ends in self.TEXT.items() for suffix in ends}
        found = []
        for path in listed(root):
            language = suffixes.get(Path(path).suffix.lower())
            if language is None or not (root / path).is_file():
                continue
            data = (root / path).read_bytes()
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
            found.append(Source(path=path, language=language, blob=blob))
        return found

    def parse(self, db: duckdb.DuckDBPyConnection, root: Path, sources: list[Source]) -> int:
        """Parse the sources whose content `astcache` lacks; return how many were parsed.

        `read_ast` skips a file whose suffix its grammar does not claim (a `.cu` file read as
        C++ yields nothing), so each source is read through a link `<grammar>/<blob>.<language>
        <suffix>` with a suffix the grammar claims, which also names the blob and language of
        every node. A content that yields no nodes is still recorded, so it is not parsed again.
        With nothing to parse and no table yet, an empty Python file gives the table its columns.
        """
        parsed = set(db.execute("SELECT blob, language FROM astcache.keys").fetchall())
        files = {
            (s.blob, s.language): root / s.path
            for s in sources
            if s.language in self.LANGUAGES and (s.blob, s.language) not in parsed
        }
        (exists,) = db.execute(
            "SELECT count(*) > 0 FROM duckdb_tables() WHERE table_name = 'nodes' "
            "AND 'astcache' IN (database_name, schema_name)"
        ).fetchone() or (False,)
        if not files and exists:
            return 0
        with tempfile.TemporaryDirectory() as scratch:
            listing = files or {(self.EMPTY, "python"): Path(scratch) / "empty.py"}
            if not files:
                listing[(self.EMPTY, "python")].write_bytes(b"")
            for (blob, language), file in listing.items():
                grammar, _ = self.LANGUAGES[language]
                suffix = self.LANGUAGES[grammar][1][0]
                link = Path(scratch) / grammar / f"{blob}.{language}{suffix}"
                link.parent.mkdir(exist_ok=True)
                link.symlink_to(file.absolute())
            nodes = self._nodes(db, scratch)
            db.execute("SET preserve_insertion_order = false")
            db.begin()
            if not exists:
                db.execute(f"CREATE TABLE astcache.nodes AS {nodes} WITH NO DATA")
            db.execute(f"INSERT INTO astcache.nodes BY NAME {nodes}")
            db.executemany("INSERT INTO astcache.keys VALUES (?, ?)", list(listing))
            db.commit()
            db.execute("RESET preserve_insertion_order")
        return len(files)

    def register(self, facts: Facts, root: Path) -> None:
        """Parse what `astcache` lacks, then name the project's files and nodes as `ast_*` views:
        `ast_files` (path, language, blob) and `ast_nodes`, columns zero-based like mcmr's."""
        db = facts.db
        attached = self._attach(db)
        sources = self.sources(root)
        self.parse(db, root, sources)
        db.execute(
            "CREATE TABLE ast_sources (fact_id VARCHAR, path VARCHAR, language VARCHAR, "
            "blob VARCHAR)"
        )
        if sources:
            db.executemany(
                "INSERT INTO ast_sources VALUES (?, ?, ?, ?)",
                [(f"ast:{s.path}", s.path, s.language, s.blob) for s in sources],
            )
        facts.adopt(
            "Ast",
            {
                "files": """
                    SELECT fact_id, path, language, blob, 1 AS start_line, 0 AS start_column
                    FROM ast_sources""",
                "nodes": """
                    SELECT s.fact_id, s.path,
                           n.* EXCLUDE (blob, language, start_column, end_column),
                           n.start_column - 1 AS start_column, n.end_column - 1 AS end_column
                    FROM astcache.nodes n
                    JOIN ast_sources s ON n.blob = s.blob AND n.language = s.language""",
            },
        )
        if self.exclusive:
            for table in ("nodes", "keys"):
                db.execute(
                    f"DELETE FROM astcache.{table} "
                    "WHERE (blob, language) NOT IN (SELECT blob, language FROM ast_sources)"
                )
        if attached:
            db.execute("DETACH astcache")
        if self.report_gaps:
            self._report_gaps(facts, root)

    def _attach(self, db: duckdb.DuckDBPyConnection) -> bool:
        """Name the cache file `astcache` in `db`, or, when another process holds it, keep the
        tables in `db` itself under the same name. Whether the file is attached."""
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            db.execute(f"ATTACH {_quoted(str(self.cache))} AS astcache")
            attached = True
        except duckdb.Error as error:
            logger.info("{} is in use, parsing into this run's own tables: {}", self.cache, error)
            db.execute("CREATE SCHEMA astcache")
            attached = False
        db.execute("CREATE TABLE IF NOT EXISTS astcache.keys (blob VARCHAR, language VARCHAR)")
        return attached

    def _read(self, scratch: str, *options: str) -> str:
        """`read_ast` over each grammar folder of `scratch`, one row per node."""
        extra = "".join(f", {option}" for option in options)
        return " UNION ALL ".join(
            f"""SELECT * FROM read_ast({_quoted(str(folder / "*"))}, '{folder.name}',
                 source := 'full', peek := 'none', ignore_errors := true{extra})"""
            for folder in Path(scratch).iterdir()
            if folder.is_dir()
        )

    def _nodes(self, db: duckdb.DuckDBPyConnection, scratch: str) -> str:
        """A `SELECT` of the nodes of the files linked in `scratch`, with the blob and language
        their links name: the parse itself in `db`, or a read of what the command line wrote."""
        select = f"""
            SELECT unnest(regexp_extract(file_path, '([0-9a-f]{{40}})\\.(\\w+)\\.\\w+$',
                                         ['blob', 'language'])),
                   * EXCLUDE (file_path, language, semantic_type),
                   semantic_type_to_string(semantic_type) AS semantic_type,
                   get_kind(semantic_type) AS kind,
                   get_super_kind(semantic_type) AS super_kind
            FROM ({self._read(scratch, "prune := ['punctuation']")})"""
        if in_process():
            db.execute("LOAD sitting_duck")
            return select
        batch = _quoted(str(Path(scratch) / "nodes.parquet"))
        self._cli(f"COPY ({select}) TO {batch} (FORMAT parquet);", scratch)
        return f"SELECT * FROM read_parquet({batch})"

    def _rows(self, select: str, scratch: str) -> list[tuple[str, int]]:
        """The (file, count) rows of `select`, parsed in this process or in the command line."""
        if in_process():
            with closing(self.budget.connect()) as db:
                db.execute("LOAD sitting_duck")
                return db.execute(select).fetchall()
        out = Path(scratch) / "rows.csv"
        self._cli(f"COPY ({select}) TO {_quoted(str(out))} (HEADER false);", scratch)
        return [
            (file, int(count))
            for file, count in (line.rsplit(",", 1) for line in out.read_text().splitlines())
        ]

    def _cli(self, script: str, scratch: str) -> None:
        """Run `script` in the DuckDB command line under the budget, from `scratch`, so nothing
        it writes lands in the caller's working directory."""
        subprocess.run(
            [str(self.cli.ready())],
            input=(
                f"{self.budget.script()}LOAD sitting_duck;\nSET threads = {os.cpu_count()};\n"
                f"SET preserve_insertion_order = false;\n{script}"
            ),
            text=True,
            capture_output=True,
            check=True,
            cwd=scratch,
        )

    def _report_gaps(self, facts: Facts, root: Path) -> None:
        """Warn about Python files Python itself accepts but whose syntax tree has errors:
        syntax the grammar lacks (PEP 646 `x[*idx]`), which the derived families skip."""
        broken = facts.db.execute(
            "SELECT DISTINCT n.path FROM ast_nodes n JOIN ast_files f USING (path) "
            "WHERE f.language = 'python' AND n.type = 'ERROR' ORDER BY 1"
        ).fetchall()
        if gaps := [path for (path,) in broken if _parses(root / path)]:
            logger.warning(
                "the Python grammar misreads {} valid files, so derived facts skip them: {}",
                len(gaps),
                ", ".join(gaps[:5]),
            )


def _parses(file: Path) -> bool:
    try:
        ast.parse(file.read_bytes())
    except SyntaxError, ValueError:
        return False
    return True


def _quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"
