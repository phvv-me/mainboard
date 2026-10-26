"""One query surface over project-owned results, including work still running."""

import json
import math
import os
from collections.abc import Collection, Generator, Mapping
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from .core.errors import MissionError
from .core.project import Project
from .render.query import Query, is_execution_timeout
from .render.stamps import ResultHash, Stamp, canonical
from .runtime.tree import FileBudget
from .state.constants import ALIAS
from .state.lake import Lake, quoted
from .state.relations import Relation, Relations, records
from .state.views import Views
from .trials.artifacts import Artifact
from .trials.store.constants import NO_TABLE

# The files an export writes, by suffix, and how DuckDB's COPY writes each.
_FORMATS = {
    ".csv": "FORMAT csv, HEADER",
    ".parquet": "FORMAT parquet, COMPRESSION zstd",
    ".json": "FORMAT json, ARRAY true",
}


class Results:
    """Read the files we already collect; no server writes into a shared DuckDB file."""

    @contextmanager
    def _connected(
        self,
        sql: str | Path,
        project: str,
        connection: duckdb.DuckDBPyConnection,
        *,
        snapshot: bool = False,
    ) -> Generator[tuple[str, int]]:
        """`connection` with every results view, and the lake while SQL naming it runs.

        It yields the SQL text and its snapshot id, zero when no snapshot was requested.
        """
        text = Results._text(sql)
        Results._single(connection, text)
        lake = self._lake(text)
        with FileBudget.permitted(), _translated(), ExitStack() as stack:
            if lake is not None:
                stack.enter_context(_mounted(connection, lake))
            self.views.load(connection, project, sql=text)
            pinned = (
                stack.enter_context(_pinned(connection)) if snapshot and lake is not None else 0
            )
            yield text, pinned

    @contextmanager
    def stream(
        self, sql: str | Path, *, project: str = "", timeout: float = 10
    ) -> Generator[Query]:
        """One cursor over a pinned lake snapshot; the caller must consume it inside the block.

        Timeout is the client's DuckDB execution limit in seconds; zero disables it. Quack
        blocking operators use the server's own limit, not this client's setting.
        """
        if timeout < 0 or not math.isfinite(timeout):
            raise MissionError("query timeout must be finite and nonnegative")
        relations = Relations()
        connection = relations.connection
        with (
            closing(connection),
            _timed(timeout),
            self._connected(sql, project, connection, snapshot=True) as (text, snapshot),
        ):
            connection.execute(f"SET max_execution_time = {math.ceil(timeout * 1000)}")
            explain = _tokens(text)[0] == "explain"
            yield Query.started(connection, text, snapshot, timeout=timeout, explain=explain)

    @staticmethod
    def _copied(
        connection: duckdb.DuckDBPyConnection,
        text: str,
        *,
        path: Path,
        options: str,
        timeout: float,
    ) -> None:
        """Write what `text` answers to `path` through a staged file, published whole by a link."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as staged:
            temporary = Path(staged) / path.name
            connection.execute(f"SET max_execution_time = {math.ceil(timeout * 1000)}")
            target = temporary.as_posix().replace("'", "''")
            connection.execute(f"COPY ({text}) TO '{target}' ({options})").fetchall()
            with temporary.open("rb+") as completed:
                os.fsync(completed.fileno())
            os.link(temporary, path)

    @staticmethod
    def _kept(relations: Relations, text: str) -> Relation:
        # EXPLAIN is a relation, but not valid after CREATE TABLE AS.
        if relations.connection.extract_statements(text)[0].type == duckdb.StatementType.EXPLAIN:
            relations.connection.sql(text).to_table("explanation")
            return relations.connection.table("explanation")
        return relations.kept(text)

    @staticmethod
    def _prepared(lake: Lake) -> Lake:
        """`lake` ready and at this release's schema, saying why when it cannot be."""
        try:
            return lake.current()
        except OSError as fault:
            raise MissionError(
                f"cannot prepare the state lake at {lake.catalog}: {fault}"
            ) from None

    @staticmethod
    def _single(connection: duckdb.DuckDBPyConnection, sql: str) -> None:
        """Refuse SQL that is not exactly one statement."""
        try:
            statements = connection.extract_statements(sql)
        except duckdb.Error as fault:
            raise MissionError(str(fault).strip()) from None
        if len(statements) != 1:
            raise ValueError("results queries require one statement")

    @staticmethod
    def _text(sql: str | Path) -> str:
        """`sql` itself, or the UTF-8 text of the file it names."""
        if not isinstance(sql, Path):
            return sql
        try:
            return sql.expanduser().read_text(encoding="utf-8")
        except UnicodeError as fault:
            raise ValueError(f"SQL file {sql} must contain UTF-8 text: {fault}") from fault

    @staticmethod
    def _writable(target: Path) -> bool:
        """Whether a file could be created at `target`: its nearest folder takes writes."""
        parent = target
        while not parent.exists():
            parent = parent.parent
        return os.access(parent, os.W_OK | os.X_OK)

    def __init__(self, root: Path, *, experiment: str = "") -> None:
        if experiment and (
            experiment in {".", ".."}
            or Path(experiment).name != experiment
            or any(c in experiment for c in "*?[]")
        ):
            raise ValueError("experiment must name one experiment directory")
        self.root = root.resolve()
        self.experiment = experiment
        self.views = Views(self.root, experiment=experiment)
        self.provenance: list[Stamp] = []

    def _artifact(self, relations: Relations, row: Mapping, *, schema: str) -> Relation | None:
        """One artifact row's table when it is a Parquet table of `schema`, None for another."""
        reference = Artifact.model_validate_json(row["reference"])
        if reference.schema_name != schema:
            return None
        if reference.media_type != "application/vnd.apache.parquet":
            raise ValueError(f"{schema} contains a non-Parquet artifact")
        # References may be project- or workspace-relative. Match only this project's
        # location, including when Results is opened on the project itself.
        root = Path(row["root"])
        root = next(
            (
                parent
                for parent in root.parents
                if reference.relative.is_relative_to(root.relative_to(parent).as_posix())
            ),
            root,
        )
        table = relations.parquet(reference.read(root))
        if "_trial" in table.columns:
            raise ValueError("artifact payload reserves the _trial provenance column")
        return table.project(f"*, {quoted(str(row['context']))} AS _trial")

    def _creatable(self, lake: Lake, *, workspace: bool) -> None:
        """Refuse a lake that is not there and cannot be created here."""
        if not workspace:
            raise MissionError(f"no state lake at {lake.catalog}; {self.root} is not a workspace")
        if not all(
            Results._writable(target) for target in (self.root, lake.home, lake.lock.parent)
        ):
            raise MissionError(
                f"no state lake at {lake.catalog}; cannot create it in a read-only workspace"
            )

    def _lake(self, sql: str) -> Lake | None:
        """The workspace's lake, ready for `sql` to read, None when `sql` does not name it.

        A lake not there yet is created, in a workspace whose folders it may write.
        """
        tokens = _tokens(sql)
        if not any(
            name.strip('"') == ALIAS and separator == "."
            for name, separator in zip(tokens, tokens[1:], strict=False)
        ):
            return None
        lake = Lake.at(self.root)
        workspace = Project().manifest(self.root).is_file()
        if not lake.exists():
            self._creatable(lake, workspace=workspace)
        return Results._prepared(lake) if workspace else lake

    def export(
        self, sql: str | Path, path: Path, *, project: str = "", timeout: float = 0
    ) -> Path:
        """Export one SELECT to a new CSV, Parquet, or JSON file, inferred from its suffix.

        Only a complete file is published and an existing destination is never overwritten.
        """
        if timeout < 0 or not math.isfinite(timeout):
            raise MissionError("query timeout must be finite and nonnegative")
        options = _FORMATS.get(path.suffix.casefold())
        if options is None:
            raise ValueError("output suffix must be .csv, .parquet, or .json")
        path = path.expanduser().absolute()
        relations = Relations()
        with _timed(timeout), self._connected(sql, project, relations.connection) as (text, _):
            ResultHash(relations.connection.sql(text).columns)
            Results._copied(
                relations.connection, text, path=path, options=options, timeout=timeout
            )
        return path

    def query(self, sql: str | Path = "SELECT * FROM runs", *, project: str = "") -> Relation:
        """`rows` as a DuckDB relation for experiments, read in whole.

        It then outlives the lake and every file the query read.
        """
        relations = Relations()
        with self._connected(sql, project, relations.connection) as (text, _):
            return self._kept(relations, text)

    def rows(self, sql: str | Path = "SELECT * FROM runs", *, project: str = "") -> list[dict]:
        """Query a fresh local snapshot of the results views and the lake, one mapping per row.

        The views are runs, trials, events, artifacts and dispatch jobs, and the workspace's state
        lake is reached as `lake.<table>`.

        sql: SELECT text or a UTF-8 file Path; strings are never filenames. Relative paths, of
            the file or inside SQL, use the caller's current directory, not this root or the
            SQL file's parent.
        project: a research directory name; omitted means all projects, still labeled.
        Network refresh belongs to `mb job list`, never an SQL side effect.
        """
        relations = Relations()
        with self._connected(sql, project, relations.connection) as (text, _):
            return records(relations.connection.sql(text))

    def stamped(self, sql: str | Path = "SELECT * FROM runs", *, project: str = "") -> Relation:
        """Materialize a result and record its rendered identity for plot provenance.

        SQL naming `lake.*` attaches the lake and pins its snapshot, creating it when needed
        in a workspace. File views and `jobs` are collected separately and are not pinned;
        queries without `lake.*` leave the snapshot id at zero.
        """
        relations = Relations()
        with self._connected(sql, project, relations.connection, snapshot=True) as (
            text,
            snapshot,
        ):
            digest = ResultHash(relations.connection.sql(text).columns)
            relation = self._kept(relations, text)
            for row in relation.iter_rows(256):
                digest.add(tuple(canonical(value) for value in row))
            self.provenance.append(digest.stamp(snapshot))
            return relation.project("*")

    def table(
        self, schema: str, *, project: str = "", runs: Collection[str] | None = None
    ) -> Relation:
        """Read verified Parquet tables from collected storage, never a source host's path.

        Original repository and machine metadata remain in the _trial provenance column.
        runs: select run identities before reading their artifacts; None selects all runs.
            An empty collection selects none. Selected artifacts still require valid bytes.
        """
        listed = self.rows("SELECT root, context, reference FROM artifacts", project=project)
        if runs is not None:
            wanted = set(runs)
            listed = [row for row in listed if json.loads(row["context"]).get("run") in wanted]
        relations = Relations()
        tables = [
            table
            for row in listed
            if (table := self._artifact(relations, row, schema=schema)) is not None
        ]
        return relations.union(tables, empty=NO_TABLE)


@contextmanager
def _mounted(connection: duckdb.DuckDBPyConnection, lake: Lake) -> Generator[None]:
    """`lake` attached to `connection` while the block runs."""
    lake.attach(connection)
    try:
        yield
    finally:
        connection.execute(f"DETACH {ALIAS}")


@contextmanager
def _pinned(connection: duckdb.DuckDBPyConnection) -> Generator[int]:
    """A transaction over the attached lake's current snapshot, answering the snapshot's id.

    It commits when the block completes and rolls back otherwise.
    """
    connection.execute("BEGIN")
    try:
        yield connection.execute(f"SELECT * FROM {ALIAS}.current_snapshot()").fetchmany(1)[0][0]
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


@contextmanager
def _timed(timeout: float) -> Generator[None]:
    """Say the client's time limit when a query runs out of it, and any other error plainly."""
    try:
        yield
    except duckdb.Error as fault:
        if is_execution_timeout(fault):
            raise MissionError(f"query timed out (client limit {timeout:g}s)") from None
        raise MissionError(str(fault).strip()) from None


def _tokens(sql: str) -> list[str]:
    encoded = sql.encode()
    positions = duckdb.tokenize(sql)
    return [
        encoded[start:end].decode().strip().casefold()
        for (start, kind), end in zip(
            positions, [pos for pos, _ in positions[1:]] + [len(encoded)], strict=True
        )
        if kind != duckdb.token_type.comment
    ]


@contextmanager
def _translated() -> Generator[None]:
    """Report a DuckDB error as a MissionError, an execution timeout as itself."""
    try:
        yield
    except duckdb.Error as fault:
        if is_execution_timeout(fault):
            raise
        raise MissionError(str(fault).strip()) from None
