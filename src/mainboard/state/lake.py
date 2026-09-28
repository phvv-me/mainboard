# The workspace's state lake: one DuckLake, its catalog a SQLite file and its data Parquet.
#
# Every rule here was learned by a rehearsal against a real workspace's state, and each is
# encoded rather than remembered:
#
# LAYOUT. The catalog is `<out>/lake.sqlite` and the data `<out>/lake/`, the catalog outside the
# data folder so a sweep of the data can never take the catalog with it. The catalog records an
# absolute data path, so a workspace moved or cloned elsewhere would read, or refuse to read, the
# old folder; every attach therefore derives the data path from the workspace root and overrides
# the recorded one.
#
# NO LONG-LIVED CONNECTIONS. Each operation opens a fresh in-memory DuckDB, attaches, works and
# detaches, the way a CLI command lives. Two processes appending at once each commit a snapshot,
# and the retry budget with the catalog's busy timeout is what makes the later one wait and retry
# rather than fail. Only maintenance takes a lock, an application-level file lock so two
# maintainers never compact at once; appends never take it and stay safe beside it.
#
# NEVER CREATED BY ACCIDENT. An ATTACH of a missing catalog creates an empty lake, and even one
# told not to leaves an empty SQLite file behind, so the catalog's existence is checked before
# any attach and `create` is the only way a lake comes to be.
#
# THE SPEC. A lake is created at the extension's latest spec, never a pinned one, and records it.
# Opening never migrates a catalog: an older one keeps working at its spec until `upgrade` is
# asked for, since a migration is a one-way step every other reader must be ready for.
#
# EXTENSIONS. ducklake and sqlite load from a directory this tool owns under the user's cache,
# never the workspace and never DuckDB's default shared with every other program. Loading comes
# first and installing only when loading fails, so a machine with no network still opens a lake
# once the extensions are there.

import json
import os
import struct
import sys
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import duckdb
from filelock import FileLock, Timeout
from patos import FrozenModel
from pydantic import Field
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from ..core.errors import MissionError
from ..core.project import Project
from .schema import BY_NAME, TABLES, VERSION, VIEWS

# The alias every attach uses, so view definitions and callers' SQL name one catalog.
ALIAS = "lake"

# The DuckDB extensions an attach needs.
_EXTENSIONS = ("ducklake", "sqlite")

# The options a lake is created with: zstd Parquet at a cheap level, snapshots kept a month for
# time travel, and files a snapshot stopped referencing kept a week before cleanup deletes them.
_OPTIONS = (
    ("parquet_compression", "'zstd'"),
    ("parquet_compression_level", "3"),
    ("expire_older_than", "'30 days'"),
    ("delete_older_than", "'7 days'"),
)

# The files a workspace kept its state in before the lake, which `ready` refuses to bury under a
# fresh empty lake until `mb center migrate-state` has imported them.
_LEGACY = (
    "dispatch/db.sqlite",
    "dispatch/holds.json",
    "dispatch/digests.json",
    "batches",
    "costs",
    "studies",
    "catalog.ndjson",
    "collection.digests.json",
    "pulse.json",
)

# How many times a commit that lost a race to another writer's snapshot is retried.
_RETRIES = 100

# How long, in milliseconds, the SQLite catalog waits on another writer's lock.
_BUSY_MS = 30000

# How often a whole operation is retried when SQLite refused it as locked, and the longest pause
# between tries. SQLite answers a transaction that read before another writer committed with an
# immediate "database is locked" its busy timeout never waits out, since waiting cannot help that
# snapshot; a fresh attach reads the new one. Nothing was committed, so a retry cannot double.
_LOCKED_ATTEMPTS = 30
_LOCKED_WAIT_S = 1.0

# The live session on each catalog, gone when its last holder is, and the lock creating one.
_SESSIONS: weakref.WeakValueDictionary[tuple[Path, int], Session] = weakref.WeakValueDictionary()
_SHARING = RLock()

# The SQLite WAL-index header fields `check` reads from the catalog's `-shm` file: `mxFrame`, the
# last valid frame in the WAL, at byte 16, and `nBackfill`, how many frames were already copied
# into the database, at byte 96. Both are native-endian 32-bit integers.
_MAX_FRAME = 16
_BACKFILL = 96


def cache_home(
    platform: str = sys.platform,
    environ: Mapping[str, str] = os.environ,
    home: Path | None = None,
) -> Path:
    """The user's cache directory on `platform`, the one every OS convention names.

    `LOCALAPPDATA` on Windows, `~/Library/Caches` on macOS, else `XDG_CACHE_HOME` or `~/.cache`.
    """
    base = home or Path.home()
    if platform == "win32" and environ.get("LOCALAPPDATA"):
        return Path(environ["LOCALAPPDATA"])
    if platform == "darwin":
        return base / "Library" / "Caches"
    return Path(environ.get("XDG_CACHE_HOME") or base / ".cache")


def _extensions() -> Path:
    """Where this tool keeps DuckDB's extensions: its own folder under the user's cache."""
    return cache_home() / Project().name / "duckdb"


def _quoted(text: str) -> str:
    """`text` as a SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def _cell(kind: str, value: object) -> object:
    """`value` as a column of SQL type `kind` stages it, None kept as SQL NULL.

    A JSON column takes a string as already JSON and serializes anything else; a timestamp
    travels as its text, which the insert casts.
    """
    if value is None or kind not in {"JSON", "TIMESTAMPTZ"}:
        return value
    if kind == "TIMESTAMPTZ" or isinstance(value, str):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def insert(
    connection: duckdb.DuckDBPyConnection, table: str, rows: Iterable[Mapping[str, object]]
) -> int:
    """Append `rows` to the attached lake's `table` in one statement, returning how many.

    Each row maps column names to values; a column a row leaves out is NULL. Timestamps may be
    ISO strings or datetimes, JSON columns documents or Python values. The batch travels as one
    typed list per column, unnested side by side by DuckDB itself, never row by row.
    """
    schema = BY_NAME[table]
    staged = list(rows)
    if not staged:
        return 0
    columns = ", ".join(f"unnest(?::{kind}[]) AS {column}" for column, kind in schema.columns)
    values = [[_cell(kind, row.get(column)) for row in staged] for column, kind in schema.columns]
    connection.execute(f"INSERT INTO {ALIAS}.{table} BY NAME SELECT {columns}", values)
    return len(staged)


class Finding(FrozenModel):
    """One thing `check` found wrong.

    table: the table it breaks, empty for the catalog as a whole.
    kind: `catalog` (missing or unreadable), `wal` (committed catalog frames lost), or
        `missing` (a data or delete file the catalog references is not on disk).
    detail: what exactly, a path or the error.
    """

    table: str = ""
    kind: str
    detail: str


class Health(FrozenModel):
    """What `check` found: every finding, none meaning the lake is whole."""

    findings: tuple[Finding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings


class Lake(FrozenModel):
    """One workspace's state lake.

    root: the workspace root; the catalog, data and lock paths all derive from it.
    extensions: where DuckDB's extensions are loaded from and installed to.
    repository: an extension repository URL or directory, DuckDB's core repository when empty.
    """

    root: Path
    extensions: Path = Field(default_factory=_extensions)
    repository: str = ""

    @classmethod
    def at(cls, root: Path) -> Lake:
        """The lake of the workspace at `root`."""
        return cls(root=root.absolute())

    @property
    def out(self) -> Path:
        """The workspace's generated-state directory, `.mb` or a legacy name."""
        return Project().out(self.root)

    @property
    def catalog(self) -> Path:
        return self.out / "lake.sqlite"

    @property
    def data(self) -> Path:
        return self.out / "lake"

    @property
    def lock(self) -> Path:
        """The maintenance lock, which creation also takes and an append never does."""
        return self.out / "run" / "lake.maint.lock"

    def exists(self) -> bool:
        return self.catalog.is_file()

    def session(self) -> Session:
        """This process's one session on this lake, shared by every holder while any holds it.

        An attach costs a tenth of a second, and a command reads the lake from several places
        (the registry, a manifest's held machines, a verdict), so they share one. A catalog file
        replaced under the same name (a restore, a fresh import) is a different lake and gets a
        session of its own, since an attach keeps reading the file it opened.
        """
        try:
            identity = (self.catalog, self.catalog.stat().st_ino)
        except FileNotFoundError:
            identity = (self.catalog, 0)
        with _SHARING:
            shared = _SESSIONS.get(identity)
            if shared is None:
                shared = _SESSIONS[identity] = Session(self)
            return shared

    def ready(self) -> Lake:
        """This lake, created first when the workspace keeps no state at all yet.

        Creation is never a way to paper over a loss: data files without their catalog, or state
        files from before the lake that were never imported, raise with the repair instead, since
        an empty lake over either would read as a workspace that never dispatched anything.
        """
        if self.exists():
            return self
        if self.data.exists():
            raise MissionError(
                f"{self.data} holds lake data but its catalog {self.catalog} is gone; restore "
                "the catalog rather than creating a lake that forgets that data"
            )
        legacy = [name for name in _LEGACY if (self.out / name).exists()]
        if legacy:
            raise MissionError(
                f"{self.out} keeps state from before the lake ({', '.join(legacy)}); import it "
                "once with `mb center migrate-state`"
            )
        try:
            self.create()
        except MissionError:
            if not self.exists():
                raise
        return self

    @contextmanager
    def open(self, *, write: bool = False) -> Generator[duckdb.DuckDBPyConnection]:
        """A fresh connection with the lake attached as `lake`, detached and closed after.

        write: attach read-write; a read-only attach refuses every write.
        Raises MissionError when the lake was never created.
        """
        with self._attached(write=write, create=False) as connection:
            yield connection

    def create(self) -> str:
        """Create the lake at the extension's latest spec with the schema, returning the spec.

        Raises MissionError when one already exists, since a second create would be a no-op at
        best and a silently different lake at worst.
        """
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self.lock):
            if self.exists():
                raise MissionError(f"a state lake already exists at {self.catalog}")
            with self._attached(write=True, create=True) as connection:
                connection.execute("BEGIN")
                for option, value in _OPTIONS:
                    connection.execute(f"CALL {ALIAS}.set_option('{option}', {value})")
                connection.execute(f"USE {ALIAS}")
                for table in TABLES:
                    connection.execute(table.ddl)
                for view in VIEWS:
                    connection.execute(view.ddl)
                spec = _spec(connection)
                _record(connection, spec)
                connection.execute("COMMIT")
        return spec

    def evolve(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Bring a lake an older release created up to this release's schema, once.

        Tables and columns are only ever added and views replaced, so a lake at an older schema
        version loses nothing; the step is recorded in `schema_log` and taken under the
        maintenance lock so two processes never race it.
        """
        if _version(connection) >= VERSION:
            return
        with FileLock(self.lock):
            if _version(connection) >= VERSION:
                return
            held = set(
                connection.execute(
                    "SELECT table_name, column_name FROM duckdb_columns() WHERE database_name = ?",
                    [ALIAS],
                ).fetchall()
            )
            tables = {table for table, _ in held}
            connection.execute("BEGIN")
            connection.execute(f"USE {ALIAS}")
            for table in TABLES:
                if table.name not in tables:
                    connection.execute(table.ddl)
                    continue
                for column, kind in table.columns:
                    if (table.name, column) not in held:
                        connection.execute(f"ALTER TABLE {table.name} ADD COLUMN {column} {kind}")
            for view in VIEWS:
                connection.execute(view.ddl.replace("CREATE VIEW", "CREATE OR REPLACE VIEW", 1))
            _record(connection, _spec(connection))
            connection.execute("COMMIT")
            connection.execute("USE memory")

    def upgrade(self) -> str:
        """Migrate the catalog to the extension's latest spec, returning the spec it is now at.

        Recorded in `schema_log` when the spec moved, so the lake says which releases wrote it.
        """
        with self._attached(write=True, create=False, migrate=True) as connection:
            spec = _spec(connection)
            recorded = connection.execute(
                f"SELECT spec FROM {ALIAS}.schema_log ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if recorded is None or recorded[0] != spec:
                _record(connection, spec)
        return spec

    def spec(self) -> str:
        """The DuckLake spec the catalog is at."""
        with self.open() as connection:
            return _spec(connection)

    def append(self, table: str, rows: Iterable[Mapping[str, object]]) -> int:
        """Append `rows` to `table` in one commit, returning how many were appended.

        Retried with a fresh attach while SQLite answers that the catalog is locked.
        """
        staged = list(rows)
        for attempt in _patiently():
            with attempt, self.open(write=True) as connection:
                insert(connection, table, staged)
        return len(staged)

    def query(self, sql: str, parameters: Iterable[object] = ()) -> list[tuple[Any, ...]]:
        """One read-only query's rows; tables and views are named `lake.<name>`."""
        with self.open() as connection:
            return connection.execute(sql, list(parameters)).fetchall()

    def maintain(self) -> bool:
        """Checkpoint the lake under the maintenance lock, False when another holds it.

        One `CHECKPOINT` flushes inlined rows to Parquet, expires old snapshots, merges small
        files, deletes files past their retention and removes orphans a killed writer left.
        Appends do not take the lock, so they keep committing beside it and retry on conflict.
        """
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(self.lock, timeout=0):
                for attempt in _patiently():
                    with attempt, self.open(write=True) as connection:
                        connection.execute(f"CHECKPOINT {ALIAS}")
        except Timeout:
            return False
        return True

    def check(self) -> Health:
        """Compare what the catalog references with what is on disk, per table.

        A deleted data file breaks only its table, and `count(*)` answers from the catalog so
        it hides the loss; this is what finds it. A lost catalog WAL is reported only when the
        WAL index proves frames were committed there and never copied into the catalog.
        """
        if not self.exists():
            return Health(findings=(Finding(kind="catalog", detail=f"{self.catalog} missing"),))
        findings = list(self._wal())
        try:
            with self.open() as connection:
                tables = connection.execute(
                    "SELECT table_name FROM duckdb_tables() WHERE database_name = ? "
                    "ORDER BY table_name",
                    [ALIAS],
                ).fetchall()
                for (table,) in tables:
                    listed = connection.execute(
                        "SELECT data_file, delete_file FROM ducklake_list_files(?, ?)",
                        [ALIAS, table],
                    ).fetchall()
                    findings.extend(
                        Finding(table=table, kind="missing", detail=path)
                        for pair in listed
                        for path in pair
                        if path and not Path(path).is_file()
                    )
        except duckdb.Error as fault:
            findings.append(Finding(kind="catalog", detail=str(fault).splitlines()[0]))
        return Health(findings=tuple(findings))

    def _wal(self) -> Iterator[Finding]:
        """A finding when the catalog's WAL is gone while its index says frames were unsaved."""
        index = Path(f"{self.catalog}-shm")
        if Path(f"{self.catalog}-wal").exists() or not index.is_file():
            return
        header = index.read_bytes()[: _BACKFILL + 4]
        if len(header) < _BACKFILL + 4:
            return
        (frames,) = struct.unpack_from("=I", header, _MAX_FRAME)
        (saved,) = struct.unpack_from("=I", header, _BACKFILL)
        if frames > saved:
            yield Finding(
                kind="wal",
                detail=f"{index.name} records {frames - saved} committed catalog frame(s) whose "
                "WAL is gone: the last commits before a crash are lost",
            )

    @contextmanager
    def _attached(
        self, *, write: bool, create: bool, migrate: bool = False
    ) -> Generator[duckdb.DuckDBPyConnection]:
        """A fresh in-memory DuckDB with the lake attached, detached and closed on the way out.

        A failure skips the detach: closing the connection drops the attachment and rolls back
        whatever transaction the failure left open. Raises MissionError before any attach when
        the lake is not being created and was never created.
        """
        if not create and not self.exists():
            raise MissionError(
                f"no state lake at {self.catalog}; create one with `mb center migrate-state`"
            )
        connection = duckdb.connect(
            config={"autoinstall_known_extensions": False, "autoload_known_extensions": False}
        )
        try:
            self.attach(connection, write=write, create=create, migrate=migrate)
            yield connection
            connection.execute("USE memory")
            connection.execute(f"DETACH {ALIAS}")
        finally:
            connection.close()

    def attach(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        write: bool = False,
        create: bool = False,
        migrate: bool = False,
    ) -> None:
        """Attach this lake to `connection` as `lake`, read-only unless `write`, so a query that
        already has its own views (`mb query`) reaches every table beside them."""
        self._load(connection)
        connection.execute(f"SET ducklake_max_retry_count = {_RETRIES}")
        connection.execute("SET enable_progress_bar = false")
        connection.execute("SET TimeZone = 'UTC'")
        options = [
            f"DATA_PATH {_quoted(self.data.as_posix() + '/')}",
            "OVERRIDE_DATA_PATH true",
            f"CREATE_IF_NOT_EXISTS {str(create).lower()}",
            f"META_BUSY_TIMEOUT {_BUSY_MS}",
        ]
        if migrate:
            options.append("AUTOMATIC_MIGRATION true")
        options.append("META_JOURNAL_MODE 'WAL'" if write else "READ_ONLY")
        target = _quoted(f"ducklake:sqlite:{self.catalog.as_posix()}")
        connection.execute(f"ATTACH {target} AS {ALIAS} ({', '.join(options)})")

    def _load(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Load the extensions from this tool's directory, installing only what fails to load.

        Raises MissionError naming the directory when an extension is neither there nor
        installable, as on a machine that never went online.
        """
        connection.execute(f"SET extension_directory = {_quoted(self.extensions.as_posix())}")
        for name in _EXTENSIONS:
            try:
                connection.load_extension(name)
            except duckdb.IOException:
                try:
                    connection.install_extension(name, repository_url=self.repository or None)
                    connection.load_extension(name)
                except duckdb.Error as fault:
                    raise MissionError(
                        f"DuckDB extension {name} is not in {self.extensions} and could not be "
                        f"installed: {str(fault).splitlines()[0]}"
                    ) from fault


def locked(fault: BaseException) -> bool:
    """Whether `fault` is SQLite refusing the catalog as locked, which a fresh attach outlives."""
    return isinstance(fault, duckdb.Error) and "database is locked" in str(fault)


def recoverable(fault: BaseException) -> bool:
    """Whether a fresh attach outlives `fault`: a locked catalog, or inlined rows another
    process's maintenance flushed out from under an attached session's cached view of them."""
    return locked(fault) or (
        isinstance(fault, duckdb.Error) and "ducklake_inlined_data" in str(fault)
    )


class Session:
    """One lake attached read-write for as long as its owner keeps it, attached on first use.

    Attaching costs a tenth of a second, so a registry or a bus keeps one session rather than
    attaching per statement, and never attaches at all when it is never used (a host installing
    an environment builds a board that never touches its registry). A statement the catalog
    refused as locked or stale is run again on a fresh attach; a failed statement committed
    nothing, so running it again cannot duplicate a row.
    """

    def __init__(self, lake: Lake) -> None:
        self.lake = lake
        self._stack = ExitStack()
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._turn = RLock()
        # Detaching is the collector's job, not a caller's: otherwise only interpreter exit
        # releases the catalog, which Windows then refuses to delete or move.
        weakref.finalize(self, self._stack.close)

    def close(self) -> None:
        """Detach now; the next statement attaches again."""
        with self._turn:
            self._stack.close()
            self._connection = None

    def rows(self, sql: str, parameters: Iterable[object] = ()) -> list[tuple[Any, ...]]:
        """What `sql` answers in this session."""
        bound = list(parameters)
        return self.run(lambda connection: connection.execute(sql, bound).fetchall())

    def append(self, table: str, rows: Iterable[Mapping[str, object]]) -> int:
        """Append `rows` to `table` in one commit, returning how many were appended."""
        staged = list(rows)
        return self.run(lambda connection: insert(connection, table, staged))

    def run[T](self, statement: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """`statement` over this session's connection, reattached while the lake refuses it.

        One statement at a time: a DuckDB connection is not safe across threads (a sampler
        publishes from its own thread while its owner reads the stream), and a per-thread cursor
        would be closed under its thread by the reattach a stale catalog snapshot needs.
        """
        with self._turn:
            return _patiently(recoverable, before=self.close)(lambda: statement(self._attached()))

    def _attached(self) -> duckdb.DuckDBPyConnection:
        """The connection, attaching first; a fresh workspace's lake is created on this use and
        an older release's lake brought up to this schema."""
        if self._connection is None:
            connection = self._stack.enter_context(self.lake.ready().open(write=True))
            self.lake.evolve(connection)
            self._connection = connection
        return self._connection


def _patiently(
    retry: Callable[[BaseException], bool] = locked, *, before: Callable[[], None] | None = None
) -> Retrying:
    """Attempts of one operation, retried while `retry` accepts the fault, the last one raised.

    before: run between attempts, as a session detaching so the next attempt attaches afresh.
    """
    return Retrying(
        retry=retry_if_exception(retry),
        stop=stop_after_attempt(_LOCKED_ATTEMPTS),
        wait=wait_random_exponential(max=_LOCKED_WAIT_S),
        before_sleep=(lambda _state: before()) if before else None,
        reraise=True,
    )


def _version(connection: duckdb.DuckDBPyConnection) -> int:
    """The newest schema version the attached lake records."""
    row = connection.execute(f"SELECT max(version) FROM {ALIAS}.schema_log").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _spec(connection: duckdb.DuckDBPyConnection) -> str:
    """The attached lake's DuckLake spec version."""
    row = connection.execute(
        f"SELECT value FROM {ALIAS}.options() WHERE option_name = 'version'"
    ).fetchone()
    return str(row[0]) if row else ""


def _record(connection: duckdb.DuckDBPyConnection, spec: str) -> None:
    """Record that the schema at `VERSION` now lives in a lake at `spec`, and by which engine."""
    engine = connection.execute("SELECT library_version FROM pragma_version()").fetchone()
    insert(
        connection,
        "schema_log",
        [
            {
                "ts": datetime.now(UTC),
                "version": VERSION,
                "spec": spec,
                "engine": engine[0] if engine else "",
            }
        ],
    )
