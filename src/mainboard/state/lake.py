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
#
# SERVED. `serving` opens DuckDB on the lake itself and serves it over Quack, DuckDB's own
# client-server protocol, on localhost only; another machine reaches it through an ssh tunnel.
# With `MB_LAKE=quack:host:port` (and `MB_LAKE_TOKEN`) set, every attach here is that served
# lake instead of the workspace's files, under the same `lake` alias, so every query and append
# runs unchanged. Learned against DuckDB 2.0.0.dev2609250715: a Quack client resolves names in
# the server's default database only (hence DuckDB opened on the lake, not attached to memory);
# bound parameters are dropped on the way to the server (hence `inlined`); a table macro's
# `query()` runs a statement twice (duckdb-quack#282), so writes go through tables only; and
# both ends must run the same DuckDB release. Maintenance stays with the machine holding the
# files.

import base64
import json
import os
import secrets
import sqlite3
import struct
import sys
import tempfile
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager, suppress
from datetime import UTC, date, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import duckdb
from filelock import FileLock, Timeout
from patos import FrozenModel
from pydantic import Field
from sqlalchemy import (
    JSON,
    ClauseElement,
    Column,
    DateTime,
    LargeBinary,
    Table,
    func,
    literal_column,
)
from sqlalchemy import select as selecting
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from ..core.errors import MissionError
from ..core.project import Project
from ..runtime.tree import FileBudget
from . import schema
from .schema import ALIAS, DIALECT, TABLES, VERSION, VIEWS, ddl, kind

# The DuckDB extensions an attach needs, and the ones reaching or serving a lake over Quack.
_EXTENSIONS = ("ducklake", "sqlite")
_QUACK = ("httpfs", "quack")

# Quack's own default port.
PORT = 9494

# The options a lake is created with: zstd Parquet at a cheap level, snapshots kept a month for
# time travel, and files a snapshot stopped referencing kept a week before cleanup deletes them.
_OPTIONS = (
    ("parquet_compression", "'zstd'"),
    ("parquet_compression_level", "3"),
    ("expire_older_than", "'30 days'"),
    ("delete_older_than", "'7 days'"),
)

# Tables of stored bytes, which keep their rows out of the catalog and their files small. A small
# insert (ten rows or fewer) otherwise stays inside the SQLite catalog until a checkpoint, which
# suits log rows and costs a catalog its size for source-file bytes: 598 blobs made D:\projects'
# catalog 102 MB of its 104. And DuckDB holds a whole row group while writing and 2,048 rows of a
# vector while reading, so 8 MB chunks in the default 122,880-row groups and 512 MB files ran a
# 24 GB center out of memory merging 27k blob files. Groups of 16 rows let a writer split files
# at 64 MB, which a 2,048-row vector never exceeds.
_BYTES = (schema.blobs,)
_BYTES_OPTIONS = (
    ("data_inlining_row_limit", "0"),
    ("parquet_row_group_size", "16"),
    ("target_file_size", "'64MB'"),
)

# The catalog size past which `check` suggests a compaction.
_CATALOG_FLOOR_BYTES = 64 << 20

# DuckDB's own largest JSON object by default, the floor of a staged file's object limit. A
# reader buffers twice its limit, so the limit follows the longest staged line: a fixed 2 GiB,
# set so a 25 MB source file could be sealed (2026-09-29), reserved 4 GiB for every append and
# ran the 7 GB macOS CI runner out of memory (2026-10-07).
_JSON_OBJECT_BYTES = 16 << 20

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
_SESSIONS: weakref.WeakValueDictionary[tuple[str, int], Session] = weakref.WeakValueDictionary()
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
    """The user's cache directory on `platform`: `~/Library/Caches` on macOS, else
    `XDG_CACHE_HOME` or `~/.cache`."""
    base = home or Path.home()
    if platform == "darwin":
        return base / "Library" / "Caches"
    return Path(environ.get("XDG_CACHE_HOME") or base / ".cache")


def _extensions() -> Path:
    """Where this tool keeps DuckDB's extensions: its own folder under the user's cache."""
    return cache_home() / Project().name / "duckdb"


def quoted(text: str) -> str:
    """`text` as a SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def _literal(value: object) -> str:
    """`value` as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, datetime):
        return f"TIMESTAMPTZ {quoted(value.isoformat())}"
    if isinstance(value, date):
        return f"DATE {quoted(value.isoformat())}"
    return quoted(str(value))


def inlined(sql: str, parameters: Sequence[object]) -> str:
    """`sql` with each `?` outside a string literal replaced by its parameter as a literal.

    What a served lake is sent, since a Quack attach drops bound parameters. Splitting on quotes
    leaves literals at the odd pieces, an escaped `''` included.
    """
    pieces = sql.split("'")
    outside = range(0, len(pieces), 2)
    marks = sum(pieces[index].count("?") for index in outside)
    if marks != len(parameters):
        raise ValueError(f"{len(parameters)} parameters for {marks} placeholders in {sql!r}")
    values = iter(parameters)
    for index in outside:
        head, *rest = pieces[index].split("?")
        pieces[index] = head + "".join(_literal(next(values)) + tail for tail in rest)
    return "'".join(pieces)


def compiled(statement: ClauseElement) -> tuple[str, list[Any]]:
    """`statement` as DuckDB SQL, and the values its `?` placeholders bind in order."""
    done = DIALECT.statement_compiler(
        DIALECT, statement, compile_kwargs={"render_postcompile": True}
    )
    bound = done.construct_params()
    return done.string, [bound[name] for name in done.positiontup or ()]


def _cell(column: Column, value: object) -> object:
    """`value` as one NDJSON field of `column`, None kept as SQL NULL.

    A JSON column takes a string as already JSON and embeds anything else as it is; bytes travel
    as base64 and a timestamp as its text, both of which the insert decodes.
    """
    match column.type, value:
        case _, None:
            return None
        case JSON(), str():
            return json.loads(value)
        case LargeBinary(), bytes():
            return base64.b64encode(value).decode("ascii")
        case DateTime(), _:
            return str(value)
    return value


@contextmanager
def ndjson(
    records: Iterable[Mapping[str, object] | str], *, typed: bool = False
) -> Generator[str]:
    """`records` (mappings, or lines already JSON) as a temporary NDJSON file, answered as the
    call reading it, its object limit the longest line's: one `json` object per record, or
    `typed`, the columns `read_json` infers from every record, a key one lacks read as null.

    DuckDB reads a file like this in a fraction of a second, while this build binds a list
    parameter at two milliseconds an element: twenty thousand rows took forty seconds.
    """
    descriptor, name = tempfile.mkstemp(prefix="mb-", suffix=".ndjson")
    longest = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as staged:
            for record in records:
                line = record if isinstance(record, str) else json.dumps(record, default=str)
                longest = max(longest, len(line.encode()))
                staged.write(line + "\n")
        path, limit = quoted(Path(name).as_posix()), max(_JSON_OBJECT_BYTES, longest + 1)
        yield (
            f"read_json({path}, format = 'newline_delimited', sample_size = -1, "
            f"maximum_object_size = {limit})"
            if typed
            else f"read_ndjson_objects({path}, maximum_object_size = {limit})"
        )
    finally:
        Path(name).unlink(missing_ok=True)


def insert(
    connection: duckdb.DuckDBPyConnection, table: Table, rows: Iterable[Mapping[str, object]]
) -> int:
    """Append `rows` to the attached lake's `table` in one statement, returning how many.

    Each row maps column names to values; a column a row leaves out is NULL. Timestamps may be
    ISO strings or datetimes, JSON columns documents or Python values, BLOB columns bytes. The
    batch travels as one NDJSON file DuckDB reads itself, never row by row.
    """
    listed = list(rows)
    if not listed:
        return 0
    with staged(table, listed) as select:
        connection.execute(f"INSERT INTO {ALIAS}.{table.name} BY NAME {select}")
    return len(listed)


@contextmanager
def staged(table: Table, rows: Iterable[Mapping[str, object]]) -> Generator[str]:
    """Stage `rows` for one statement as a SELECT typed like `table`'s columns (see `insert`).

    The rows wait in a temporary NDJSON file while the block runs, which the SELECT reads.
    """
    columns = ", ".join(_decoded(column) for column in table.columns)
    cells = (
        {column.name: _cell(column, row.get(column.name)) for column in table.columns}
        for row in rows
    )
    with ndjson(cells) as source:
        yield f"SELECT {columns} FROM {source}"


def _decoded(column: Column) -> str:
    """The SQL reading `column` back out of a staged NDJSON object."""
    name = column.name
    match column.type:
        case JSON():
            return f"json->'{name}' AS {name}"
        case LargeBinary():
            return f"from_base64(json->>'{name}') AS {name}"
    return f"CAST(json->>'{name}' AS {kind(column)}) AS {name}"


class Finding(FrozenModel):
    """One thing `check` found wrong.

    table: the table it breaks, empty for the catalog as a whole.
    kind: `catalog` (missing or unreadable), `wal` (committed catalog frames lost), `missing`
        (a data or delete file the catalog references is not on disk), `bloat` (a catalog
        grown past what a compaction would leave), `evidence` (an indexed file's object absent,
        short of a chunk or failing a checksum) or `unreadable` (kept chunks DuckDB cannot
        read back).
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
    served: the `quack:host:port` a lake is served at, attached instead of the workspace's own
        files when set; `MB_LAKE` by default, its token `MB_LAKE_TOKEN`.
    """

    root: Path
    extensions: Path = Field(default_factory=_extensions)
    repository: str = ""
    served: str = Field(default_factory=lambda: Project().variable("LAKE").read())

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

    @property
    def token(self) -> Path:
        """Where `serving` keeps the token it hands clients, so a restart keeps theirs valid."""
        return self.out / "run" / "lake.token"

    def exists(self) -> bool:
        return bool(self.served) or self.catalog.is_file()

    def session(self) -> Session:
        """This process's one session on this lake, shared by every holder while any holds it.

        An attach costs a tenth of a second, and a command reads the lake from several places
        (the registry, a manifest's held machines, a verdict), so they share one. A catalog file
        replaced under the same name (a restore, a fresh import) is a different lake and gets a
        session of its own, since an attach keeps reading the file it opened.
        """
        try:
            identity = (self.served or str(self.catalog), self.catalog.stat().st_ino)
        except FileNotFoundError:
            identity = (self.served or str(self.catalog), 0)
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
        try:
            self.create()
        except MissionError:
            if not self.exists():
                raise
        return self

    def current(self) -> Lake:
        """This lake, ready and at this release's schema, for a reader that attaches read-only.

        Only a write session evolves a lake, so a query right after an upgrade would otherwise
        read the views an older release left (a `lake.runs` without its `project` column).
        """
        self.ready()
        if self.served:
            return self
        with self.open() as connection:
            if _version(connection) >= VERSION:
                return self
        with self.open(write=True) as connection:
            self.evolve(connection)
        return self

    @contextmanager
    def open(self, *, write: bool = False) -> Generator[duckdb.DuckDBPyConnection]:
        """A fresh connection with the lake attached as `lake`, detached and closed after.

        write: attach read-write; a read-only attach refuses every write.
        Raises MissionError when the lake was never created.
        """
        with FileBudget.permitted(), self._attached(write=write, create=False) as connection:
            yield connection

    def create(self) -> str:
        """Create the lake at the extension's latest spec with the schema, returning the spec.

        Raises MissionError when one already exists, since a second create would be a no-op at
        best and a silently different lake at worst.
        """
        self._held("create")
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
                    connection.execute(ddl(table))
                for view in VIEWS:
                    connection.execute(view.ddl)
                spec = _spec(connection)
                _record(connection, spec)
                connection.execute("COMMIT")
                # A table's settings wait for the table to exist outside its transaction.
                _set_apart(connection)
        return spec

    def evolve(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Bring a lake an older release created up to this release's schema, once.

        Tables and columns are only ever added and views replaced, so a lake at an older schema
        version loses nothing; the step is recorded in `schema_log` and taken under the
        maintenance lock so two processes never race it.
        """
        if self.served or _version(connection) >= VERSION:
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
                    connection.execute(ddl(table))
                    continue
                for column in table.columns:
                    if (table.name, column.name) not in held:
                        connection.execute(
                            f"ALTER TABLE {table.name} ADD COLUMN {column.name} {kind(column)}"
                        )
            # A table's settings wait for the table to exist outside its transaction.
            connection.execute("COMMIT")
            connection.execute("BEGIN")
            _set_apart(connection)
            for view in VIEWS:
                connection.execute(view.ddl)
            _record(connection, _spec(connection))
            connection.execute("COMMIT")
            connection.execute("USE memory")

    def upgrade(self) -> str:
        """Migrate the catalog to the extension's latest spec, returning the spec it is now at.

        Recorded in `schema_log` when the spec moved, so the lake says which releases wrote it.
        """
        self._held("upgrade")
        with self._attached(write=True, create=False, migrate=True) as connection:
            spec = _spec(connection)
            latest = (
                selecting(schema.schema_log.c.spec)
                .order_by(literal_column("rowid").desc())
                .limit(1)
            )
            recorded = connection.execute(*compiled(latest)).fetchone()
            if recorded is None or recorded[0] != spec:
                _record(connection, spec)
        return spec

    def spec(self) -> str:
        """The DuckLake spec the catalog is at."""
        with self.open() as connection:
            return _spec(connection)

    def append(self, table: Table, rows: Iterable[Mapping[str, object]]) -> int:
        """Append `rows` to `table` in one commit, returning how many were appended.

        Retried with a fresh attach while SQLite answers that the catalog is locked.
        """
        staged = list(rows)
        for attempt in _patiently():
            with attempt, self.open(write=True) as connection:
                insert(connection, table, staged)
        return len(staged)

    def transact[T](self, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """`work` in one transaction on a fresh write attach, committed whole or not at all.

        The lake is brought to this release's schema first. Retried with a fresh attach while
        SQLite answers that the catalog is locked; a failure commits nothing, so a retry cannot
        double what `work` appends.
        """
        return _patiently()(lambda: self._transacted(work))

    def _transacted[T](self, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        with self.open(write=True) as connection:
            self.evolve(connection)
            connection.execute("BEGIN")
            done = work(connection)
            connection.execute("COMMIT")
        return done

    def query(
        self, statement: ClauseElement | str, parameters: Iterable[object] = ()
    ) -> list[tuple[Any, ...]]:
        """One read-only query's rows; tables and views are named `lake.<name>`."""
        with self.open() as connection:
            return self.execute(connection, statement, parameters).fetchall()

    def execute(
        self,
        connection: duckdb.DuckDBPyConnection,
        statement: ClauseElement | str,
        parameters: Iterable[object] = (),
    ) -> duckdb.DuckDBPyConnection:
        """`statement` run on `connection`, a built statement compiled or SQL with `parameters`
        bound, inlined for a served lake."""
        sql, bound = (
            (statement, list(parameters)) if isinstance(statement, str) else compiled(statement)
        )
        if self.served and bound:
            return connection.execute(inlined(sql, bound))
        return connection.execute(sql, bound)

    def maintain(self) -> bool:
        """Checkpoint the lake under the maintenance lock, False when another holds it.

        One `CHECKPOINT` flushes inlined rows to Parquet, expires old snapshots, merges small
        files, deletes files past their retention and removes orphans a killed writer left.
        Appends do not take the lock, so they keep committing beside it and retry on conflict.
        The catalog file keeps the pages the flush freed until a `VACUUM` gives them back, which
        waits for no one: another process holding the catalog just leaves them for next time.
        """
        self._held("compact")
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(self.lock, timeout=0):
                for attempt in _patiently():
                    with attempt, self.open(write=True) as connection:
                        connection.execute(f"CHECKPOINT {ALIAS}")
                with (
                    suppress(sqlite3.OperationalError),
                    closing(sqlite3.connect(self.catalog, timeout=0)) as catalog,
                ):
                    catalog.execute("VACUUM")
        except Timeout:
            return False
        return True

    def check(self) -> Health:
        """Compare what the catalog references with what is on disk, per table.

        A deleted data file breaks only its table, and `count(*)` answers from the catalog so
        it hides the loss; this is what finds it. A lost catalog WAL is reported only when the
        WAL index proves frames were committed there and never copied into the catalog.
        """
        self._held("check")
        if not self.exists():
            return Health(findings=(Finding(kind="catalog", detail=f"{self.catalog} missing"),))
        findings = list(self._wal())
        if (size := self.catalog.stat().st_size) > _CATALOG_FLOOR_BYTES:
            findings.append(
                Finding(
                    kind="bloat",
                    detail=f"the catalog holds {size >> 20} MB; `mb lake compact` flushes it to "
                    "Parquet and gives the space back",
                )
            )
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
                f"no state lake at {self.catalog}; the first command recording state creates it"
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
        already has its own views (`mb query`) reaches every table beside them.

        A served lake is attached over Quack instead, read-write whatever `write` says: what a
        client may do there is the server's to decide.
        """
        connection.execute("SET enable_progress_bar = false")
        connection.execute("SET TimeZone = 'UTC'")
        if self.served:
            self._reach(connection)
            return
        self._load(connection, _EXTENSIONS)
        connection.execute(f"SET ducklake_max_retry_count = {_RETRIES}")
        options = [
            f"DATA_PATH {quoted(self.data.as_posix() + '/')}",
            "OVERRIDE_DATA_PATH true",
            f"CREATE_IF_NOT_EXISTS {str(create).lower()}",
            f"META_BUSY_TIMEOUT {_BUSY_MS}",
        ]
        if migrate:
            options.append("AUTOMATIC_MIGRATION true")
        options.append("META_JOURNAL_MODE 'WAL'" if write else "READ_ONLY")
        target = quoted(f"ducklake:sqlite:{self.catalog.as_posix()}")
        connection.execute(f"ATTACH {target} AS {ALIAS} ({', '.join(options)})")

    @contextmanager
    def serving(self, port: int = PORT, token: str = "") -> Generator[tuple[str, str]]:
        """This lake served over Quack on localhost `port` while the block runs, yielding the
        URI and the token a client attaches with.

        token: what clients must present; the kept one (else a fresh one, then kept) when empty.
        DuckDB is opened on the catalog itself, the one database a Quack client resolves names
        in, after an ordinary attach brought the schema up to date. Appends from this machine
        keep going straight to the files beside it, as they always do.
        """
        self._held("serve")
        with self.ready().open(write=True) as connection:
            self.evolve(connection)
            self._load(connection, _QUACK)
        self._rehome()
        token = token or self._kept_token()
        uri = f"quack:localhost:{port}"
        with (
            FileBudget.permitted(),
            closing(
                duckdb.connect(
                    f"ducklake:sqlite:{self.catalog.as_posix()}",
                    config={
                        "extension_directory": self.extensions.as_posix(),
                        "autoinstall_known_extensions": False,
                    },
                )
            ) as served,
        ):
            try:
                served.execute(f"SET GLOBAL ducklake_max_retry_count = {_RETRIES}")
                served.execute("SET GLOBAL TimeZone = 'UTC'")
                self._load(served, _QUACK)
                served.execute(f"CALL quack_serve({quoted(uri)}, token => {quoted(token)})")
                yield uri, token
                served.execute(f"CALL quack_stop({quoted(uri)})")
            except duckdb.Error as fault:
                raise MissionError(
                    f"could not serve the lake at {uri}: {_first(fault)}"
                ) from fault

    def _kept_token(self) -> str:
        """The token this lake is served with, made and kept on first use."""
        if not self.token.is_file():
            self.token.parent.mkdir(parents=True, exist_ok=True)
            self.token.write_text(secrets.token_urlsafe(24), encoding="utf-8", newline="\n")
        return self.token.read_text(encoding="utf-8").strip()

    def _rehome(self) -> None:
        """Point the catalog's recorded data path at this workspace's data folder.

        Serving opens the catalog as it is, and a catalog carried from another machine records
        that machine's folder, which an ordinary attach overrides and serving cannot. Files are
        recorded relative to the data path, so this one row is all that moves.
        """
        here = self.data.as_posix() + "/"
        with FileLock(self.lock), sqlite3.connect(self.catalog) as catalog:
            catalog.execute(
                "UPDATE ducklake_metadata SET value = ? "
                "WHERE key = 'data_path' AND scope IS NULL AND value <> ?",
                (here, here),
            )
        catalog.close()

    def _reach(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Attach the lake served at `served` as `lake`, saying plainly why when it cannot be."""
        self._load(connection, _QUACK)
        token = Project().variable("LAKE_TOKEN").read()
        options = f" (TOKEN {quoted(token)})" if token else ""
        try:
            connection.execute(f"ATTACH {quoted(self.served)} AS {ALIAS}{options}")
        except duckdb.Error as fault:
            said = _first(fault)
            if "deserialize" in said.lower():
                said = (
                    f"it answers in a protocol DuckDB {duckdb.__version__} here cannot read; "
                    "both ends must run the same DuckDB release"
                )
            elif "connect" in said.lower():
                said = "nothing answers there; start `mb lake serve` and the ssh tunnel to it"
            raise MissionError(f"cannot reach the lake at {self.served}: {said}") from fault

    def _held(self, verb: str) -> None:
        """Refuse `verb` on a served lake: it works on the files, on the machine holding them."""
        if self.served:
            raise MissionError(
                f"the lake is served from {self.served}; `{verb}` works on its files, so run it "
                "on the machine holding them"
            )

    def _load(self, connection: duckdb.DuckDBPyConnection, names: Sequence[str]) -> None:
        """Load the extensions from this tool's directory, installing only what fails to load.

        Raises MissionError naming the directory when an extension is neither there nor
        installable, as on a machine that never went online.
        """
        connection.execute(f"SET extension_directory = {quoted(self.extensions.as_posix())}")
        for name in names:
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


def _first(fault: BaseException) -> str:
    """The first line of what `fault` says."""
    return (str(fault).splitlines() or [""])[0]


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
        # releases the catalog.
        weakref.finalize(self, self._stack.close)

    def close(self) -> None:
        """Detach now; the next statement attaches again."""
        with self._turn:
            self._stack.close()
            self._connection = None

    def rows(
        self, statement: ClauseElement | str, parameters: Iterable[object] = ()
    ) -> list[tuple[Any, ...]]:
        """What `statement` answers in this session."""
        bound = list(parameters)
        return self.run(
            lambda connection: self.lake.execute(connection, statement, bound).fetchall()
        )

    def append(self, table: Table, rows: Iterable[Mapping[str, object]]) -> int:
        """Append `rows` to `table` in one commit, returning how many were appended."""
        staged = list(rows)
        return self.run(lambda connection: insert(connection, table, staged))

    def run[T](self, statement: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """`statement` over this session's connection, reattached while the lake refuses it.

        One statement at a time: a DuckDB connection is not safe across threads (a sampler
        publishes from its own thread while its owner reads the stream), and a per-thread cursor
        would be closed under its thread by the reattach a stale catalog snapshot needs.
        """
        with self._turn, FileBudget.permitted():
            return _patiently(recoverable, before=self.close)(lambda: statement(self._attached()))

    def _attached(self) -> duckdb.DuckDBPyConnection:
        """The connection, attaching first; a fresh workspace's lake is created on this use and
        an older release's lake brought up to this schema."""
        if self._connection is None:
            connection = self._stack.enter_context(
                self.lake.ready()._attached(write=True, create=False)
            )
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
    newest = selecting(func.max(schema.schema_log.c.version))
    row = connection.execute(*compiled(newest)).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _spec(connection: duckdb.DuckDBPyConnection) -> str:
    """The attached lake's DuckLake spec version."""
    row = connection.execute(
        f"SELECT value FROM {ALIAS}.options() WHERE option_name = 'version'"
    ).fetchone()
    return str(row[0]) if row else ""


def _set_apart(connection: duckdb.DuckDBPyConnection) -> None:
    """Give every `_BYTES` table its own options."""
    for table in _BYTES:
        for option, value in _BYTES_OPTIONS:
            connection.execute(
                f"CALL {ALIAS}.set_option('{option}', {value}, table_name => '{table.name}')"
            )


def _record(connection: duckdb.DuckDBPyConnection, spec: str) -> None:
    """Record that the schema at `VERSION` now lives in a lake at `spec`, and by which engine."""
    engine = connection.execute("SELECT library_version FROM pragma_version()").fetchone()
    insert(
        connection,
        schema.schema_log,
        [
            {
                "ts": datetime.now(UTC),
                "version": VERSION,
                "spec": spec,
                "engine": engine[0] if engine else "",
            }
        ],
    )
