"""A lake's catalog kept: created at the extension's latest spec, evolved to this release's
schema, upgraded on request, checked against the files it references, and compacted."""

import sqlite3
import struct
from collections.abc import Iterator, Set
from contextlib import closing, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import duckdb
from filelock import FileLock, Timeout
from patos import FrozenModel
from sqlalchemy import func, literal_column
from sqlalchemy import select as selecting

from ...core.errors import MissionError
from .. import schema
from ..constants import ALIAS, VERSION
from ..schema import ddl, kind
from ..storage.database import first_line
from .attachment import Access, Attachment
from .session import patiently
from .staging import compiled, insert

_HELD_COLUMNS_SQL = "SELECT table_name, column_name FROM duckdb_columns() WHERE database_name = ?"
_ALIAS_TABLE_NAMES_SQL = (
    "SELECT table_name FROM duckdb_tables() WHERE database_name = ? ORDER BY table_name"
)
_TABLE_FILES_SQL = "SELECT data_file, delete_file FROM ducklake_list_files(?, ?)"
_DUCKDB_VERSION_SQL = "SELECT library_version FROM pragma_version()"

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
# 24 GB center out of memory merging 27k blob files. The Parquet writer consumes whole vectors,
# so a row limit alone cannot split a vector of 32 large chunks. A one-byte flush threshold
# makes every input vector from `blobs._Batch` its own group. The batch bounds those vectors
# by payload bytes, splitting below the writer's minimum 2,048 rows only for large chunks.
_BYTES = (schema.blobs,)
_BYTES_OPTIONS = (
    ("data_inlining_row_limit", "0"),
    ("parquet_row_group_size", "16"),
    ("parquet_row_group_size_bytes", "'1B'"),
    ("target_file_size", "'64MB'"),
)

# The catalog size past which `check` suggests a compaction.
_CATALOG_FLOOR_BYTES = 64 << 20

# The SQLite WAL-index header fields `check` reads from the catalog's `-shm` file: `mxFrame`, the
# last valid frame in the WAL, at byte 16, and `nBackfill`, how many frames were already copied
# into the database, at byte 96. Both are native-endian 32-bit integers.
_MAX_FRAME = 16
_BACKFILL = 96


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


class Maintained(Attachment):
    """A lake's catalog, kept at this release's schema and checked against its files.

    A lake is created at the extension's latest spec, never a pinned one, and records it.
    Opening never migrates a catalog: an older one keeps working at its spec until `upgrade` is
    asked for, since a migration is a one-way step every other reader must be ready for.
    """

    @staticmethod
    def _additions(held: Set[tuple[str, str]]) -> list[str]:
        """The DDL adding each table and column of this release's schema a lake does not hold."""
        tables = {table for table, _ in held}
        return [ddl(table) for table in schema.tables if table.name not in tables] + [
            f"ALTER TABLE {table.name} ADD COLUMN {column.name} {kind(column)}"
            for table in schema.tables
            if table.name in tables
            for column in table.columns
            if (table.name, column.name) not in held
        ]

    def _bloated(self) -> Iterator[Finding]:
        """A finding when the catalog grew past what a compaction would leave."""
        if (size := self.catalog.stat().st_size) > _CATALOG_FLOOR_BYTES:
            yield Finding(
                kind="bloat",
                detail=f"the catalog holds {size >> 20} MB; `mb lake compact` flushes it to "
                "Parquet and gives the space back",
            )

    def _compacted(self) -> None:
        """Checkpoint under the maintenance lock, then give the catalog's freed pages back.

        Raises Timeout when another process holds the lock.
        """
        with FileLock(self.lock, timeout=0):
            for attempt in patiently():
                with attempt, self.open(write=True) as connection:
                    connection.execute(f"CHECKPOINT {ALIAS}")
            with (
                suppress(sqlite3.OperationalError),
                closing(sqlite3.connect(self.catalog, timeout=0)) as catalog,
            ):
                catalog.execute("VACUUM")

    def _evolved(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Add what this release's schema holds and the lake lacks, then its views and log row."""
        held = set(
            connection.execute(
                _HELD_COLUMNS_SQL,
                [ALIAS],
            ).fetchall()
        )
        connection.execute("BEGIN")
        connection.execute(f"USE {ALIAS}")
        for statement in Maintained._additions(held):
            connection.execute(statement)
        _set_apart(connection)
        _described(connection)
        connection.execute("COMMIT")
        connection.execute("USE memory")

    def _missing(self) -> Iterator[Finding]:
        """A finding for every data or delete file the catalog references that the disk lacks."""
        with self.open() as connection:
            tables = connection.execute(
                _ALIAS_TABLE_NAMES_SQL,
                [ALIAS],
            ).fetchall()
            for (table,) in tables:
                listed = connection.execute(
                    _TABLE_FILES_SQL,
                    [ALIAS, table],
                ).fetchall()
                yield from (
                    Finding(table=table, kind="missing", detail=path)
                    for pair in listed
                    for path in pair
                    if path and not Path(path).is_file()
                )

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

    def check(self) -> Health:
        """Compare what the catalog references with what is on disk, per table.

        A deleted data file breaks only its table, and `count(*)` answers from the catalog so
        it hides the loss; this is what finds it. A lost catalog WAL is reported only when the
        WAL index proves frames were committed there and never copied into the catalog.
        """
        self._held("check")
        if not self.exists():
            return Health(findings=(Finding(kind="catalog", detail=f"{self.catalog} missing"),))
        findings = [*self._wal(), *self._bloated()]
        try:
            findings += self._missing()
        except (duckdb.Error, MissionError) as fault:
            findings.append(Finding(kind="catalog", detail=first_line(fault)))
        return Health(findings=tuple(findings))

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
            with self.attached(access=Access.CREATE) as connection:
                connection.execute("BEGIN")
                for option, value in _OPTIONS:
                    connection.execute(f"CALL {ALIAS}.set_option('{option}', {value})")
                connection.execute(f"USE {ALIAS}")
                for table in schema.tables:
                    connection.execute(ddl(table))
                _set_apart(connection)
                spec = _described(connection)
                connection.execute("COMMIT")
        return spec

    def current(self) -> Self:
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

    def evolve(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Bring a lake an older release created up to this release's schema, once.

        Tables and columns are only ever added and views replaced, so a lake at an older schema
        version loses nothing; the step is recorded in `schema_log` and taken under the
        maintenance lock so two processes never race it.
        """
        if self.served or _version(connection) >= VERSION:
            return
        with FileLock(self.lock):
            if _version(connection) < VERSION:
                self._evolved(connection)

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
            self._compacted()
        except Timeout:
            return False
        return True

    def ready(self) -> Self:
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

    def spec(self) -> str:
        """The DuckLake spec the catalog is at."""
        with self.open() as connection:
            return _spec(connection)

    def upgrade(self) -> str:
        """Migrate the catalog to the extension's latest spec, returning the spec it is now at.

        Recorded in `schema_log` when the spec moved, so the lake says which releases wrote it.
        """
        self._held("upgrade")
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self.lock), self.attached(access=Access.MIGRATE) as connection:
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


def _described(connection: duckdb.DuckDBPyConnection) -> str:
    """Replace every view and record the schema at its spec, answering the spec."""
    for view in schema.views:
        connection.execute(view.ddl)
    spec = _spec(connection)
    _record(connection, spec)
    return spec


def _record(connection: duckdb.DuckDBPyConnection, spec: str) -> None:
    """Record that the schema at `VERSION` now lives in a lake at `spec`, and by which engine."""
    engine = connection.execute(_DUCKDB_VERSION_SQL).fetchone()
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


def _set_apart(connection: duckdb.DuckDBPyConnection) -> None:
    """Give every `_BYTES` table its own options, in the transaction that creates or evolves it."""
    settings = ", ".join(f"{option} = {value}" for option, value in _BYTES_OPTIONS)
    for table in _BYTES:
        connection.execute(f"ALTER TABLE {ALIAS}.{table.name} SET ({settings})")


def _spec(connection: duckdb.DuckDBPyConnection) -> str:
    """The attached lake's DuckLake spec version."""
    row = connection.execute(
        f"SELECT value FROM {ALIAS}.options() WHERE option_name = 'version'"
    ).fetchone()
    return str(row[0]) if row else ""


def _version(connection: duckdb.DuckDBPyConnection) -> int:
    """The newest schema version the attached lake records."""
    newest = selecting(func.max(schema.schema_log.c.version))
    row = connection.execute(*compiled(newest)).fetchone()
    return int(row[0]) if row and row[0] is not None else 0
