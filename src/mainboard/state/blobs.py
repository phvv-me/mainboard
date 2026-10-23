# CONTENT-ADDRESSED BYTES IN THE LAKE, the one reader and writer of its `blobs` and `checksums`.
#
# An object is its SHA-256 and its bytes in ordered chunks of `CHUNK_BYTES`, so a multi-gigabyte
# evidence file fits rows DuckDB stages and Parquet stores comfortably, while a source file stays
# the single chunk it always was. A row written before chunking has no ordinal and reads as chunk
# zero, the whole object. The bytes are stored exactly as they came: a Parquet receipt is never
# decoded or re-encoded, so every digest pinned anywhere keeps verifying.
#
# STAGING. Chunks are gathered across objects and land in one insert per `STAGED_BYTES`, since
# DuckLake writes a Parquet file per insert: staged one object at a time, thousands of small
# evidence files became thousands of 34 KB data files at 68 MB a minute. They travel the way every
# lake insert does (`lake.staged`), since DuckDB copies a whole BLOB on every slice of it, so
# cutting chunks out of one spooled file in SQL ran out of memory. They are staged inside the
# caller's transaction, so they commit together or not at all.
#
# CUSTODY. An object is held only when it is intact, and one verifier decides that for every
# reader and every destructive step: the ordinals run from zero without a gap, the copies of a
# chunk agree with each other and with each checksum recorded for it, and all the chunks together
# hash to the digest. Identical copies from two writers that raced on one object are one chunk; a
# damaged object is not held, and staging it again replaces what the lake kept of it, inside the
# staging transaction. Each chunk's MD5 lands in `checksums` in the same insert, computed by DuckDB
# from the very bytes whose SHA-256 was just verified. MD5 because a checksum must be one every
# DuckDB build computes: the `hashfuncs` community extension (xxh3) has no build for DuckDB 2.0.0,
# and DuckDB's own `hash` may change between releases. A
# chunk kept before checksums existed leans on the whole digest alone, and nothing here records a
# checksum for it: reading never writes.

import hashlib
import os
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from io import BytesIO
from itertools import batched, groupby
from operator import itemgetter
from pathlib import Path
from typing import IO

import duckdb
from patos import FrozenModel
from sqlalchemy import func, select

from ..core.errors import MissionError
from . import schema
from .lake import ALIAS, Lake, staged
from .owned import OwnedFile

# One chunk of an object, and the most bytes of chunks one insert stages.
CHUNK_BYTES = 8 << 20
STAGED_BYTES = 256 << 20
# A Parquet group holds at most this many payload bytes. Tiny chunks can use ordinary
# 2,048-row vectors; large chunks need smaller input vectors to honor the byte budget.
_ROW_GROUP_BYTES = 16 << 20
_VECTOR_ROWS = 2048

_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# The temporary table a staged insert's chunks are decoded into, read once for `blobs` and once
# for `checksums`.
_STAGED = "staged_chunks"

# One stored copy of a chunk: its digest, ordinal and bytes (None when the cell was lost), the MD5
# DuckDB computes from those bytes now and every MD5 recorded for the chunk (None for a chunk
# kept before checksums).
type _Copy = tuple[str, int, bytes | None, str | None, list[str] | None]


class Audit(FrozenModel):
    """What the lake holds of one object, verified through its complete byte stream.

    chunks: distinct ordinals held, and `last` the highest; a gap shows as `chunks <= last`,
        a lost last chunk only as a short `size`.
    size: bytes across those chunks, one copy each.
    intact: contiguous ordinals, agreeing copies and checksums, and a matching whole SHA-256.
    """

    chunks: int
    last: int
    size: int
    intact: bool

    def fault(self, size: int) -> str:
        """What keeps this from being the intact `size`-byte object, empty when nothing does."""
        if self.chunks != self.last + 1:
            return f"{self.last + 1 - self.chunks} of its {self.last + 1} chunks are missing"
        if self.size != size:
            return f"the lake holds {self.size} of its {size} bytes"
        return "" if self.intact else "a chunk fails its checksum"


class Blobs:
    """The lake's content-addressed objects, each once, as ordered chunks."""

    def __init__(self, lake: Lake) -> None:
        self.lake = lake

    def audit(
        self, connection: duckdb.DuckDBPyConnection, digests: Collection[str]
    ) -> dict[str, Audit]:
        """Verify each object among `digests` the lake holds any row of; the absent are left out.

        Writes nothing: not a byte, a row or a checksum.
        """
        return {
            digest: self._verify(digest, copies)
            for digest, copies in self._objects(connection, digests)
        }

    def chunks(self, connection: duckdb.DuckDBPyConnection, digest: str) -> Iterator[bytes]:
        """`digest`'s chunks in order, one query each, so memory holds one chunk at a time.

        An object's chunks share one staging, so the catalog's per-file statistics find them
        without reading the rest of the table. Unverified: the caller hashes as it goes.
        """
        kept = schema.blobs
        ordinal = func.coalesce(kept.c.ordinal, 0)
        for number in self._ordinals(connection, digest):
            row = self.lake.execute(
                connection,
                select(kept.c.bytes).where(kept.c.sha256 == digest, ordinal == number).limit(1),
            ).fetchone()
            if row is not None:
                yield row[0]

    def held(self, connection: duckdb.DuckDBPyConnection, digests: Collection[str]) -> set[str]:
        """The digests among `digests` the lake holds an intact object for."""
        return {
            digest for digest, audit in self.audit(connection, digests).items() if audit.intact
        }

    def inspect(
        self, connection: duckdb.DuckDBPyConnection, digest: str, sink: IO[bytes] | None = None
    ) -> Audit:
        """Verify one object while writing its bytes to `sink`; one the lake lacks is no object."""
        for _, copies in self._objects(connection, [digest]):
            return self._verify(digest, copies, sink)
        return Audit(chunks=0, last=-1, size=0, intact=False)

    def read(
        self, connection: duckdb.DuckDBPyConnection, digests: Collection[str]
    ) -> dict[str, bytes]:
        """The intact objects among `digests`, by digest; one absent or damaged is left out."""
        found: dict[str, bytes] = {}
        for digest, copies in self._objects(connection, digests):
            with BytesIO() as sink:
                if self._verify(digest, copies, sink).intact:
                    found[digest] = sink.getvalue()
        return found

    def stage(
        self, connection: duckdb.DuckDBPyConnection, objects: Mapping[str, bytes | Path]
    ) -> int:
        """Stage each digest's object, bytes already read or a file streamed from disk, as its
        chunks and their checksums; how many bytes they were.

        Staged inside the caller's open transaction, one insert per `STAGED_BYTES`. A damaged
        object the lake already holds under one of these digests is dropped first, so the
        transaction replaces it whole or leaves it as it was. Raises MissionError when a file's
        bytes no longer hash to its digest, so the caller's transaction is abandoned rather than
        committing an object under the wrong name.
        """
        damaged = [
            digest
            for digest, copies in self._objects(connection, objects)
            if not self._verify(digest, copies).intact
        ]
        for table in ("blobs", "checksums") if damaged else ():
            self.lake.execute(
                connection,
                f"DELETE FROM {ALIAS}.{table} WHERE sha256 IN (SELECT unnest(?::VARCHAR[]))",
                [damaged],
            )
        batch = _Batch(connection)
        size = sum(batch.keep(digest, path) for digest, path in objects.items())
        batch.flush()
        return size

    def write(
        self,
        connection: duckdb.DuckDBPyConnection,
        digest: str,
        target: Path,
        *,
        root: Path | None = None,
        size: int | None = None,
        replace: bool = False,
    ) -> bool:
        """Write `digest`'s object to `target` under `root` (its folder by default).

        False when the lake holds no intact copy of the right `size`. The bytes are verified in a
        staged file beside `target`, then published whole. A different file already at `target`
        is refused with MissionError, unless `replace` lets a cache drop what went stale.
        """
        owned = OwnedFile(target, root=root or target.parent)
        with (
            owned.parent(create=True) as directory,
            owned.staging(directory=directory) as (staged_name, stream),
        ):
            audit = self.inspect(connection, digest, stream)
            stream.flush()
            os.fsync(stream.fileno())
            if not audit.intact or (size is not None and audit.size != size):
                return False
            owned.publish(staged_name, directory=directory, digest=digest, replace=replace)
            return True

    @staticmethod
    def _verify(digest: str, copies: Iterable[_Copy], sink: IO[bytes] | None = None) -> Audit:
        """Judge an object from its stored copies in ordinal order, writing its bytes to `sink`.

        Intact needs ordinals from zero without a gap, every copy of a chunk the same bytes and
        agreeing with each recorded checksum, and the whole hashing to `digest`; a chunk with no
        recorded checksum leans on that SHA-256 alone.
        """
        hasher, size, chunks, last, intact = hashlib.sha256(), 0, 0, -1, True
        for ordinal, group in groupby(copies, key=itemgetter(1)):
            stored = list(group)
            payload = stored[0][2] or b""
            intact = (
                intact
                and ordinal == chunks
                and all(
                    copy[2] == payload and all(recorded == copy[3] for recorded in copy[4] or ())
                    for copy in stored
                )
            )
            hasher.update(payload)
            size, chunks, last = size + len(payload), chunks + 1, ordinal
            if sink is not None:
                sink.write(payload)
        return Audit(
            chunks=chunks,
            last=last,
            size=size,
            intact=intact and chunks > 0 and hasher.hexdigest() == digest,
        )

    def _copies(
        self,
        connection: duckdb.DuckDBPyConnection,
        digests: Collection[str],
        ordinal: int | None = None,
    ) -> list[_Copy]:
        """Every stored copy of the chunks of `digests`, or of their chunk `ordinal` alone."""
        listed = _digests(digests)
        return self.lake.execute(
            connection,
            "SELECT b.sha256, coalesce(b.ordinal, 0), b.bytes, md5(b.bytes), c.sums "
            f"FROM {ALIAS}.blobs b LEFT JOIN (SELECT sha256, ordinal, list(DISTINCT md5) AS sums "
            f"FROM {ALIAS}.checksums WHERE sha256 IN (SELECT unnest(?::VARCHAR[])) "
            "GROUP BY ALL) c ON c.sha256 = b.sha256 AND c.ordinal = coalesce(b.ordinal, 0) "
            "WHERE b.sha256 IN (SELECT unnest(?::VARCHAR[])) "
            "AND (?::BIGINT IS NULL OR coalesce(b.ordinal, 0) = ?::BIGINT)",
            [listed, listed, ordinal, ordinal],
        ).fetchall()

    def _objects(
        self, connection: duckdb.DuckDBPyConnection, digests: Collection[str]
    ) -> Iterator[tuple[str, Iterable[_Copy]]]:
        """Each object among `digests` with any row, with its stored copies in ordinal order.

        Objects come together in one query, except one of more chunks than a staging window
        holds, which is read a chunk at a time so memory never holds the whole of it.
        """
        if not digests:
            return
        counted = dict(
            self.lake.execute(
                connection,
                "SELECT sha256, count(DISTINCT coalesce(ordinal, 0)) "
                f"FROM {ALIAS}.blobs WHERE sha256 IN (SELECT unnest(?::VARCHAR[])) "
                "GROUP BY sha256",
                [_digests(digests)],
            ).fetchall()
        )
        window = STAGED_BYTES // CHUNK_BYTES
        stored: dict[str, list[_Copy]] = {
            digest: [] for digest, chunks in counted.items() if chunks <= window
        }
        for copy in self._copies(connection, stored) if stored else ():
            stored[copy[0]].append(copy)
        for digest, copies in stored.items():
            yield digest, sorted(copies, key=itemgetter(1))
        for digest in counted.keys() - stored.keys():
            yield digest, self._streamed(connection, digest)

    def _ordinals(self, connection: duckdb.DuckDBPyConnection, digest: str) -> list[int]:
        kept = schema.blobs
        ordinal = func.coalesce(kept.c.ordinal, 0)
        rows = self.lake.execute(
            connection, select(ordinal).distinct().where(kept.c.sha256 == digest).order_by(ordinal)
        ).fetchall()
        return [number for (number,) in rows]

    def _streamed(self, connection: duckdb.DuckDBPyConnection, digest: str) -> Iterator[_Copy]:
        for number in self._ordinals(connection, digest):
            yield from self._copies(connection, [digest], number)


class _Batch:
    """Chunks on their way into one insert, and how many bytes they hold."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self.rows: list[dict[str, str | int | bytes]] = []
        self.size = 0

    def flush(self) -> None:
        """Insert every waiting chunk into `blobs` and its checksum into `checksums`."""
        if not self.rows:
            return
        with staged(schema.blobs, self.rows) as staging:
            self.connection.execute(
                f"CREATE OR REPLACE TEMP TABLE {_STAGED} AS "
                f"SELECT *, row_number() OVER () - 1 AS position FROM ({staging})"
            )
        # Parquet flushes only between input vectors; a one-byte threshold flushes every
        # vector. Bound their bytes, splitting only when an ordinary vector would be too big.
        options = dict(
            self.connection.execute(
                f"SELECT option_name, value FROM {ALIAS}.options() "
                "WHERE option_name = 'parquet_row_group_size_bytes' "
                "AND (scope = 'GLOBAL' OR (scope = 'SCHEMA' AND scope_entry = 'main') "
                "OR (scope = 'TABLE' AND scope_entry = 'main.blobs')) "
                "ORDER BY CASE scope WHEN 'GLOBAL' THEN 0 WHEN 'SCHEMA' THEN 1 ELSE 2 END"
            ).fetchmany(6)
        )
        if "parquet_row_group_size_bytes" not in options:
            # Existing schema-current lakes need this correction without a schema bump.
            self.connection.execute(
                f"CALL {ALIAS}.set_option('parquet_row_group_size_bytes', '1B', "
                "table_name => 'blobs')"
            )
        self.connection.execute(f"INSERT INTO {ALIAS}.blobs BY NAME {self._source()}")
        self.connection.execute(
            f"INSERT INTO {ALIAS}.checksums BY NAME SELECT sha256, ordinal, md5(bytes) AS md5 "
            f"FROM {_STAGED}"
        )
        self.connection.execute(f"DROP TABLE {_STAGED}")
        self.rows, self.size = [], 0

    def _source(self) -> str:
        """One scan for small chunks; byte-sized windows for large or mixed chunks.

        Each window scans the staged table, so partition by actual payload bytes rather than
        the largest chunk's row limit: even a mixed batch has only O(staged bytes / budget)
        UNION arms, and cannot turn a million tiny objects into a quadratic UNION tree.
        """
        source = f"SELECT sha256, ordinal, bytes FROM {_STAGED}"
        sizes = []
        for row in self.rows:
            payload = row["bytes"]
            assert isinstance(payload, bytes)
            sizes.append(len(payload))
        group_rows = _ROW_GROUP_BYTES // max(1, max(sizes))
        if group_rows >= _VECTOR_ROWS:
            return source
        sources = []
        start = size = 0
        for position, chunk_size in enumerate(sizes):
            if size and size + chunk_size > _ROW_GROUP_BYTES:
                sources.append(f"{source} WHERE position >= {start} AND position < {position}")
                start, size = position, 0
            size += chunk_size
        sources.append(f"{source} WHERE position >= {start}")
        while len(sources) > 1:
            sources = [
                "(" + ") UNION ALL (".join(pair) + ")"
                for pair in batched(sources, 2, strict=False)
            ]
        return sources[0]

    def keep(self, digest: str, source: bytes | Path) -> int:
        """Add `source`'s chunks as `digest`'s, returning how many bytes it held.

        Inserts whenever `STAGED_BYTES` are waiting. Raises MissionError when the chunks do not
        hash to `digest`.
        """
        hasher, size = hashlib.sha256(), 0
        for ordinal, payload in enumerate(_pieces(source)):
            hasher.update(payload)
            self.rows.append({"sha256": digest, "ordinal": ordinal, "bytes": payload})
            self.size += len(payload)
            size += len(payload)
            if self.size >= STAGED_BYTES:
                self.flush()
        if hasher.hexdigest() != digest:
            where = source if isinstance(source, Path) else "an object"
            raise MissionError(f"{where} no longer hashes to {digest}; not kept")
        return size


def _pieces(source: bytes | Path) -> Iterator[bytes]:
    """`source`'s bytes in `CHUNK_BYTES` pieces; an empty object is one empty chunk, so it is
    held."""
    if isinstance(source, bytes):
        yield from (
            source[at : at + CHUNK_BYTES] for at in range(0, len(source) or 1, CHUNK_BYTES)
        )
        return
    with source.open("rb") as stream:
        payload = stream.read(CHUNK_BYTES)
        yield payload
        while payload := stream.read(CHUNK_BYTES):
            yield payload


def _digests(digests: Collection[str]) -> list[str]:
    """A validated relation input; scalar IN lists exceed DuckLake's expression depth."""
    for digest in digests:
        if not _DIGEST.fullmatch(digest):
            raise ValueError(f"not a SHA-256 digest: {digest!r}")
    return sorted(digests)
