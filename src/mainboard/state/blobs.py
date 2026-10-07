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
# caller's transaction, so they commit together or not at all: an object is held once any row of
# it is. Two writers racing on one object each add the same chunks, which a read takes once per
# ordinal. A read hands back only bytes that hash to the digest asked for, so a damaged object
# reads as absent rather than as wrong bytes.
#
# CHECKSUMS. Each staged chunk's MD5 lands in `checksums` in the same insert's transaction,
# computed by DuckDB from the very bytes whose SHA-256 was just verified, so `audit` proves every
# chunk intact by recomputing it in SQL, in parallel, without a byte crossing into Python (the
# SHA-256 of a chunked object cannot be recomputed that way, since it runs across chunks). MD5
# because a checksum must be one every DuckDB build computes: the `hashfuncs` community extension
# (xxh3) has no build for the DuckDB 2.0 development release this tool runs, and DuckDB's own
# `hash` may change between releases. A chunk written before checksums existed gets its checksum
# from the first audit's read, trusting the bytes as their ingest verified them.

import hashlib
import logging
import os
import re
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO

import duckdb
from patos import FrozenModel
from sqlalchemy import func, select

from ..core.errors import MissionError
from . import schema
from .lake import ALIAS, Lake, staged

# One chunk of an object, and the most bytes of chunks one insert stages.
CHUNK_BYTES = 8 << 20
STAGED_BYTES = 256 << 20

_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# The temporary tables a staged insert's chunks are decoded into (read once for `blobs` and once
# for `checksums`), and an audit's recomputed checksums are kept in, one narrow row per chunk.
_STAGED = "staged_chunks"
_SUMMED = "summed_chunks"

logger = logging.getLogger(__name__)


class Audit(FrozenModel):
    """What the lake holds of one object, found without reading its bytes into Python.

    chunks: distinct ordinals held, and `last` the highest; a gap shows as `chunks <= last`,
        a lost last chunk only as a short `size`.
    size: bytes across those chunks, one copy each.
    intact: whether every copy of every chunk matches its recorded checksum.
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

    def audit(self, connection: duckdb.DuckDBPyConnection) -> dict[str, Audit]:
        """Every object the lake holds, each chunk's checksum recomputed by DuckDB in parallel.

        Reads every byte of `blobs` once, inside DuckDB. A chunk with no recorded checksum yet
        (written before checksums existed) has it recorded from that same read, trusting the
        bytes as their ingest verified them.
        """
        self.lake.execute(
            connection,
            f"CREATE OR REPLACE TEMP TABLE {_SUMMED} AS SELECT sha256, coalesce(ordinal, 0) "
            f"AS ordinal, octet_length(bytes) AS size, md5(bytes) AS md5 FROM {ALIAS}.blobs",
        )
        recorded = self.lake.execute(
            connection,
            f"INSERT INTO {ALIAS}.checksums BY NAME SELECT DISTINCT ON (sha256, ordinal) "
            f"sha256, ordinal, md5 FROM {_SUMMED} s WHERE NOT EXISTS (SELECT 1 FROM "
            f"{ALIAS}.checksums c WHERE c.sha256 = s.sha256 AND c.ordinal = s.ordinal)",
        ).fetchone()
        if recorded and recorded[0]:
            logger.info("recorded the checksum of %d chunks kept before checksums", recorded[0])
        rows = self.lake.execute(
            connection,
            "SELECT sha256, count(*), max(ordinal), sum(size), bool_and(ok) FROM ("
            "SELECT s.sha256, s.ordinal, any_value(s.size) AS size, "
            "bool_and(coalesce(s.md5 = c.md5, false)) AS ok "
            f"FROM {_SUMMED} s LEFT JOIN {ALIAS}.checksums c "
            "ON c.sha256 = s.sha256 AND c.ordinal = s.ordinal GROUP BY ALL) GROUP BY sha256",
        ).fetchall()
        self.lake.execute(connection, f"DROP TABLE {_SUMMED}")
        return {
            digest: Audit(chunks=chunks, last=last, size=size, intact=intact)
            for digest, chunks, last, size, intact in rows
        }

    def chunks(self, connection: duckdb.DuckDBPyConnection, digest: str) -> Iterator[bytes]:
        """`digest`'s chunks in order, one query each, so memory holds one chunk at a time.

        An object's chunks share one staging, so the catalog's per-file statistics find them
        without reading the rest of the table. Unverified: `write` hashes as it goes.
        """
        kept = schema.blobs
        ordinal = func.coalesce(kept.c.ordinal, 0)
        object_ = kept.c.sha256 == digest
        ordinals = self.lake.execute(
            connection, select(ordinal).distinct().where(object_).order_by(ordinal)
        ).fetchall()
        for (number,) in ordinals:
            row = self.lake.execute(
                connection, select(kept.c.bytes).where(object_, ordinal == number).limit(1)
            ).fetchone()
            if row is not None:
                yield row[0]

    def held(self, connection: duckdb.DuckDBPyConnection, digests: Collection[str]) -> set[str]:
        """The digests among `digests` the lake holds an object for."""
        if not digests:
            return set()
        found = self.lake.execute(
            connection,
            f"SELECT DISTINCT sha256 FROM {ALIAS}.blobs "
            "WHERE sha256 IN (SELECT unnest(?::VARCHAR[]))",
            [_digests(digests)],
        )
        return {digest for (digest,) in found.fetchall()}

    def read(
        self, connection: duckdb.DuckDBPyConnection, digests: Collection[str]
    ) -> dict[str, bytes]:
        """The intact objects among `digests`, by digest; one absent or damaged is left out."""
        if not digests:
            return {}
        rows = self.lake.execute(
            connection,
            f"SELECT DISTINCT ON (sha256, o) sha256, o, bytes FROM (SELECT sha256, "
            f"coalesce(ordinal, 0) AS o, bytes FROM {ALIAS}.blobs WHERE sha256 IN "
            "(SELECT unnest(?::VARCHAR[]))) ORDER BY sha256, o",
            [_digests(digests)],
        ).fetchall()
        assembled: dict[str, list[bytes]] = {}
        for digest, _, payload in rows:
            assembled.setdefault(digest, []).append(payload)
        wholes = {digest: b"".join(parts) for digest, parts in assembled.items()}
        return {
            digest: whole
            for digest, whole in wholes.items()
            if hashlib.sha256(whole).hexdigest() == digest
        }

    def stage(
        self, connection: duckdb.DuckDBPyConnection, objects: Mapping[str, bytes | Path]
    ) -> int:
        """Append each digest's object, bytes already read or a file streamed from disk, as its
        chunks and their checksums; how many bytes they were.

        Staged inside the caller's open transaction, one insert per `STAGED_BYTES`. Raises
        MissionError when a file's bytes no longer hash to its digest, so the caller's
        transaction is abandoned rather than committing an object under the wrong name.
        """
        batch = _Batch(connection)
        size = sum(batch.keep(digest, path) for digest, path in objects.items())
        batch.flush()
        return size

    def write(self, connection: duckdb.DuckDBPyConnection, digest: str, target: Path) -> bool:
        """Write `digest`'s object to `target` one chunk at a time, False when the lake holds no
        intact copy; `target` is replaced only by bytes that verified."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            whole = self._copy(connection, digest, stream)
        if not whole:
            temporary.unlink()
            return False
        os.replace(temporary, target)
        return True

    def _copy(self, connection: duckdb.DuckDBPyConnection, digest: str, sink: IO[bytes]) -> bool:
        """Stream `digest`'s object into `sink`, whether it was held and intact."""
        hasher, seen = hashlib.sha256(), False
        for payload in self.chunks(connection, digest):
            hasher.update(payload)
            seen = True
            sink.write(payload)
        return seen and hasher.hexdigest() == digest


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
            self.connection.execute(f"CREATE OR REPLACE TEMP TABLE {_STAGED} AS {staging}")
        self.connection.execute(
            f"INSERT INTO {ALIAS}.blobs BY NAME SELECT sha256, ordinal, bytes FROM {_STAGED}"
        )
        self.connection.execute(
            f"INSERT INTO {ALIAS}.checksums BY NAME SELECT sha256, ordinal, md5(bytes) AS md5 "
            f"FROM {_STAGED}"
        )
        self.connection.execute(f"DROP TABLE {_STAGED}")
        self.rows, self.size = [], 0

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
