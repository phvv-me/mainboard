"""Lossless, bounded Parquet storage for immutable artifact bytes."""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from patos import FrozenModel

from ...state.relations import Relations, write_parquet

_CHUNKS_BY_DIGEST_SQL = (
    "SELECT ordinal, paths, payload FROM read_parquet(?) WHERE sha256 = ? ORDER BY ordinal"
)
_CREATE_CHUNKS_TABLE_SQL = (
    "CREATE TABLE chunks (sha256 VARCHAR, ordinal UINTEGER, paths VARCHAR[], payload BLOB)"
)
_ARCHIVE_SHARD_SIZE_BUDGET_BYTES = 1000000
_ARTIFACT_CHUNK_SIZE_BYTES = 65536
_PART_FLUSH_THRESHOLD_BYTES = 524288


class ArtifactChunk(FrozenModel):
    """One ordered piece of an object, with its original logical paths."""

    sha256: str
    ordinal: int
    paths: list[str]
    payload: bytes


class ParquetArtifacts:
    """Deduplicate exact bytes while retaining every receipt's original path."""

    @classmethod
    def read(cls, path: Path, *, boundary: Path, digest: str) -> bytes:
        """Resolve an archived path only inside its declared logical root."""
        path = path.absolute()
        boundary = boundary.absolute()
        path.relative_to(boundary)
        for parent in path.parents:
            if not parent.is_relative_to(boundary):
                break
            payload = cls(parent).lookup(path, digest)
            if payload is not None:
                return payload
        raise FileNotFoundError(path)

    @staticmethod
    def _write_part(directory: Path, part: int, chunks: Iterable[ArtifactChunk]) -> None:
        target = directory / f"part-{part:05d}.parquet"
        relations = Relations()
        relations.connection.execute(_CREATE_CHUNKS_TABLE_SQL)
        relations.connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?)",
            [(chunk.sha256, chunk.ordinal, chunk.paths, chunk.payload) for chunk in chunks],
        )
        write_parquet(relations.connection.table("chunks"), target, level=19)
        if target.stat().st_size >= _ARCHIVE_SHARD_SIZE_BUDGET_BYTES:
            raise ValueError(f"archive shard exceeds the artifact size budget: {target}")

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.directory = self.root / "artifacts.parquet"

    def _pack_bucket(
        self, prefix: str, digests: Iterable[str], objects: Mapping[str, list[str]]
    ) -> None:
        directory = self.directory / prefix
        directory.mkdir()
        chunks: list[ArtifactChunk] = []
        size = 0
        part = 0
        for digest in digests:
            paths = objects[digest]
            data = (self.root / paths[0]).read_bytes()
            for ordinal, offset in enumerate(
                range(0, max(len(data), 1), _ARTIFACT_CHUNK_SIZE_BYTES)
            ):
                chunk = ArtifactChunk(
                    sha256=digest,
                    ordinal=ordinal,
                    paths=paths if ordinal == 0 else [],
                    payload=data[offset : offset + _ARTIFACT_CHUNK_SIZE_BYTES],
                )
                chunks.append(chunk)
                size += len(chunk.payload) + sum(len(path.encode()) for path in chunk.paths)
                if size >= _PART_FLUSH_THRESHOLD_BYTES:
                    self._write_part(directory, part, chunks)
                    part += 1
                    chunks = []
                    size = 0
        if chunks:
            self._write_part(directory, part, chunks)

    def lookup(self, path: Path, digest: str) -> bytes | None:
        """The bytes of `digest` when this archive lists `path` under it, else None."""
        bucket = self.directory / digest[:2]
        if not bucket.is_dir():
            return None
        rows = (
            Relations()
            .sql(
                _CHUNKS_BY_DIGEST_SQL,
                [str(bucket / "*.parquet"), digest],
            )
            .fetchall()
        )
        if not rows or path.relative_to(self.root).as_posix() not in rows[0][1]:
            return None
        if [ordinal for ordinal, _, _ in rows] != list(range(len(rows))):
            raise ValueError(f"incomplete archived artifact: {path}")
        return b"".join(payload for _, _, payload in rows)

    def pack(self, paths: Iterable[Path]) -> None:
        """Write a new archive without removing or changing any input file."""
        objects: dict[str, list[str]] = defaultdict(list)
        for path in sorted(paths):
            relative = path.absolute().relative_to(self.root).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.parent.name == "objects" and digest != path.name:
                raise ValueError(f"content-addressed artifact changed: {path}")
            objects[digest].append(relative)
        self.directory.mkdir()
        buckets: dict[str, list[str]] = defaultdict(list)
        for digest in sorted(objects):
            buckets[digest[:2]].append(digest)
        for prefix, digests in buckets.items():
            self._pack_bucket(prefix, digests, objects)
