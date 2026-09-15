"""Lossless, bounded Parquet storage for immutable artifact bytes."""

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from patos import FrozenModel


class ArtifactChunk(FrozenModel):
    """One ordered piece of an object, with its original logical paths."""

    sha256: str
    ordinal: int
    paths: list[str]
    payload: bytes


class ParquetArtifacts:
    """Deduplicate exact bytes while retaining every receipt's original path."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.directory = self.root / "artifacts.parquet"

    @classmethod
    def read(cls, path: Path, *, boundary: Path, digest: str) -> bytes:
        """Resolve an archived path only inside its declared logical root."""
        path = path.absolute()
        boundary = boundary.absolute()
        path.relative_to(boundary)
        for parent in path.parents:
            if not parent.is_relative_to(boundary):
                break
            archive = cls(parent)
            bucket = archive.directory / digest[:2]
            if bucket.is_dir():
                rows = (
                    pl.scan_parquet(bucket / "*.parquet")
                    .filter(pl.col("sha256") == digest)
                    .sort("ordinal")
                    .collect()
                )
                if rows.is_empty():
                    continue
                if path.relative_to(parent).as_posix() not in rows["paths"][0]:
                    continue
                if rows["ordinal"].to_list() != list(range(rows.height)):
                    raise ValueError(f"incomplete archived artifact: {path}")
                return b"".join(rows["payload"].to_list())
        raise FileNotFoundError(path)

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

    def _pack_bucket(self, prefix: str, digests: list[str], objects: dict[str, list[str]]) -> None:
        directory = self.directory / prefix
        directory.mkdir()
        chunks: list[ArtifactChunk] = []
        size = 0
        part = 0
        for digest in digests:
            paths = objects[digest]
            data = (self.root / paths[0]).read_bytes()
            for ordinal, offset in enumerate(range(0, max(len(data), 1), 65536)):
                chunk = ArtifactChunk(
                    sha256=digest,
                    ordinal=ordinal,
                    paths=paths if ordinal == 0 else [],
                    payload=data[offset : offset + 65536],
                )
                chunks.append(chunk)
                size += len(chunk.payload) + sum(len(path.encode()) for path in chunk.paths)
                if size >= 524288:
                    self._write_part(directory, part, chunks)
                    part += 1
                    chunks = []
                    size = 0
        if chunks:
            self._write_part(directory, part, chunks)

    @staticmethod
    def _write_part(directory: Path, part: int, chunks: list[ArtifactChunk]) -> None:
        target = directory / f"part-{part:05d}.parquet"
        frame = pl.DataFrame(
            [chunk.model_dump() for chunk in chunks],
            schema={
                "sha256": pl.String,
                "ordinal": pl.UInt32,
                "paths": pl.List(pl.String),
                "payload": pl.Binary,
            },
        )
        frame.write_parquet(target, compression="zstd", compression_level=19)
        if target.stat().st_size >= 1000000:
            raise ValueError(f"archive shard exceeds the Git file budget: {target}")
