"""Read live event streams and their exact-byte Parquet snapshots."""

from compression import zstd
from pathlib import Path

import duckdb

from ..state.database import STARTUP
from .frames import Frame, parse_tail

_PART_BYTES = 512 * 1024
_CHUNK_BYTES = 64 * 1024


class FrameFile:
    """Preserve wire offsets and incomplete tails when compacting a finished stream."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read_bytes(self) -> bytes:
        """Return the original wire file, including a possible incomplete final frame."""
        if self.path.is_dir():
            parts = (self.path / "part-*.parquet").as_posix()
            with duckdb.connect(config={**STARTUP}) as connection:
                rows = connection.execute(
                    "SELECT ordinal, wire FROM read_parquet(?) ORDER BY ordinal", [parts]
                ).fetchall()
            if [ordinal for ordinal, _ in rows] != list(range(len(rows))):
                raise ValueError(f"incomplete event archive: {self.path}")
            return b"".join(wire for _, wire in rows)
        return self.path.read_bytes()

    def frames(self) -> list[Frame]:
        """Decode complete events while preserving the live reader's tail semantics."""
        raw = self.read_bytes()
        if self.path.name.endswith((".zst", ".zst.parquet")):
            raw = zstd.decompress(raw)
        complete, _, _ = raw.rpartition(b"\n")
        return parse_tail(complete.decode() + "\n")

    def archive(self) -> Path:
        """Create bounded Zstandard Parquet parts and verify their exact byte round trip."""
        raw = self.read_bytes()
        size = max(len(raw), 1)
        target = self.path.with_name(self.path.name + ".parquet")
        target.mkdir()
        with duckdb.connect(config={**STARTUP}) as connection:
            for part, start in enumerate(range(0, size, _PART_BYTES)):
                chunks = [
                    raw[offset : offset + _CHUNK_BYTES]
                    for offset in range(start, min(start + _PART_BYTES, size), _CHUNK_BYTES)
                ]
                first = start // _CHUNK_BYTES
                written = (target / f"part-{part:05d}.parquet").as_posix().replace("'", "''")
                connection.execute(
                    "COPY (SELECT unnest(?::UINTEGER[]) AS ordinal, unnest(?::BLOB[]) AS wire) "
                    f"TO '{written}' (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 19)",
                    [list(range(first, first + len(chunks))), chunks],
                )
        if FrameFile(target).read_bytes() != raw:
            raise ValueError(f"event archive changed bytes: {self.path}")
        return target
