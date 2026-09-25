"""Read live event streams and their exact-byte Parquet snapshots."""

from compression import zstd
from pathlib import Path

import polars as pl

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
            rows = pl.read_parquet(self.path / "part-*.parquet").sort("ordinal")
            if rows["ordinal"].to_list() != list(range(rows.height)):
                raise ValueError(f"incomplete event archive: {self.path}")
            return b"".join(rows["wire"].to_list())
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
        for part, start in enumerate(range(0, size, _PART_BYTES)):
            chunks = [
                raw[offset : offset + _CHUNK_BYTES]
                for offset in range(start, min(start + _PART_BYTES, size), _CHUNK_BYTES)
            ]
            first = start // _CHUNK_BYTES
            pl.DataFrame(
                {"ordinal": range(first, first + len(chunks)), "wire": chunks},
                schema={"ordinal": pl.UInt32, "wire": pl.Binary},
            ).write_parquet(
                target / f"part-{part:05d}.parquet", compression="zstd", compression_level=19
            )
        if FrameFile(target).read_bytes() != raw:
            raise ValueError(f"event archive changed bytes: {self.path}")
        return target
