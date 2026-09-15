"""Read live event streams and their exact-byte Parquet snapshots."""

from compression import zstd
from pathlib import Path

import polars as pl

from .frames import Frame, parse_tail


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
        target = self.path.with_name(self.path.name + ".parquet")
        target.mkdir()
        for part, start in enumerate(range(0, max(len(raw), 1), 524288)):
            chunks = [
                raw[offset : offset + 65536]
                for offset in range(start, min(start + 524288, max(len(raw), 1)), 65536)
            ]
            pl.DataFrame(
                {"ordinal": range(start // 65536, start // 65536 + len(chunks)), "wire": chunks},
                schema={"ordinal": pl.UInt32, "wire": pl.Binary},
            ).write_parquet(
                target / f"part-{part:05d}.parquet", compression="zstd", compression_level=19
            )
        if FrameFile(target).read_bytes() != raw:
            raise ValueError(f"event archive changed bytes: {self.path}")
        return target
