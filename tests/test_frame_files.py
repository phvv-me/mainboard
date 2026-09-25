"""Compaction preserves complete events, truncated wire tails, and compressed sources."""

from compression import zstd
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from mainboard.observe.files import FrameFile
from mainboard.observe.frames import Frame, Kind, encode


@pytest.mark.parametrize("compressed", [False, True])
def test_event_archive_preserves_wire_and_partial_utf8(tmp_path: Path, compressed: bool) -> None:
    frame = Frame(
        job="stream",
        kind=Kind.line,
        at=datetime(2026, 9, 15, tzinfo=UTC),
        payload={"text": "exact 中文"},
    )
    raw = encode(frame).encode() + b'{"tail":"\xe4\xb8'
    source = tmp_path / ("events.ndjson.zst" if compressed else "events.ndjson")
    source.write_bytes(zstd.compress(raw) if compressed else raw)
    original = FrameFile(source)
    target = FrameFile(original.archive())
    assert target.read_bytes() == source.read_bytes()
    assert target.frames() == original.frames() == [frame]
    source.unlink()
    assert target.frames() == [frame]


def test_event_archive_bounds_parts_and_checks_missing_chunks(tmp_path: Path) -> None:
    source = tmp_path / "events.ndjson"
    source.write_bytes(b"x" * 1100000)
    target = FrameFile(FrameFile(source).archive())
    parts = list(target.path.glob("*.parquet"))
    assert len(parts) == 3
    assert all(part.stat().st_size < 1000000 for part in parts)
    assert target.read_bytes() == source.read_bytes()
    first = min(parts)
    frame = pl.read_parquet(first)
    frame.filter(pl.col("ordinal") != 0).write_parquet(first, compression="zstd")
    with pytest.raises(ValueError, match="incomplete"):
        target.read_bytes()


def test_an_archive_that_does_not_read_back_its_exact_bytes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compaction that lost one byte would replace the live stream with a silent corruption."""
    source = tmp_path / "events.ndjson"
    source.write_bytes(b"x" * 100)
    read = FrameFile.read_bytes

    def lossy(archived: FrameFile) -> bytes:
        whole = read(archived)
        return whole[:-1] if archived.path.is_dir() else whole

    monkeypatch.setattr(FrameFile, "read_bytes", lossy)
    with pytest.raises(ValueError, match="changed bytes"):
        FrameFile(source).archive()
