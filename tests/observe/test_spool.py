from compression import zstd
from typing import TYPE_CHECKING

import pytest  # ruff:ignore[typing-only-third-party-import]  reason=hypothesis inspect.signature()'s this test, so its TempPathFactory annotation must resolve at runtime since=2026-08-17
from hypothesis import given
from hypothesis import strategies as st

from mainboard.observe import Kind, Spool, encoded_length, follow

from ..strategies import TEXT
from .support import ended, line

if TYPE_CHECKING:
    from pathlib import Path


@given(texts=st.lists(TEXT, min_size=2, max_size=8))
def test_every_frame_written_is_recovered_in_offset_order_across_any_roll(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    """A roll boundary is invisible to a reader, and an offset returns only the suffix past it."""
    root = tmp_path_factory.mktemp("spool")
    with Spool(root, "job1", roll_bytes=32) as spool:
        written = [spool.append(line(text=text)) for text in texts]
        recovered = spool.frames_from(0)
        suffix = spool.frames_from(written[-1].offset)
    assert [frame.payload["text"] for frame in recovered] == texts
    assert [frame.offset for frame in recovered] == [frame.offset for frame in written]
    assert suffix == [written[-1]]


def test_a_reopened_spool_resumes_where_the_last_one_stopped(tmp_path: Path) -> None:
    """An empty directory, a live tail and an archive with no live segment each resume right."""
    with Spool(tmp_path, "job1") as fresh:
        assert (fresh.segment_start, fresh.offset) == (0, 0)
        written = fresh.append(line(text="a"))
    tail = written.offset + encoded_length(written)
    with Spool(tmp_path, "job1") as live:
        assert (live.segment_start, live.offset) == (0, tail)

    rolled = tmp_path / "rolled"
    with Spool(rolled, "job1", roll_bytes=1) as roller:
        archived = roller.append(line(text="a"))
    end = archived.offset + encoded_length(archived)
    with Spool(rolled, "job1") as after:
        assert (after.segment_start, after.offset) == (end, end)

    orphan = tmp_path / "orphan"
    (orphan / "job1").mkdir(parents=True)
    body = "irrelevant-but-valid-length"
    name = f"{0:020d}-{len(body):020d}.ndjson.zst"
    (orphan / "job1" / name).write_bytes(zstd.compress(body.encode()))
    with Spool(orphan, "job1") as recovered:
        assert (recovered.segment_start, recovered.offset) == (len(body), len(body))


def test_heartbeat_publishes_the_status_a_poll_channel_reads(tmp_path: Path) -> None:
    """Nothing published yet answers `None`, never a stale reading a poll would trust."""
    with Spool(tmp_path, "job1") as spool:
        assert spool.status() is None
        spool.append(line())
        spool.heartbeat("running")
        status = spool.status()
        assert status is not None
        assert status["state"] == "running"
        assert status["offset"] == spool.offset
        assert status["base"] == spool.segment_start
        assert "updated_at" in status


def test_follow_replays_a_spool_and_stops_once_the_job_has_ended(tmp_path: Path) -> None:
    """The `ended` frame ends the tail, and so does catching up to a spool already marked ended."""
    with Spool(tmp_path, "job1") as spool:
        spool.append(line())
        closing = spool.append(ended())
        spool.heartbeat("ended")
        replayed = list(follow(spool, 0))
        caught_up = list(follow(spool, closing.offset + encoded_length(closing)))
    assert [frame.kind for frame in replayed] == [Kind.line, Kind.ended]
    assert caught_up == []


def test_follow_paces_with_the_injected_sleeper_while_the_job_is_still_running(
    tmp_path: Path,
) -> None:
    """A job with nothing new to say is polled on the caller's cadence rather than spun on."""
    calls: list[float] = []
    with Spool(tmp_path, "job1") as spool:

        def sleeper(interval: float) -> None:
            calls.append(interval)
            spool.append(ended())
            spool.heartbeat("ended")

        frames = list(follow(spool, 0, interval=0.01, sleeper=sleeper))
    assert calls == [0.01]
    assert [frame.kind for frame in frames] == [Kind.ended]
