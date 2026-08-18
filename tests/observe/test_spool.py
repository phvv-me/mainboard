from compression import zstd
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest  # ruff:ignore[typing-only-third-party-import]  reason=hypothesis inspect.signature()'s this test, so its TempPathFactory annotation must resolve at runtime since=2026-08-17
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard.observe import Frame, Kind, Spool, follow

if TYPE_CHECKING:
    from pathlib import Path


_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _line(offset: int = 0, text: str = "hi") -> Frame:
    return Frame(job="job1", kind=Kind.line, offset=offset, at=_AT, payload={"text": text})


def test_append_stamps_the_true_write_position(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    first = spool.append(_line(text="a"))
    second = spool.append(_line(text="b"))
    spool.close()
    assert first.offset == 0
    assert second.offset > first.offset


def test_frames_from_zero_returns_everything_written(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    spool.append(_line(text="a"))
    spool.append(_line(text="b"))
    spool.close()
    reader = Spool(tmp_path, "job1")
    frames = reader.frames_from(0)
    reader.close()
    assert [frame.payload["text"] for frame in frames] == ["a", "b"]


def test_frames_from_mid_offset_returns_only_the_suffix(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    first = spool.append(_line(text="a"))
    second = spool.append(_line(text="b"))
    frames = spool.frames_from(second.offset)
    spool.close()
    assert [frame.offset for frame in frames] == [second.offset]
    assert first.offset < second.offset


def test_roll_compresses_the_live_segment_and_frames_from_still_spans_it(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1", roll_bytes=1)
    spool.append(_line(text="a"))
    spool.append(_line(text="b"))
    archives = list((tmp_path / "job1").glob("*.ndjson.zst"))
    assert archives
    frames = spool.frames_from(0)
    spool.close()
    assert [frame.payload["text"] for frame in frames] == ["a", "b"]


def test_resume_after_a_roll_with_no_live_content_yet(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1", roll_bytes=1)
    spool.append(_line(text="a"))
    resumed = Spool(tmp_path, "job1")
    assert resumed.segment_start == spool.offset
    assert resumed.offset == spool.offset
    spool.close()
    resumed.close()


def test_resume_from_an_archive_only_directory_with_no_prior_spool(tmp_path: Path) -> None:

    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    body = "irrelevant-but-valid-length"
    (job_dir / f"{0:020d}-{len(body):020d}.ndjson.zst").write_bytes(zstd.compress(body.encode()))
    spool = Spool(tmp_path, "job1")
    assert spool.segment_start == len(body)
    assert spool.offset == len(body)
    spool.close()


def test_close_releases_the_file_handle(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    spool.close()
    assert spool.handle.closed


def test_context_manager_closes_on_exit(tmp_path: Path) -> None:
    with Spool(tmp_path, "job1") as spool:
        spool.append(_line())
    assert spool.handle.closed


def test_heartbeat_publishes_status_atomically(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    spool.append(_line())
    spool.heartbeat("running")
    status = spool.status()
    spool.close()
    assert status is not None
    assert status["state"] == "running"
    assert status["offset"] == spool.offset
    assert status["base"] == spool.segment_start
    assert "updated_at" in status


def test_status_is_none_before_any_heartbeat(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    status = spool.status()
    spool.close()
    assert status is None


def test_follow_stops_after_replaying_an_already_ended_job(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    spool.append(_line())
    spool.append(Frame(job="job1", kind=Kind.ended, at=_AT, payload={"exit_code": 0}))
    spool.heartbeat("ended")
    frames = list(follow(spool, 0))
    spool.close()
    assert [frame.kind for frame in frames] == [Kind.line, Kind.ended]


def test_follow_stops_once_it_has_caught_up_to_an_already_ended_job(tmp_path: Path) -> None:
    spool = Spool(tmp_path, "job1")
    ended = spool.append(Frame(job="job1", kind=Kind.ended, at=_AT, payload={"exit_code": 0}))
    spool.heartbeat("ended")
    caught_up = ended.offset + len(ended.model_dump_json()) + 1
    frames = list(follow(spool, caught_up))
    spool.close()
    assert frames == []


def test_follow_paces_with_the_injected_sleeper_while_the_job_is_still_running(
    tmp_path: Path,
) -> None:
    spool = Spool(tmp_path, "job1")
    calls: list[float] = []

    def sleeper(interval: float) -> None:
        calls.append(interval)
        spool.append(Frame(job="job1", kind=Kind.ended, at=_AT, payload={"exit_code": 0}))
        spool.heartbeat("ended")

    frames = list(follow(spool, 0, interval=0.01, sleeper=sleeper))
    spool.close()
    assert calls == [0.01]
    assert [frame.kind for frame in frames] == [Kind.ended]


@settings(deadline=None)
@given(texts=st.lists(st.text(max_size=12), min_size=1, max_size=20))
def test_frames_from_recovers_every_frame_across_random_rolls(
    tmp_path_factory: pytest.TempPathFactory, texts: list[str]
) -> None:
    root = tmp_path_factory.mktemp("spool")
    spool = Spool(root, "job1", roll_bytes=32)
    written = [spool.append(_line(text=text)) for text in texts]
    recovered = spool.frames_from(0)
    spool.close()
    assert [frame.payload["text"] for frame in recovered] == texts
    assert [frame.offset for frame in recovered] == [frame.offset for frame in written]
