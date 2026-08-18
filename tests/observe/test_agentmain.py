from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mainboard.observe import Frame, Kind, Spool
from mainboard.observe.agentmain import Args, main, parse_args, wrap

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pytest
    from pytest_subprocess import FakeProcess

_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _ticking(start: datetime, step: float) -> Iterator[datetime]:
    """An unbounded clock starting at `start`, advancing by `step` seconds on every call."""
    moment = start
    while True:
        yield moment
        moment = moment.fromtimestamp(moment.timestamp() + step, tz=UTC)


def test_parse_args_defaults_to_a_bare_wrap_with_no_child() -> None:
    assert parse_args([]) == Args(root="", job="", follow=False, from_offset=0, child=())


def test_parse_args_reads_every_flag_and_the_child_argv() -> None:
    args = parse_args(
        [
            "--root",
            "/spool",
            "--job",
            "job1",
            "--follow",
            "--from-offset",
            "42",
            "--",
            "echo",
            "hi",
        ]
    )
    assert args == Args(
        root="/spool", job="job1", follow=True, from_offset=42, child=("echo", "hi")
    )


def test_parse_args_ignores_an_unrecognized_flag() -> None:
    assert parse_args(["--mystery", "--root", "/spool"]) == Args(
        root="/spool", job="", follow=False, from_offset=0, child=()
    )


def test_wrap_spools_output_lines_started_and_ended(
    tmp_path: Path, fake_process: FakeProcess
) -> None:
    fake_process.register(["echo", "hi"], stdout=["line one", "line two"], returncode=0)
    clock = _ticking(_AT, 0.0)
    spool = Spool(tmp_path, "job1")
    code = wrap(
        spool,
        ["echo", "hi"],
        sample_interval=1000.0,
        now=lambda: next(clock),
        rss=lambda pid: 0,
    )
    frames = spool.frames_from(0)
    spool.close()
    assert code == 0
    assert [frame.kind for frame in frames] == [Kind.started, Kind.line, Kind.line, Kind.ended]
    assert [frame.payload.get("text") for frame in frames if frame.kind is Kind.line] == [
        "line one",
        "line two",
    ]
    assert frames[-1].payload["exit_code"] == 0


def test_wrap_records_a_nonzero_exit_code(tmp_path: Path, fake_process: FakeProcess) -> None:
    fake_process.register(["false"], stdout=[], returncode=7)
    spool = Spool(tmp_path, "job1")
    code = wrap(spool, ["false"], now=lambda: _AT, rss=lambda pid: 0)
    frames = spool.frames_from(0)
    spool.close()
    assert code == 7
    assert frames[-1].payload["exit_code"] == 7


def test_wrap_samples_rss_and_heartbeats_once_the_interval_elapses(
    tmp_path: Path, fake_process: FakeProcess
) -> None:
    fake_process.register(["stream"], stdout=["a", "b", "c"], returncode=0)
    clock = _ticking(_AT, 10.0)
    spool = Spool(tmp_path, "job1")
    wrap(spool, ["stream"], sample_interval=5.0, now=lambda: next(clock), rss=lambda pid: 12345)
    samples = [frame for frame in spool.frames_from(0) if frame.kind is Kind.sample]
    status = spool.status()
    spool.close()
    assert samples
    assert all(sample.payload["rss"] == 12345 for sample in samples)
    assert status is not None and status["state"] == "ended"


def test_main_wrap_mode_runs_the_child_and_returns_its_exit_code(
    tmp_path: Path, fake_process: FakeProcess
) -> None:
    fake_process.register(["echo", "hi"], stdout=["line one"], returncode=0)
    code = main(["--root", str(tmp_path), "--job", "job1", "--", "echo", "hi"])
    assert code == 0
    reader = Spool(tmp_path, "job1")
    frames = reader.frames_from(0)
    reader.close()
    assert [frame.kind for frame in frames] == [Kind.started, Kind.line, Kind.ended]


def test_main_follow_mode_replays_and_prints_ndjson(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    writer = Spool(tmp_path, "job1")
    writer.append(Frame(job="job1", kind=Kind.line, at=_AT, payload={"text": "a"}))
    writer.append(Frame(job="job1", kind=Kind.ended, at=_AT, payload={"exit_code": 0}))
    writer.heartbeat("ended")
    writer.close()

    code = main(["--root", str(tmp_path), "--job", "job1", "--follow", "--from-offset", "0"])
    printed = capsys.readouterr().out
    assert code == 0
    assert printed.count("\n") == 2
    assert '"kind":"ended"' in printed
