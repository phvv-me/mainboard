from datetime import UTC
from typing import TYPE_CHECKING

import pytest

from mainboard.observe import Kind, Spool
from mainboard.observe.agentmain import Args, main, parse_args, wrap

from .conftest import AT, ended, line

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime
    from pathlib import Path

    from pytest_subprocess import FakeProcess


def _ticking(start: datetime, step: float) -> Iterator[datetime]:
    """An unbounded clock starting at `start`, advancing by `step` seconds on every call."""
    moment = start
    while True:
        yield moment
        moment = moment.fromtimestamp(moment.timestamp() + step, tz=UTC)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], Args(root="", job="", follow=False, from_offset=0, child=())),
        (
            ["--root", "/spool", "--job", "job1", "--follow", "--from-offset", "42", "--", "hi"],
            Args(root="/spool", job="job1", follow=True, from_offset=42, child=("hi",)),
        ),
        (
            ["--mystery", "--root", "/spool"],
            Args(root="/spool", job="", follow=False, from_offset=0, child=()),
        ),
    ],
)
def test_parse_args_reads_every_flag_and_ignores_what_it_does_not_know(
    argv: list[str], expected: Args
) -> None:
    """Everything after a literal `--` is the wrapped child, and nothing before it has to be."""
    assert parse_args(argv) == expected


@pytest.mark.parametrize(
    ("argv", "stdout", "code", "step", "interval", "kinds"),
    [
        (
            ["echo", "hi"],
            ["line one", "line two"],
            0,
            0.0,
            1000.0,
            [Kind.started, Kind.line, Kind.line, Kind.ended],
        ),
        (
            ["stream"],
            ["a", "b"],
            0,
            10.0,
            5.0,
            [Kind.started, Kind.line, Kind.sample, Kind.line, Kind.sample, Kind.ended],
        ),
        (["false"], [], 7, 0.0, 1000.0, [Kind.started, Kind.ended]),
    ],
)
def test_wrap_spools_the_childs_output_its_rss_samples_and_its_exit_code(
    tmp_path: Path,
    fake_process: FakeProcess,
    argv: list[str],
    stdout: list[str],
    code: int,
    step: float,
    interval: float,
    kinds: list[Kind],
) -> None:
    """A sample lands only once the interval has really elapsed between two output lines."""
    fake_process.register(argv, stdout=stdout, returncode=code)
    clock = _ticking(AT, step)
    with Spool(tmp_path, "job1") as spool:
        returned = wrap(
            spool,
            argv,
            sample_interval=interval,
            now=lambda: next(clock),
            rss=lambda pid: 12345,
        )
        frames = spool.frames_from(0)
        status = spool.status()
    assert returned == code
    assert [frame.kind for frame in frames] == kinds
    assert [frame.payload.get("text") for frame in frames if frame.kind is Kind.line] == stdout
    assert all(frame.payload["rss"] == 12345 for frame in frames if frame.kind is Kind.sample)
    assert frames[-1].payload["exit_code"] == code
    assert status is not None
    assert status["state"] == "ended"


def test_main_wraps_a_child_and_returns_its_exit_code(
    tmp_path: Path, fake_process: FakeProcess
) -> None:
    """The wrap-and-run half of the entry point, spool opened and closed around the child."""
    fake_process.register(["echo", "hi"], stdout=["line one"], returncode=0)
    assert main(["--root", str(tmp_path), "--job", "job1", "--", "echo", "hi"]) == 0
    with Spool(tmp_path, "job1") as reader:
        assert [frame.kind for frame in reader.frames_from(0)] == [
            Kind.started,
            Kind.line,
            Kind.ended,
        ]


def test_main_follow_mode_replays_the_spool_as_ndjson(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The read half, which is what a `StreamChannel` reads on the other end of an ssh exec."""
    with Spool(tmp_path, "job1") as writer:
        writer.append(line(text="a"))
        writer.append(ended())
        writer.heartbeat("ended")

    assert main(["--root", str(tmp_path), "--job", "job1", "--follow", "--from-offset", "0"]) == 0
    printed = capsys.readouterr().out
    assert printed.count("\n") == 2
    assert '"kind":"ended"' in printed
