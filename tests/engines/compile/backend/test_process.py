import io
from typing import TYPE_CHECKING

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile.backend import Process

if TYPE_CHECKING:
    from pytest_subprocess import FakeProcess

# plumbum resolves a bare command through `which()` before spawning, so a fake registration
# must match the resolved absolute path it actually passes to `Popen`, not the bare name.
_ECHO = str(local.which("echo"))
_TRUE = str(local.which("true"))
_FALSE = str(local.which("false"))
_SH = str(local.which("sh"))


def test_output_returns_stdout_and_replays_everything_before_it_raises(
    fp: FakeProcess, capsys: pytest.CaptureFixture[str]
) -> None:
    """A query's text is the answer, and a failed one is a user-facing report first."""
    fp.register([_ECHO, "hi"], stdout="hi\n")
    assert Process.output(local["echo"]["hi"], "echo") == "hi\n"

    fp.register([_FALSE], returncode=1, stdout="context\n", stderr="boom\n")
    with pytest.raises(MissionError, match="`a query` failed"):
        Process.output(local["false"], "a query")
    assert capsys.readouterr() == ("context\n", "boom\n")


def test_each_run_shape_reports_exactly_what_its_caller_asked_for(fp: FakeProcess) -> None:
    """A step wants a verdict, a passthrough wants the code, and a tty program wants neither."""
    fp.register([_TRUE], returncode=0)
    fp.register([_FALSE], returncode=1)
    assert Process.foreground(local["true"]) is True
    assert Process.foreground(local["false"]) is False

    fp.register([_SH, "-c", "exit 7"], returncode=7)
    assert Process.passthrough(local["sh"]["-c", "exit 7"]) == 7

    fp.register([_TRUE], returncode=3)
    assert Process.handover(local["true"]) == 3


def test_stream_tees_both_output_streams_while_retaining_them(
    fp: FakeProcess, capsys: pytest.CaptureFixture[str]
) -> None:
    fp.register([_ECHO], stdout="out line\n", stderr="err line\n")
    result = Process.stream(local["echo"])
    assert (result.stdout, result.stderr) == ("out line\n", "err line\n")
    assert capsys.readouterr() == ("out line\n", "err line\n")


@pytest.mark.parametrize(
    ("raw", "buffer_size", "text"),
    [
        pytest.param(
            "zeb\xe9ra".encode(),
            1,
            "zeb\xe9ra",
            id="a-two-byte-character-split-across-two-incremental-reads",
        ),
        pytest.param(
            b"zebra\xc3",
            io.DEFAULT_BUFFER_SIZE,
            "zebra�",
            id="a-lead-byte-whose-continuation-never-arrived",
        ),
    ],
)
def test_relay_decodes_a_pipe_one_read_at_a_time(raw: bytes, buffer_size: int, text: str) -> None:
    """Streaming forwards each pipe read whole and splits no character.

    `read1` returns after one pipe read, so a short protocol message reaches its client
    immediately, and the incremental decoder carries a partial character across the boundary.
    """
    destination = io.StringIO()
    stream = io.BufferedReader(io.BytesIO(raw), buffer_size=buffer_size)
    assert Process.relay(stream, destination, "utf-8") == text
    assert destination.getvalue() == text
