import io
import sys
from subprocess import DEVNULL
from typing import TYPE_CHECKING, cast

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.engines.compile.backend import Process

if TYPE_CHECKING:
    from plumbum.commands.base import BaseCommand
    from pytest_subprocess import FakeProcess

_PYTHON = sys.executable
_ECHO = (_PYTHON, "-c", "print('hi')")
_TRUE = (_PYTHON, "-c", "")
_FALSE = (_PYTHON, "-c", "raise SystemExit(1)")


def test_output_returns_stdout_and_replays_everything_before_it_raises(
    fp: FakeProcess, capsys: pytest.CaptureFixture[str]
) -> None:
    """A query's text is the answer, and a failed one is a user-facing report first."""
    fp.register(_ECHO, stdout="hi\n")
    assert Process.output(local[_PYTHON][_ECHO[1:]], "echo") == "hi\n"

    fp.register(_FALSE, returncode=1, stdout="context\n", stderr="boom\n")
    with pytest.raises(MissionError, match="`a query` failed"):
        Process.output(local[_PYTHON][_FALSE[1:]], "a query")
    assert capsys.readouterr() == ("context\n", "boom\n")


def test_each_run_shape_reports_exactly_what_its_caller_asked_for(fp: FakeProcess) -> None:
    """A step wants a verdict, a passthrough wants the code, and a tty program wants neither."""
    fp.register(_TRUE, returncode=0)
    fp.register(_FALSE, returncode=1)
    assert Process.foreground(local[_PYTHON][_TRUE[1:]]) is True
    assert Process.foreground(local[_PYTHON][_FALSE[1:]]) is False

    exit_seven = (_PYTHON, "-c", "raise SystemExit(7)")
    fp.register(exit_seven, returncode=7)
    assert Process.passthrough(local[_PYTHON][exit_seven[1:]]) == 7

    fp.register(_TRUE, returncode=3)
    assert Process.handover(local[_PYTHON][_TRUE[1:]]) == 3


def test_stream_tees_both_output_streams_while_retaining_them(
    fp: FakeProcess, capsys: pytest.CaptureFixture[str]
) -> None:
    fp.register(_ECHO, stdout="out line\n", stderr="err line\n")
    result = Process.stream(local[_PYTHON][_ECHO[1:]])
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


def test_relay_keeps_unicode_evidence_when_the_console_cannot_encode_it() -> None:
    """A legacy Windows code page changes display only, never the retained child output."""
    raw_destination = io.BytesIO()
    destination = io.TextIOWrapper(raw_destination, encoding="cp1252", write_through=True)
    stream = io.BufferedReader(io.BytesIO(b"zebra\xc3"))

    assert Process.relay(stream, destination, "utf-8") == "zebra�"
    assert raw_destination.getvalue() == b"zebra?"


def test_detached_processes_release_terminal_handles_with_the_platform_lifetime_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The updater outlives its caller without inheriting that caller's terminal."""

    class Command:
        calls: list[dict[str, int]]

        def __init__(self) -> None:
            self.calls = []

        def popen(self, **kwargs: int) -> None:
            self.calls.append(kwargs)

    command = Command()
    monkeypatch.setattr("platform.system", lambda: "Windows")

    Process.detached(cast("BaseCommand", command))

    assert command.calls == [
        {
            "stdin": DEVNULL,
            "stdout": DEVNULL,
            "stderr": DEVNULL,
            "creationflags": 0x00000200 | 0x00000008,
        }
    ]

    command.calls.clear()
    monkeypatch.setattr("platform.system", lambda: "Linux")
    Process.detached(cast("BaseCommand", command))

    assert command.calls == [
        {
            "stdin": DEVNULL,
            "stdout": DEVNULL,
            "stderr": DEVNULL,
            "start_new_session": True,
        }
    ]
