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


def test_output_returns_stdout_on_success(fp: FakeProcess) -> None:
    fp.register([_ECHO, "hi"], stdout="hi\n")
    assert Process.output(local["echo"]["hi"], "echo") == "hi\n"


def test_output_replays_and_raises_on_failure(
    fp: FakeProcess, capsys: pytest.CaptureFixture[str]
) -> None:
    fp.register([_FALSE], returncode=1, stdout="context\n", stderr="boom\n")
    with pytest.raises(MissionError, match="`a query` failed"):
        Process.output(local["false"], "a query")
    captured = capsys.readouterr()
    assert captured.out == "context\n"
    assert captured.err == "boom\n"


def test_foreground_reports_success_as_a_bool(fp: FakeProcess) -> None:
    fp.register([_TRUE], returncode=0)
    fp.register([_FALSE], returncode=1)
    assert Process.foreground(local["true"]) is True
    assert Process.foreground(local["false"]) is False


def test_passthrough_preserves_the_exact_exit_code(fp: FakeProcess) -> None:
    fp.register([_SH, "-c", "exit 7"], returncode=7)
    assert Process.passthrough(local["sh"]["-c", "exit 7"]) == 7


def test_stream_tees_both_output_streams(
    fp: FakeProcess, capsys: pytest.CaptureFixture[str]
) -> None:
    fp.register([_ECHO], stdout="out line\n", stderr="err line\n")
    result = Process.stream(local["echo"])
    assert result.stdout == "out line\n"
    assert result.stderr == "err line\n"
    captured = capsys.readouterr()
    assert captured.out == "out line\n"
    assert captured.err == "err line\n"


def test_handover_gives_the_child_the_real_terminal(fp: FakeProcess) -> None:
    fp.register([_TRUE], returncode=3)
    assert Process.handover(local["true"]) == 3


def test_relay_copies_incremental_pipe_reads_across_a_split_multibyte_char() -> None:
    destination = io.StringIO()
    # A 2-byte UTF-8 character split across two `read1` calls exercises the incremental
    # decoder's carry-over path, not just whole-character chunks.
    stream = io.BufferedReader(io.BytesIO("zeb\xe9ra".encode()), buffer_size=1)
    text = Process.relay(stream, destination, "utf-8")
    assert text == "zeb\xe9ra"
    assert destination.getvalue() == "zeb\xe9ra"


def test_relay_flushes_a_dangling_incomplete_char_at_end_of_stream() -> None:
    destination = io.StringIO()
    # `\xc3` alone is the lead byte of a 2-byte UTF-8 character with its continuation byte
    # missing: the incremental decoder buffers it as incomplete, and only
    # `decode(b"", final=True)` forces it out, as U+FFFD since the reader uses
    # `errors="replace"`.
    stream = io.BufferedReader(io.BytesIO(b"zebra\xc3"))
    text = Process.relay(stream, destination, "utf-8")
    assert text == "zebra�"
    assert destination.getvalue() == "zebra�"
