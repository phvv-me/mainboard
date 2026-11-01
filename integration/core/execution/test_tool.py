"""`captured` runs a child to its text output and lets the stdlib's own failures through."""

import subprocess
import sys

import pytest

from mainboard.core import captured

_PRINTS = "import sys; sys.stdout.write('out'); sys.stderr.write('err'); sys.exit(3)"
_BAD_BYTES = "import sys; sys.stdout.buffer.write(b'a\\xffb')"
_ENVIRONMENT = "import os; print(os.environ['MB_PROBE'], 'HOME' in os.environ)"
_SLEEPS = "import time; time.sleep(30)"


def test_captured_answers_each_stream_and_the_exit_status_of_a_failing_child() -> None:
    done = captured([sys.executable, "-c", _PRINTS])

    assert (done.stdout, done.stderr, done.returncode) == ("out", "err", 3)


def test_captured_replaces_bytes_that_are_not_utf8_instead_of_raising() -> None:
    assert captured([sys.executable, "-c", _BAD_BYTES]).stdout == "a\N{REPLACEMENT CHARACTER}b"


def test_captured_gives_the_child_exactly_the_environment_it_was_handed() -> None:
    done = captured([sys.executable, "-c", _ENVIRONMENT], env={"MB_PROBE": "seen"})

    assert done.stdout == "seen False\n"


@pytest.mark.parametrize(
    ("argv", "timeout", "raised"),
    [
        pytest.param(
            [sys.executable, "-c", _SLEEPS],
            0.5,
            subprocess.TimeoutExpired,
            id="a-child-that-outlives-its-timeout",
        ),
        pytest.param(["no-such-program"], None, FileNotFoundError, id="an-absent-program"),
    ],
)
def test_captured_raises_what_subprocess_raises_for_a_child_that_cannot_answer(
    argv: list[str], timeout: float | None, raised: type[Exception]
) -> None:
    with pytest.raises(raised):
        captured(argv, timeout=timeout)
