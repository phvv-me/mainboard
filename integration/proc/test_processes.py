"""`Processes` ends trees on request and bounds a command, the same on macOS and Linux."""

import signal
import subprocess

import pytest

from mainboard.enums import Exit
from mainboard.proc import Processes

from ..builders import survivors


def test_killing_pids_stops_each_tree_and_answers_the_pids_that_were_gone(running) -> None:
    finished = subprocess.Popen(["true"])
    finished.wait()
    shell, members = running("sleep 60 & wait", workers=1)

    answered = Processes().kill([shell.pid, finished.pid])

    assert (answered, shell.wait(), survivors(members)) == (
        [finished.pid],
        -signal.SIGTERM,
        [],
    )


@pytest.mark.parametrize(
    ("script", "status"),
    [
        pytest.param("exit 7", 7, id="a-command-that-exits-on-its-own"),
        pytest.param("kill -9 $$", -signal.SIGKILL, id="a-command-killed-by-a-signal"),
        pytest.param("sleep 60 & wait", Exit.TIMED_OUT, id="a-command-past-its-bound"),
    ],
)
def test_a_bounded_command_answers_its_own_status_or_124_when_it_had_to_be_stopped(
    script: str, status: int
) -> None:
    assert Processes().timeout(1.0, ["sh", "-c", script]) == status
