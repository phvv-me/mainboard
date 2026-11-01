"""`ProcessTree` ends a command with everything it started, however stubborn the processes are."""

import pytest

from mainboard.core import ProcessTree

from ...builders import survivors

_COOPERATIVE = "sleep 60 & sleep 60 & wait"
_STUBBORN = "trap '' TERM; sleep 60 & sleep 60 & wait"


@pytest.mark.parametrize(
    ("script", "killed"),
    [
        pytest.param(_COOPERATIVE, False, id="a-tree-that-stops-when-asked"),
        pytest.param(_STUBBORN, True, id="a-tree-that-ignores-the-request-and-is-killed-after-it"),
    ],
)
def test_stopping_a_tree_ends_every_process_and_says_whether_it_had_to_be_killed(
    running, script: str, killed: bool
) -> None:
    shell, members = running(script)

    stopped = ProcessTree(shell, grace=0.5).stop()

    assert (stopped, survivors(members)) == (killed, [])


def test_terminating_a_tree_spares_what_ignores_it_and_killing_it_spares_nothing(running) -> None:
    shell, members = running(_STUBBORN)
    tree = ProcessTree(shell)

    tree.terminate()
    asked = survivors(members, within=0.5)
    tree.kill()

    assert (len(asked), survivors(members)) == (len(members), [])


def test_ending_a_pid_answers_whether_it_was_there_to_end(running) -> None:
    shell, members = running("sleep 60 & wait", workers=1)

    there = ProcessTree.end(shell.pid, force=True)
    shell.wait()
    gone = ProcessTree.end(shell.pid)

    assert (there, gone, survivors(members)) == (True, False, [])
