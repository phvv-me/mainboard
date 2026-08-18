import pytest
from patos import IllegalTransition

from mainboard.dispatch import VERDICTS
from mainboard.dispatch.verdicts import (
    FAILED,
    OK,
    QUEUED,
    RUNNING,
    TERMINAL,
    TIMEOUT,
    UNKNOWN,
    VANISHED,
    tracker,
)


def test_verdicts_table_matches_the_declared_shape() -> None:
    assert VERDICTS[QUEUED] == {RUNNING, VANISHED}
    assert VERDICTS[RUNNING] == {OK, FAILED, VANISHED, TIMEOUT}
    for terminal in (OK, FAILED, VANISHED, UNKNOWN, TIMEOUT):
        assert VERDICTS[terminal] == set()


def test_terminal_is_every_verdict_no_move_can_leave() -> None:
    assert {OK, FAILED, VANISHED, UNKNOWN, TIMEOUT} == TERMINAL
    assert QUEUED not in TERMINAL and RUNNING not in TERMINAL


def test_tracker_starts_at_queued_by_default() -> None:
    assert tracker().current == QUEUED


def test_tracker_follows_the_ordinary_lifecycle() -> None:
    machine = tracker()
    assert machine.to(RUNNING) == RUNNING
    assert machine.to(OK) == OK
    assert machine.is_terminal()


def test_a_queued_job_may_vanish_directly() -> None:
    machine = tracker()
    assert machine.to(VANISHED) == VANISHED


@pytest.mark.parametrize("terminal", [OK, FAILED, VANISHED, TIMEOUT])
def test_running_may_settle_into_any_terminal(terminal: str) -> None:
    machine = tracker(RUNNING)
    assert machine.to(terminal) == terminal


def test_a_regression_from_finished_back_to_running_is_illegal() -> None:
    machine = tracker(OK)
    with pytest.raises(IllegalTransition):
        machine.to(RUNNING)


def test_a_regression_from_failed_back_to_running_is_illegal() -> None:
    machine = tracker(FAILED)
    with pytest.raises(IllegalTransition):
        machine.to(RUNNING)


def test_queued_cannot_jump_straight_to_a_terminal_other_than_vanished() -> None:
    machine = tracker(QUEUED)
    with pytest.raises(IllegalTransition):
        machine.to(OK)
