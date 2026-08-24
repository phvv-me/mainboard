import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from patos import IllegalTransition

from mainboard.dispatch import VERDICTS
from mainboard.dispatch.vocabulary import (
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

_WORDS = st.sampled_from(sorted(VERDICTS))


def test_the_table_declares_exactly_the_lifecycle_dispatch_promises() -> None:
    assert VERDICTS[QUEUED] == {RUNNING, VANISHED}
    assert VERDICTS[RUNNING] == {OK, FAILED, VANISHED, TIMEOUT}
    assert {OK, FAILED, VANISHED, UNKNOWN, TIMEOUT} == TERMINAL
    assert QUEUED not in TERMINAL and RUNNING not in TERMINAL
    assert tracker().current == QUEUED


@given(start=_WORDS, target=_WORDS)
@example(start=QUEUED, target=RUNNING)
@example(start=QUEUED, target=VANISHED)
@example(start=RUNNING, target=TIMEOUT)
@example(start=QUEUED, target=OK)
@example(start=OK, target=RUNNING)
@example(start=FAILED, target=RUNNING)
def test_only_a_declared_move_is_allowed_and_every_other_one_raises(
    start: str, target: str
) -> None:
    """A settled terminal sliding back to `running` is the regression the table exists to stop.

    start: the verdict the tracker was built at.
    target: the verdict a scheduler then reported.
    """
    machine = tracker(start)
    if target in VERDICTS[start]:
        assert machine.to(target) == target
        assert machine.is_terminal() == (not VERDICTS[target])
        return
    with pytest.raises(IllegalTransition):
        machine.to(target)
    assert machine.current == start
