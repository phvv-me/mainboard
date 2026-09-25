import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from patos import IllegalTransition

from mainboard.dispatch.vocabulary import (
    CANCELLED,
    FAILED,
    OK,
    QUEUED,
    RUNNING,
    SKIPPED,
    TERMINAL,
    TIMEOUT,
    UNKNOWN,
    VANISHED,
    VERDICTS,
    tracker,
)

_WORDS = st.sampled_from(sorted(VERDICTS))


def test_the_table_declares_exactly_the_lifecycle_dispatch_promises() -> None:
    assert VERDICTS[QUEUED] == {RUNNING, VANISHED, CANCELLED}
    assert VERDICTS[RUNNING] == {OK, FAILED, VANISHED, TIMEOUT, CANCELLED}
    assert {OK, FAILED, VANISHED, UNKNOWN, TIMEOUT, CANCELLED, SKIPPED} == TERMINAL
    assert QUEUED not in TERMINAL and RUNNING not in TERMINAL
    assert tracker().current == QUEUED


@given(start=_WORDS, target=_WORDS)
@example(start=QUEUED, target=RUNNING)
@example(start=QUEUED, target=VANISHED)
@example(start=RUNNING, target=TIMEOUT)
@example(start=QUEUED, target=CANCELLED)
@example(start=CANCELLED, target=RUNNING)
@example(start=QUEUED, target=OK)
@example(start=OK, target=RUNNING)
@example(start=FAILED, target=RUNNING)
def test_only_a_declared_move_is_allowed_and_every_other_one_raises(
    start: str, target: str
) -> None:
    """A settled terminal sliding back to `running` is the regression the table exists to stop."""
    machine = tracker(start)
    if target in VERDICTS[start]:
        assert machine.to(target) == target
        assert machine.is_terminal() == (not VERDICTS[target])
        return
    with pytest.raises(IllegalTransition):
        machine.to(target)
    assert machine.current == start
