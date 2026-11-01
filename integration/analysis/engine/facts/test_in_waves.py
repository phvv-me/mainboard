"""`in_waves` runs a dependency graph on threads: every item once, after the items it waits for.
A failure surfaces and stops everything that waits on it, and a graph of nothing runs nothing."""

from graphlib import CycleError

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.engine.facts import in_waves

from .fakes import Ledger

type Graph = dict[int, set[int]]


@st.composite
def _graphs(draw: st.DrawFn, *, minimum: int = 0) -> Graph:
    """A dependency graph whose every item waits for up to four items before it."""
    size = draw(st.integers(minimum, 40))
    return {
        item: set(draw(st.lists(st.integers(0, item - 1), max_size=4))) if item else set()
        for item in range(size)
    }


@st.composite
def _failing(draw: st.DrawFn) -> tuple[Graph, int]:
    """A non-empty dependency graph and the one item whose work fails."""
    waits = draw(_graphs(minimum=1))
    return waits, draw(st.sampled_from(sorted(waits)))


@settings(max_examples=60)
@example(waits={0: set(), 1: {0}, 2: {1}, 3: {0, 2}})
@example(waits={0: set(), 1: set(), 2: {0, 1}})
@given(waits=_graphs())
def test_every_item_runs_once_and_only_after_the_items_it_waits_for(waits: Graph) -> None:
    ledger = Ledger()

    in_waves(waits, ledger)

    assert sorted(ledger.started) == sorted(waits)
    assert all(waits[item] <= ledger.seen[item] for item in waits)


@settings(max_examples=30)
@example(drawn=({0: set(), 1: {0}, 2: {1}}, 0))
@example(drawn=({0: set(), 1: set(), 2: {0, 1}}, 1))
@given(drawn=_failing())
def test_a_failure_propagates_and_nothing_that_waits_on_it_starts(
    drawn: tuple[Graph, int],
) -> None:
    waits, failing = drawn
    ledger = Ledger(failing=failing)

    with pytest.raises(RuntimeError, match=f"item {failing} failed"):
        in_waves(waits, ledger)

    assert not any(failing in waits[item] for item in ledger.started)


def test_an_empty_graph_runs_no_work_at_all() -> None:
    ledger = Ledger()

    in_waves({}, ledger)

    assert ledger.started == []


def test_a_cycle_is_refused_before_any_work_starts() -> None:
    waits = {0: {1}, 1: {0}}
    ledger = Ledger()

    with pytest.raises(CycleError):
        in_waves(waits, ledger)

    assert ledger.started == []
