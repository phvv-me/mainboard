import math

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.lab import Lane
from mainboard.lab.lane import orders, validates

from ..strategies import TEXT, WORDS


@given(name=WORDS, warmup=st.booleans(), prompt=TEXT)
def test_a_lane_carries_its_name_and_whatever_extras_it_was_declared_with(
    *, name: str, warmup: bool, prompt: str
) -> None:
    lane = Lane(name=name, warmup=warmup, prompt=prompt)
    assert lane.name == name
    assert lane.warmup is warmup
    assert lane.prompt == prompt


@given(names=st.lists(WORDS, unique=True, max_size=3), block=st.integers(0, 30))
@example(names=[], block=3)
@example(names=["only"], block=5)
@example(names=["cold", "warm"], block=1)
def test_orders_runs_every_lane_permutation_once_before_repeating_one(
    *, names: list[str], block: int
) -> None:
    lanes = tuple(Lane(name=name) for name in names)
    cycle = math.factorial(len(lanes))
    ordering = orders(lanes, block)
    assert sorted(lane.name for lane in ordering) == sorted(names)
    assert orders(lanes, block + cycle) == ordering
    assert (
        len({tuple(lane.name for lane in orders(lanes, step)) for step in range(cycle)}) == cycle
    )


@given(count=st.integers(0, 3), cycles=st.integers(0, 3), remainder=st.integers(0, 5))
@example(count=2, cycles=1, remainder=0)
@example(count=2, cycles=1, remainder=1)
def test_validates_accepts_only_a_block_count_that_completes_whole_cycles(
    *, count: int, cycles: int, remainder: int
) -> None:
    lanes = tuple(Lane(name=f"lane{index}") for index in range(count))
    cycle = math.factorial(count)
    validates(cycles * cycle, lanes)
    if remainder % cycle:
        with pytest.raises(MissionError, match="not a multiple"):
            validates(cycles * cycle + remainder, lanes)
