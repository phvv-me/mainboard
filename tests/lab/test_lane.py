import pytest

from mainboard import MissionError
from mainboard.lab import Lane
from mainboard.lab.lane import orders, validates


def test_lane_name_is_required() -> None:
    assert Lane(name="cold").name == "cold"


def test_lane_accepts_arbitrary_extras() -> None:
    lane = Lane(name="warm", warmup=True, prompt="hello")
    assert lane.warmup is True
    assert lane.prompt == "hello"


def test_orders_cycles_through_every_permutation() -> None:
    lanes = (Lane(name="cold"), Lane(name="warm"))
    assert [lane.name for lane in orders(lanes, 0)] == ["cold", "warm"]
    assert [lane.name for lane in orders(lanes, 1)] == ["warm", "cold"]
    assert [lane.name for lane in orders(lanes, 2)] == ["cold", "warm"]


def test_orders_with_a_single_lane_always_returns_it() -> None:
    lanes = (Lane(name="only"),)
    assert orders(lanes, 0) == lanes
    assert orders(lanes, 5) == lanes


def test_orders_with_no_lanes_returns_empty() -> None:
    assert orders((), 3) == ()


def test_validates_accepts_a_multiple_of_the_permutation_count() -> None:
    lanes = (Lane(name="cold"), Lane(name="warm"))
    validates(4, lanes)


def test_validates_rejects_a_non_multiple() -> None:
    lanes = (Lane(name="cold"), Lane(name="warm"))
    with pytest.raises(MissionError, match="not a multiple"):
        validates(3, lanes)


def test_validates_with_no_lanes_accepts_any_block_count() -> None:
    validates(5, ())
