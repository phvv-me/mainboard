"""A hold lasts the time `--for` says, in the spellings a person writes."""

import pytest

from mainboard import MissionError
from mainboard.holding.holds import Cards, Holding


@pytest.mark.parametrize(
    ("written", "seconds"),
    [
        pytest.param("3h", 3 * 3600, id="hours"),
        pytest.param("90m", 90 * 60, id="minutes"),
        pytest.param("1h30m", 90 * 60, id="hours-and-minutes"),
        pytest.param("02:30:00", 150 * 60, id="a-clock"),
    ],
)
def test_a_hold_lasts_as_long_as_its_duration_says(written: str, seconds: int) -> None:
    assert Holding(duration=written).seconds == seconds


@pytest.mark.parametrize(
    "written", [pytest.param("soon", id="a-word"), pytest.param("", id="nothing")]
)
def test_a_hold_duration_nobody_can_read_is_refused_naming_the_spellings(written: str) -> None:
    with pytest.raises(MissionError, match="a hold lasts `3h`, `90m`, `1h30m` or `HH:MM:SS`"):
        Holding(duration=written).asked(Cards())
