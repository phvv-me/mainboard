"""A scheduler walltime is the clock form alone, which is how a job record carries it."""

import pytest

from mainboard.runtime import walltime_seconds


@pytest.mark.parametrize(
    ("written", "seconds"),
    [
        pytest.param("07:59:59", 8 * 3600 - 1, id="a-short-queue-ceiling"),
        pytest.param("25:00:00", 25 * 3600, id="hours-past-a-day"),
        pytest.param("00:00:00", 0, id="zero"),
    ],
)
def test_a_clock_walltime_is_whole_seconds(written: str, seconds: int) -> None:
    assert walltime_seconds(written) == seconds


@pytest.mark.parametrize(
    "written",
    [
        pytest.param("2h", id="a-spoken-duration"),
        pytest.param("90", id="a-bare-number"),
        pytest.param("1:30", id="a-clock-without-seconds"),
        pytest.param("", id="nothing"),
    ],
)
def test_a_walltime_a_scheduler_would_refuse_is_refused_here_too(written: str) -> None:
    with pytest.raises(ValueError):
        walltime_seconds(written)
