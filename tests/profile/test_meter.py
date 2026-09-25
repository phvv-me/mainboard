from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import repeat

from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard import Meter


class Readings:
    """A `MemorySource`-shaped stand-in yielding its next reading on each access."""

    def __init__(self, used_gb: Iterable[float]) -> None:
        self._used = iter(used_gb)

    @property
    def memory(self) -> Readings:
        return self

    @property
    def used_gb(self) -> float:
        return next(self._used)


@dataclass
class FakeMachine:
    host: Readings
    gpus: Sequence[Readings]


# A handful of readings is a small space, so a trimmed budget covers it.
@settings(max_examples=10)
@given(
    readings=st.lists(
        st.floats(min_value=0.0, max_value=1024.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    ),
    gpu_used=st.lists(
        st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
        max_size=3,
    ),
)
def test_the_meter_peaks_over_its_samples_and_a_fresh_one_reads_zero(
    readings: list[float], gpu_used: Sequence[float]
) -> None:
    """The meter reads at enter, at each `sample()`, and at exit; with no readings it reports
    zeroes rather than raising on an empty maximum."""
    machine = FakeMachine(Readings(readings), tuple(Readings(repeat(used)) for used in gpu_used))
    fresh = Meter(machine)
    assert (fresh.peak_host_gb, fresh.peak_gpu_gb, fresh.host_delta_gb) == (0.0, 0.0, 0.0)

    with fresh as meter:
        for _ in range(len(readings) - 2):
            meter.sample()
    assert meter.peak_host_gb == max(readings)
    assert meter.host_delta_gb == readings[-1] - readings[0]
    assert meter.peak_gpu_gb == sum(gpu_used)
    assert meter.elapsed_s >= 0.0
