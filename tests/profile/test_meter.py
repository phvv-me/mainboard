from typing import TYPE_CHECKING

from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard import Meter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class GrowingHost:
    """Host whose memory usage climbs on each access to drive peak tracking."""

    def __init__(self, used_gb: Iterable[float]) -> None:
        self._used = iter(used_gb)

    @property
    def memory(self) -> GrowingHost:
        return self

    @property
    def used_gb(self) -> float:
        return next(self._used)


class FixedGpu:
    """A `MemorySource`-shaped stand-in reporting a fixed used amount."""

    def __init__(self, used_gb: float = 8.0) -> None:
        self._used_gb = used_gb

    @property
    def memory(self) -> FixedGpu:
        return self

    @property
    def used_gb(self) -> float:
        return self._used_gb


class FakeMachine:
    """Stand-in machine exposing only the host and gpus the meter samples."""

    def __init__(self, host: GrowingHost, gpus: Sequence[FixedGpu]) -> None:
        self.host = host
        self.gpus = gpus


# The axes here are a handful of readings, so a trimmed budget covers them and leaves the
# suite's wall time where it was.
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
    readings: list[float], gpu_used: list[float]
) -> None:
    """Peaks are the maximum over every sample and the delta spans first to last.

    The meter reads at enter, at each explicit `sample()`, and at exit, so a list of
    readings is consumed exactly once each. A meter that never sampled has no readings at
    all and reports zeroes rather than raising on an empty maximum.
    """
    machine = FakeMachine(GrowingHost(readings), tuple(FixedGpu(used) for used in gpu_used))
    fresh = Meter(machine)
    assert (fresh.peak_host_gb, fresh.peak_gpu_gb, fresh.host_delta_gb) == (0.0, 0.0, 0.0)

    with fresh as meter:
        for _ in range(len(readings) - 2):
            meter.sample()
    assert meter.peak_host_gb == max(readings)
    assert meter.host_delta_gb == readings[-1] - readings[0]
    assert meter.peak_gpu_gb == sum(gpu_used)
    assert meter.elapsed_s >= 0.0
