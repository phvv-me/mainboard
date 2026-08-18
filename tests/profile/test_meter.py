from typing import TYPE_CHECKING

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

    @property
    def memory(self) -> FixedGpu:
        return self

    @property
    def used_gb(self) -> float:
        return 8.0


class FakeMachine:
    """Stand-in machine exposing only the host and gpus the meter samples."""

    def __init__(self, host: GrowingHost, gpus: Sequence[FixedGpu]) -> None:
        self.host = host
        self.gpus = gpus


def test_meter_tracks_peaks_and_delta() -> None:
    """The meter samples at enter, on `sample()`, and at exit, peaking the maximum."""
    machine = FakeMachine(GrowingHost([10.0, 30.0, 20.0]), (FixedGpu(),))
    meter = Meter(machine)
    with meter:
        meter.sample()
    assert meter.peak_host_gb == 30.0
    assert meter.host_delta_gb == 10.0
    assert meter.peak_gpu_gb == 8.0
    assert meter.elapsed_s >= 0.0


def test_meter_empty_before_use_is_zeroed() -> None:
    """A meter that never sampled reports zero peaks and delta."""
    machine = FakeMachine(GrowingHost([]), ())
    meter = Meter(machine)
    assert meter.peak_host_gb == 0.0
    assert meter.peak_gpu_gb == 0.0
    assert meter.host_delta_gb == 0.0
