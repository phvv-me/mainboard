from __future__ import annotations

import mainboard
from mainboard import Memory, Meter, meter
from mainboard.enums import Vendor
from mainboard.gpu import GPU


class GrowingHost:
    """Host whose memory usage climbs on each access to drive peak tracking."""

    def __init__(self, used_gb: list[float]) -> None:
        self._used = iter(used_gb)

    @property
    def memory(self) -> Memory:
        gib = next(self._used)
        return Memory(scope="system", total_bytes=64 * 1024**3, used_bytes=int(gib * 1024**3))


class FakeMachine:
    """Stand-in machine exposing only the host and gpus the meter samples."""

    def __init__(self, host: GrowingHost, gpus: tuple[GPU, ...]) -> None:
        self.host = host
        self.gpus = gpus


def test_meter_tracks_peaks_and_delta() -> None:
    """The meter samples at enter, on `sample()`, and at exit, peaking the maximum."""

    class FixedGpu(GPU):
        vendor: Vendor = Vendor.NVIDIA

        @property
        def memory(self) -> Memory:
            return Memory(scope="vram", total_bytes=24 * 1024**3, used_bytes=8 * 1024**3)

    machine = FakeMachine(GrowingHost([10.0, 30.0, 20.0]), (FixedGpu(index=0),))
    m = Meter(machine)
    with m:
        m.sample()
    assert m.peak_host_gb == 30.0
    assert m.host_delta_gb == 10.0
    assert m.peak_gpu_gb == 8.0
    assert m.elapsed_s >= 0.0


def test_meter_empty_before_use_is_zeroed() -> None:
    """A meter that never sampled reports zero peaks and delta."""
    machine = FakeMachine(GrowingHost([]), ())
    m = Meter(machine)
    assert m.peak_host_gb == 0.0
    assert m.peak_gpu_gb == 0.0
    assert m.host_delta_gb == 0.0


def test_meter_binds_to_the_singleton_machine() -> None:
    """`meter()` opens a context bound to the real `Machine` singleton."""
    with meter() as m:
        pass
    assert isinstance(m, Meter)
    assert isinstance(m.machine, mainboard.Machine)
    assert m.elapsed_s >= 0.0
