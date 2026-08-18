import os
from typing import TYPE_CHECKING

import pytest

from mainboard.profile import Activity, TraceCollector, Tracer, annotate
from mainboard.profile import spans as span_module

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture(autouse=True)
def reset_profiling_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: no leftover active profiler or cached tracer between tests.

    `monkeypatch.setattr` (rather than a direct `module._private = ...` assignment) both
    resets the module-private state up front and restores it automatically at teardown,
    covering a prior test that failed before clearing its own active session.
    """
    monkeypatch.setattr(span_module, "_active", None, raising=False)
    monkeypatch.setattr(annotate, "_tracer", None, raising=False)


class FakeUtilization:
    """A `DeviceUtilization`-shaped stand-in with fixed compute/memory percentages."""

    def __init__(self, *, gpu_pct: int = 0, memory_pct: int = 0) -> None:
        self.gpu_pct = gpu_pct
        self.memory_pct = memory_pct


class FakeEnergy:
    """A `DeviceEnergy`-shaped stand-in with a fixed power draw."""

    def __init__(self, power_w: float = 0.0) -> None:
        self.power_w = power_w


class FakeThermal:
    """A `DeviceThermal`-shaped stand-in with fixed temperature and throttle state."""

    def __init__(
        self,
        temperature_c: int = 0,
        *,
        is_throttling: bool = False,
        throttle_names: tuple[str, ...] = (),
    ) -> None:
        self.temperature_c = temperature_c
        self.is_throttling = is_throttling
        self.throttle_names = list(throttle_names)


class FakeMemory:
    """A `DeviceMemory`-shaped stand-in with fixed capacity and pressure."""

    def __init__(self, *, total_gb: float = 0.0, percent_used: float = 0.0) -> None:
        self.total_gb = total_gb
        self.percent_used = percent_used


class FakeProcess:
    """A `DeviceProcess`-shaped stand-in: one process's device memory footprint."""

    def __init__(self, *, pid: int, used_bytes: int) -> None:
        self.pid = pid
        self.used_bytes = used_bytes


class FakeSnapshot:
    """A `DeviceSnapshot`-shaped stand-in: one point-in-time device reading."""

    def __init__(
        self,
        unit_name: str = "probe",
        processes: Sequence[FakeProcess] = (),
        utilization: FakeUtilization | None = None,
        energy: FakeEnergy | None = None,
        thermal: FakeThermal | None = None,
    ) -> None:
        self.unit_name = unit_name
        self.processes = processes
        self.utilization = utilization or FakeUtilization()
        self.energy = energy or FakeEnergy()
        self.thermal = thermal or FakeThermal()


class FakeGPU:
    """A `DeviceProbe`-shaped stand-in: a whole fake device, live and snapshottable.

    snapshot_error, when set, makes every `snapshot()` call raise it once (test hook for
    the sampler's failed-read path).
    """

    def __init__(
        self,
        *,
        vendor: str = "unknown",
        label: str = "probe",
        arch_key: str = "unknown",
        peak_bandwidth_gbs: float = 0.0,
        utilization: FakeUtilization | None = None,
        memory: FakeMemory | None = None,
        reading: FakeSnapshot | None = None,
    ) -> None:
        self.vendor = vendor
        self.label = label
        self.arch_key = arch_key
        self.peak_bandwidth_gbs = peak_bandwidth_gbs
        self.utilization = utilization or FakeUtilization()
        self.memory = memory or FakeMemory()
        self.reading = reading or FakeSnapshot()

    def snapshot(self, name: str = "") -> FakeSnapshot:
        del name
        return self.reading


def one_process_gpu() -> FakeGPU:
    """A GPU whose snapshot carries the current process at a fixed memory footprint.

    Mirrors the profiler-sampling fixture the old package called `one_gpu`: 40 bytes
    used by this process, 25% compute / 10% memory-controller utilization.
    """
    return FakeGPU(
        label="probe",
        reading=FakeSnapshot(
            unit_name="probe",
            processes=(FakeProcess(pid=os.getpid(), used_bytes=40),),
            utilization=FakeUtilization(gpu_pct=25, memory_pct=10),
        ),
    )


def clock_tracer() -> Tracer:
    """A no-op tracer with a monotonic device clock and KERNEL/MEMCPY deep support.

    Stands in for any real backend in the sampling/annotation tests: `timestamp`
    ticks a counter for region windows, and `open` hands back the no-op base
    collector so a deep trace records windows without a GPU.
    """

    class ClockTracer(Tracer):
        def __init__(self) -> None:
            self.clock = 0

        def open(self, kinds: Activity) -> TraceCollector:
            del kinds
            return TraceCollector()

        def supported(self) -> Activity:
            return Activity.KERNEL | Activity.MEMCPY

        def timestamp(self) -> int:
            self.clock += 1
            return self.clock

    return ClockTracer()


@pytest.fixture
def one_gpu(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """A one-GPU host whose snapshot carries fixed telemetry, with a clock tracer installed."""
    gpu = one_process_gpu()
    monkeypatch.setattr(annotate, "_tracer", clock_tracer())
    return gpu
