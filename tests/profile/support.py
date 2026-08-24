import os
from collections.abc import Sequence

from rich.console import Console, RenderableType

from mainboard.profile import (
    Activity,
    ActivityRecord,
    KernelTrace,
    MemcpyTrace,
    Profile,
    RegionSummary,
    RegionWindow,
    TraceCollector,
    Tracer,
)


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


def kernel(
    name: str,
    ns: int,
    *,
    start_ns: int = 0,
    grid: str = "",
    block: str = "",
    registers: int = 0,
    static_shared_mem: int = 0,
    dynamic_shared_mem: int = 0,
) -> KernelTrace:
    """A `KernelTrace` named `name` lasting `ns` nanoseconds from `start_ns`.

    Every launch-shape field is spelled out so callers name only the axis they vary and
    the call stays type-checked, which a `**shape` passthrough cannot be.
    """
    return KernelTrace(
        name=name,
        start_ns=start_ns,
        end_ns=start_ns + ns,
        grid=grid,
        block=block,
        registers=registers,
        static_shared_mem=static_shared_mem,
        dynamic_shared_mem=dynamic_shared_mem,
    )


def traced_profile() -> Profile:
    """A profile with two regions and kernels/memcpys binned across their device windows.

    The one fixture every deep-report reader shares: `gemm` straddles both regions, `relu`
    sits inside the first, and there is exactly one copy and one generic activity.
    """
    return Profile(
        device="dev",
        summaries=(
            RegionSummary(name="encode", wall_ms=2.0, avg_util_pct=50.0, avg_power_w=100.0),
            RegionSummary(name="decode", wall_ms=1.0),
        ),
        windows=(
            RegionWindow(name="encode", start_ns=0, end_ns=1000, wall_ns=2_000_000),
            RegionWindow(name="decode", start_ns=1000, end_ns=2000, wall_ns=1_000_000),
        ),
        kernels=(
            kernel("gemm", 600, grid="8x1x1", block="256x1x1"),
            kernel("gemm", 400, start_ns=1000),
            kernel("relu", 100, start_ns=600),
        ),
        memcpys=(MemcpyTrace(kind="HtoD", start_ns=0, end_ns=100, bytes_moved=4096),),
        activities=(
            ActivityRecord(kind="runtime", name="cudaLaunchKernel", start_ns=0, end_ns=5),
        ),
    )


def render(renderable: RenderableType) -> str:
    """Render a rich renderable to plain text, for content assertions."""
    console = Console(no_color=True, width=120, record=True)
    console.print(renderable)
    return console.export_text()
