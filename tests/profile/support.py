import os
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field

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


class FakeActivityKind:
    """The `cupti.ActivityKind` members the collector enables, numbered like CUPTI's own."""

    CONCURRENT_KERNEL = 10
    MEMCPY = 1
    MEMSET = 4
    SYNCHRONIZATION = 8
    OVERHEAD = 16
    MEMORY = 32
    JIT = 64
    RUNTIME = 128
    DRIVER = 256
    MEMORY_POOL = 512


class RecordingSession:
    """A `SpanSession` stand-in that records every name opened and every wall time closed."""

    def __init__(self) -> None:
        self.entered: list[str] = []
        self.walls: list[int] = []

    def enter(self, name: str) -> int:
        self.entered.append(name)
        return len(self.entered)

    def exit(self, token: int, *, wall_ns: int) -> None:
        del token
        self.walls.append(wall_ns)


@dataclass(kw_only=True)
class FakeUtilization:
    """A `DeviceUtilization`-shaped stand-in."""

    gpu_pct: int = 0
    memory_pct: int = 0


@dataclass
class FakeEnergy:
    """A `DeviceEnergy`-shaped stand-in."""

    power_w: float = 0.0


@dataclass
class FakeThermal:
    """A `DeviceThermal`-shaped stand-in."""

    temperature_c: int = 0
    _: KW_ONLY
    is_throttling: bool = False
    throttle_names: Sequence[str] = ()


@dataclass(kw_only=True)
class FakeMemory:
    """A `DeviceMemory`-shaped stand-in."""

    total_gb: float = 0.0
    percent_used: float = 0.0


@dataclass(kw_only=True)
class FakeProcess:
    """A `DeviceProcess`-shaped stand-in: one process's device memory footprint."""

    pid: int
    used_bytes: int


@dataclass
class FakeSnapshot:
    """A `DeviceSnapshot`-shaped stand-in: one point-in-time device reading."""

    unit_name: str = "probe"
    processes: Sequence[FakeProcess] = ()
    utilization: FakeUtilization = field(default_factory=FakeUtilization)
    energy: FakeEnergy = field(default_factory=FakeEnergy)
    thermal: FakeThermal = field(default_factory=FakeThermal)


@dataclass(kw_only=True)
class FakeGPU:
    """A `DeviceProbe`-shaped stand-in: a whole fake device, live and snapshottable."""

    vendor: str = "unknown"
    label: str = "probe"
    arch_key: str = "unknown"
    peak_bandwidth_gbs: float = 0.0
    utilization: FakeUtilization = field(default_factory=FakeUtilization)
    memory: FakeMemory = field(default_factory=FakeMemory)
    reading: FakeSnapshot = field(default_factory=FakeSnapshot)

    def snapshot(self, name: str = "") -> FakeSnapshot:
        del name
        return self.reading


def one_process_gpu() -> FakeGPU:
    """A GPU whose snapshot shows this process at 40 bytes, 25% compute, 10% memory utilization."""
    return FakeGPU(
        reading=FakeSnapshot(
            processes=(FakeProcess(pid=os.getpid(), used_bytes=40),),
            utilization=FakeUtilization(gpu_pct=25, memory_pct=10),
        ),
    )


def clock_tracer() -> Tracer:
    """A no-op tracer with a ticking device clock and KERNEL/MEMCPY support.

    `open` hands back the no-op base collector, so a deep trace records windows without a GPU.
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
    """A `KernelTrace` lasting `ns` nanoseconds from `start_ns`; shape fields stay typed."""
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
    """Two regions: `gemm` straddles both, `relu` sits in the first, plus one copy and one
    generic activity."""
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
    console = Console(no_color=True, width=120, record=True)
    console.print(renderable)
    return console.export_text()
