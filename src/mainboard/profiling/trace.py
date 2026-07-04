"""Deep per-operation tracing: kernel/memcpy records and bottleneck ranking.

The deep tier is *opt-in* (``Profiler(trace=True)``) and built for minimal impact:
collection is asynchronous and buffered (CUPTI Activity, ROCprofiler) so nothing runs
on the application's launch path, records are attributed to regions **after the fact**
by GPU-timestamp binning (no per-region synchronize in the default mode), and a single
device sync drains the buffers at session end. The records are vendor-neutral
:class:`KernelTrace` / :class:`MemcpyTrace`; :class:`BottleneckReport` ranks where GPU
time goes. :class:`TraceCollector` is the no-op base a vendor backend overrides.
"""

from collections import defaultdict
from enum import Flag
from types import TracebackType
from typing import TYPE_CHECKING

from ..models.base import FrozenModel
from .protocols import KernelActivity, MemcpyActivity

if TYPE_CHECKING:
    from collections.abc import Sequence


class Activity(Flag):
    """The CUPTI activity kinds to trace, combined with ``|``.

    Vendor backends map each member to their native activity kind. ``KERNEL`` and
    ``MEMCPY`` become typed :class:`KernelTrace` / :class:`MemcpyTrace`; the rest become
    generic :class:`ActivityRecord`s. ``DEFAULT`` (kernels + memcpy) is the minimal,
    low-impact set; ``ALL`` is every kind (runtime/driver are high-volume — opt in). The
    ``label`` is the lowercase name used in records and on the Perfetto timeline.
    """

    KERNEL = 1
    MEMCPY = 2
    MEMSET = 4
    SYNC = 8
    OVERHEAD = 16
    MEMORY = 32
    JIT = 64
    RUNTIME = 128
    DRIVER = 256
    MEMORY_POOL = 512
    DEFAULT = KERNEL | MEMCPY
    ALL = (
        KERNEL | MEMCPY | MEMSET | SYNC | OVERHEAD | MEMORY | JIT | RUNTIME | DRIVER | MEMORY_POOL
    )

    @property
    def label(self) -> str:
        return self.name.lower() if self.name else "activity"


_MEMCPY_KIND = {
    0: "unknown",
    1: "HtoD",
    2: "DtoH",
    3: "HtoA",
    4: "AtoH",
    5: "AtoA",
    6: "AtoD",
    7: "DtoA",
    8: "DtoD",
    9: "HtoH",
    10: "PtoP",
}


_MAX_THREADS_PER_BLOCK = 1024  # the hardware cap, the occupancy-proxy denominator


class KernelTrace(FrozenModel):
    """One GPU kernel execution, with its device-clock start/end in nanoseconds.

    The launch shape (``grid``/``block``, shared memory split into static/dynamic,
    registers per thread) comes straight from the CUPTI activity record so a bottleneck
    report can reason about occupancy without re-launching the kernel.
    """

    name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    grid: str = ""
    block: str = ""
    static_shared_mem: int = 0
    dynamic_shared_mem: int = 0
    registers: int = 0

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0

    @property
    def shared_mem(self) -> int:
        """Total per-block shared memory (static + dynamic) in bytes."""
        return self.static_shared_mem + self.dynamic_shared_mem

    @property
    def threads_per_block(self) -> int:
        """Threads in one block — the product of the block dimensions."""
        product = 1
        for dim in self.block.split("x"):
            product *= int(dim) if dim.isdigit() else 1
        return product

    @property
    def occupancy_pct(self) -> float:
        """Launch-shape occupancy proxy: threads-per-block over the hardware max (1024).

        The base CUPTI activity record carries the launch config but not achieved
        occupancy, so this is the static upper bound the block size alone implies.
        """
        return 100.0 * self.threads_per_block / _MAX_THREADS_PER_BLOCK

    @classmethod
    def from_activity(cls, act: KernelActivity) -> KernelTrace:
        """Build from a CUPTI CONCURRENT_KERNEL activity (snake_case attributes)."""
        return cls(
            name=act.name,
            start_ns=act.start,
            end_ns=act.end,
            grid=f"{act.grid_x}x{act.grid_y}x{act.grid_z}",
            block=f"{act.block_x}x{act.block_y}x{act.block_z}",
            static_shared_mem=act.static_shared_memory,
            dynamic_shared_mem=act.dynamic_shared_memory,
            registers=act.registers_per_thread,
        )


class MemcpyTrace(FrozenModel):
    """One memory copy, with device-clock start/end, direction, and bytes moved."""

    kind: str = "unknown"
    start_ns: int = 0
    end_ns: int = 0
    bytes_moved: int = 0

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def bandwidth_gbps(self) -> float:
        return self.bytes_moved / self.duration_ns if self.duration_ns > 0 else 0.0

    @classmethod
    def from_activity(cls, act: MemcpyActivity) -> MemcpyTrace:
        """Build from a CUPTI MEMCPY activity (snake_case attributes)."""
        return cls(
            kind=_MEMCPY_KIND.get(int(act.copy_kind), f"kind_{act.copy_kind}"),
            start_ns=act.start,
            end_ns=act.end,
            bytes_moved=getattr(act, "bytes", 0),
        )


class ActivityRecord(FrozenModel):
    """A generic timed CUPTI activity — the kinds beyond kernel/memcpy.

    kind: the activity-kind label (``memset``/``runtime``/``driver``/``sync``/...).
    name: the resolved name (API function for runtime/driver, else the kind label).
    """

    kind: str = ""
    name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    correlation_id: int = 0

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class RegionWindow(FrozenModel):
    """A region's device-clock window, used to attribute kernels by timestamp."""

    name: str
    start_ns: int
    end_ns: int
    wall_ns: int


class TraceCollector:
    """No-op deep-trace collector; a vendor backend overrides to gather records.

    Context manager: enter starts collection, exit drains it. The records carry
    device-clock timestamps so the profiler bins them into regions afterwards.
    """

    def __enter__(self) -> TraceCollector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def stop(self) -> None:
        """Drain and stop collection (a single device sync happens here)."""

    def reset(self) -> None:
        """Drain and discard collected records, to start a fresh measurement window."""

    def flush(self) -> None:
        """Deliver buffered records to this collector (no clear) so reads see them."""

    def kernels(self) -> list[KernelTrace]:
        return []

    def memcpys(self) -> list[MemcpyTrace]:
        return []

    def activities(self) -> list[ActivityRecord]:
        """Generic timed records for the enabled non-kernel/memcpy activity kinds."""
        return []


class CallbackSession:
    """No-op CUPTI Callback subscription; vendor backends count API calls by name.

    The Callback API intercepts CUDA runtime/driver calls *synchronously* (unlike the
    asynchronous Activity stream), so it answers "which API functions were called, and
    how often" without buffering. Use as a context manager; read :meth:`counts` after.
    """

    def __enter__(self) -> CallbackSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def stop(self) -> None:
        """Unsubscribe from the callback domains."""

    def counts(self) -> dict[str, int]:
        """API function name -> number of calls observed."""
        return {}


class HotKernel(FrozenModel):
    """A kernel name's share of total kernel time."""

    name: str
    calls: int
    total_ns: int
    avg_ns: float
    share_pct: float


class HotRegion(FrozenModel):
    """A region's share of total kernel time (kernels binned by timestamp)."""

    name: str
    kernel_count: int
    kernel_ns: int
    wall_ns: int
    share_pct: float


class BottleneckReport(FrozenModel):
    """Where GPU time goes: compute-vs-copy split, hot regions and hot kernels."""

    total_kernel_ns: int
    total_memcpy_ns: int
    total_memcpy_bytes: int
    compute_pct: float
    memcpy_pct: float
    hot_regions: tuple[HotRegion, ...]
    hot_kernels: tuple[HotKernel, ...]

    @classmethod
    def from_traces(
        cls,
        windows: Sequence[RegionWindow],
        kernels: Sequence[KernelTrace],
        memcpys: Sequence[MemcpyTrace],
        top: int = 10,
    ) -> BottleneckReport:
        """Bin kernels into region windows by start timestamp and rank the hot spots."""
        total_kernel = sum(k.duration_ns for k in kernels)
        total_memcpy = sum(m.duration_ns for m in memcpys)
        denom = total_kernel + total_memcpy or 1
        return cls(
            total_kernel_ns=total_kernel,
            total_memcpy_ns=total_memcpy,
            total_memcpy_bytes=sum(m.bytes_moved for m in memcpys),
            compute_pct=100.0 * total_kernel / denom,
            memcpy_pct=100.0 * total_memcpy / denom,
            hot_regions=cls._hot_regions(windows, kernels, total_kernel or 1, top),
            hot_kernels=cls._hot_kernels(kernels, total_kernel or 1, top),
        )

    @staticmethod
    def _region_of(start_ns: int, windows: Sequence[RegionWindow]) -> str:
        """The innermost region containing ``start_ns`` — the narrowest matching window.

        Nested regions share the outer's window, so attribute a kernel to the tightest
        enclosing region (smallest span), not the last one appended. Kernels in no window
        (e.g. work awaited by a ``synchronize`` outside any ``region``) are labeled
        ``(outside regions)`` so unattributed GPU time stays visible, not silently blank.
        """
        best, best_span = "(outside regions)", None
        for window in windows:
            if not window.start_ns <= start_ns < window.end_ns:
                continue
            span = window.end_ns - window.start_ns
            if best_span is None or span < best_span:
                best, best_span = window.name, span
        return best

    @classmethod
    def _hot_regions(
        cls,
        windows: Sequence[RegionWindow],
        kernels: Sequence[KernelTrace],
        total: int,
        top: int,
    ) -> tuple[HotRegion, ...]:
        counts: defaultdict[str, int] = defaultdict(int)
        nanos: defaultdict[str, int] = defaultdict(int)
        for kernel in kernels:
            name = cls._region_of(kernel.start_ns, windows)
            counts[name] += 1
            nanos[name] += kernel.duration_ns
        walls = {w.name: w.wall_ns for w in windows}
        ranked = sorted(nanos.items(), key=lambda kv: kv[1], reverse=True)[:top]
        return tuple(
            HotRegion(
                name=name,
                kernel_count=counts[name],
                kernel_ns=ns,
                wall_ns=walls.get(name, 0),
                share_pct=100.0 * ns / total,
            )
            for name, ns in ranked
        )

    @staticmethod
    def _hot_kernels(
        kernels: Sequence[KernelTrace],
        total: int,
        top: int,
    ) -> tuple[HotKernel, ...]:
        counts: defaultdict[str, int] = defaultdict(int)
        nanos: defaultdict[str, int] = defaultdict(int)
        for kernel in kernels:
            counts[kernel.name] += 1
            nanos[kernel.name] += kernel.duration_ns
        ranked = sorted(nanos.items(), key=lambda kv: kv[1], reverse=True)[:top]
        return tuple(
            HotKernel(
                name=name,
                calls=counts[name],
                total_ns=ns,
                avg_ns=ns / counts[name],
                share_pct=100.0 * ns / total,
            )
            for name, ns in ranked
        )
