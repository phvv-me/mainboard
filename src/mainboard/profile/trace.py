# Native activity records, span attribution, and bottleneck ranking.

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from enum import Flag, auto
from types import TracebackType

from patos import FrozenModel

from .protocols import KernelActivity, MemcpyActivity


class Activity(Flag):
    """The activity kinds to trace, combined with `|`, each mapped to a vendor's native kind.

    KERNEL and MEMCPY become typed `KernelTrace`/`MemcpyTrace`, the rest `ActivityRecord`.
    DEFAULT is the minimal, low-impact set; ALL adds the high-volume runtime/driver kinds.
    """

    KERNEL = auto()
    MEMCPY = auto()
    MEMSET = auto()
    SYNC = auto()
    OVERHEAD = auto()
    MEMORY = auto()
    JIT = auto()
    RUNTIME = auto()
    DRIVER = auto()
    MEMORY_POOL = auto()
    DEFAULT = KERNEL | MEMCPY
    ALL = (
        KERNEL | MEMCPY | MEMSET | SYNC | OVERHEAD | MEMORY | JIT | RUNTIME | DRIVER | MEMORY_POOL
    )

    @property
    def label(self) -> str:
        """The lowercase name used in records and on the Perfetto timeline."""
        return self.name.lower() if self.name else "activity"


_MEMCPY_KIND = dict(
    enumerate(
        ("unknown", "HtoD", "DtoH", "HtoA", "AtoH", "AtoA", "AtoD", "DtoA", "DtoD", "HtoH", "PtoP")
    )
)
_MAX_THREADS_PER_BLOCK = 1024  # the hardware cap, the occupancy-proxy denominator


def memcpy_kind(code: int) -> str:
    """A CUPTI copy-kind code's direction, `kind_<code>` outside the known table."""
    return _MEMCPY_KIND.get(code, f"kind_{code}")


class KernelTrace(FrozenModel):
    """One GPU kernel execution in device-clock nanoseconds, with its CUPTI launch shape.

    correlation_id: the runtime call that launched it, a separate host-clock activity; the
        join that attributes a kernel to its callsite rather than only to its region.
    """

    name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    correlation_id: int = 0
    grid: str = ""
    block: str = ""
    static_shared_mem: int = 0
    dynamic_shared_mem: int = 0
    registers: int = 0
    device_id: int | None = None
    context_id: int | None = None
    stream_id: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0

    @property
    def occupancy_pct(self) -> float:
        """Threads per block over the hardware max, the static upper bound on occupancy.

        The CUPTI activity record carries the launch config but not achieved occupancy.
        """
        return 100.0 * self.threads_per_block / _MAX_THREADS_PER_BLOCK

    @property
    def shared_mem(self) -> int:
        """Per-block static plus dynamic shared memory in bytes."""
        return self.static_shared_mem + self.dynamic_shared_mem

    @property
    def threads_per_block(self) -> int:
        """The product of the block dimensions, a non-numeric one counting as 1."""
        return math.prod(int(dim) if dim.isdigit() else 1 for dim in self.block.split("x"))

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
            correlation_id=getattr(act, "correlation_id", 0),
            device_id=getattr(act, "device_id", None),
            context_id=getattr(act, "context_id", None),
            stream_id=getattr(act, "stream_id", None),
        )


class MemcpyTrace(FrozenModel):
    """One memory copy in device-clock nanoseconds, with its direction and bytes moved.

    correlation_id: the runtime call that issued the copy, as on a kernel.
    """

    kind: str = "unknown"
    start_ns: int = 0
    end_ns: int = 0
    correlation_id: int = 0
    bytes_moved: int = 0
    device_id: int | None = None
    context_id: int | None = None
    stream_id: int | None = None

    @property
    def bandwidth_gbps(self) -> float:
        return self.bytes_moved / self.duration_ns if self.duration_ns > 0 else 0.0

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @classmethod
    def from_activity(cls, act: MemcpyActivity) -> MemcpyTrace:
        """Build from a CUPTI MEMCPY activity (snake_case attributes)."""
        return cls(
            kind=memcpy_kind(int(act.copy_kind)),
            start_ns=act.start,
            end_ns=act.end,
            bytes_moved=getattr(act, "bytes", 0),
            correlation_id=getattr(act, "correlation_id", 0),
            device_id=getattr(act, "device_id", None),
            context_id=getattr(act, "context_id", None),
            stream_id=getattr(act, "stream_id", None),
        )


class ActivityRecord(FrozenModel):
    """A timed activity of a kind beyond kernel/memcpy.

    kind: the `Activity.label` (`memset`, `runtime`, `driver`, `sync`, ...).
    name: the API function for runtime/driver, else the kind label.
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
    """No-op deep-trace collector; a vendor backend overrides it to gather records.

    Enter starts collection and exit drains it; the profiler bins the device-clock records into
    regions afterwards.
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

    def activities(self) -> list[ActivityRecord]:
        """Generic timed records for the enabled non-kernel/memcpy activity kinds."""
        return []

    def checkpoint(self, activities: Activity) -> tuple[int, int]:
        """Drain one context; return its delivered-record and lifetime-loss counters.

        The requested kinds must actually be enabled; a collector without windows refuses.
        """
        raise RuntimeError("synchronized activity windows are unavailable on this backend")

    @property
    def device_index(self) -> int:
        """Return the captured CUDA-visible ordinal, not a physical NVML index."""
        raise RuntimeError("this collector has no synchronized CUDA device")

    def dropped(self) -> int:
        """Number of native records discarded by a bounded capture buffer."""
        return 0

    def flush(self) -> None:
        """Deliver buffered records, without clearing, so reads see them."""

    def kernels(self, *, since: int | None = None, until: int | None = None) -> list[KernelTrace]:
        return []

    def memcpys(self, *, since: int | None = None, until: int | None = None) -> list[MemcpyTrace]:
        return []

    def reset(self) -> None:
        """Drain and discard collected records, starting a fresh measurement window."""

    def stop(self) -> None:
        """Drain and stop collection (a single device sync happens here)."""


class CallbackSession:
    """No-op CUPTI Callback subscription; vendor backends count API calls by name.

    The Callback API intercepts runtime/driver calls synchronously, unlike the asynchronous
    Activity stream, so it counts calls without buffering. Read `counts` after the context.
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

    def counts(self) -> dict[str, int]:
        """Calls observed per API function name."""
        return {}

    def stop(self) -> None:
        """Unsubscribe from the callback domains."""


def busy_ns(spans: Iterable[tuple[int, int]]) -> int:
    """How long the device was busy: the union of the half-open `(start_ns, end_ns)` spans.

    A share against wall must divide this, never a summed duration. A sum counts concurrent or
    nested work (two streams, a copy under a kernel, a launching call beside its kernel) once
    per record, which answers how much WORK was done. The factor is structural, not
    statistical: reproducibility's `recovery_cost` read device shares of 1.908 and 1.860 of its
    own wall from a sum, and its 2026-08-29 referee reproduced that at ten launches.
    """
    ordered = sorted((start, end) for start, end in spans if end > start)
    busy, reach = 0, None
    for start, end in ordered:
        if reach is None or start > reach:
            busy += end - start
            reach = end
        elif end > reach:
            busy += end - reach
            reach = end
    return busy


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
    """Where GPU time goes: compute-vs-copy split, hot regions and hot kernels.

    total_kernel_ns/total_memcpy_ns: summed WORK time, one entry per traced record.
    device_busy_ns: CLOCK time busy with either (see `busy_ns`), the only one of the three a
        share against wall may divide.
    """

    total_kernel_ns: int
    total_memcpy_ns: int
    total_memcpy_bytes: int
    device_busy_ns: int
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
        share = total_kernel or 1
        walls = {w.name: w.wall_ns for w in windows}
        by_region = ((cls._region_of(k.start_ns, windows), k.duration_ns) for k in kernels)
        return cls(
            total_kernel_ns=total_kernel,
            total_memcpy_ns=total_memcpy,
            total_memcpy_bytes=sum(m.bytes_moved for m in memcpys),
            device_busy_ns=busy_ns(
                [
                    *((span.start_ns, span.end_ns) for span in kernels),
                    *((span.start_ns, span.end_ns) for span in memcpys),
                ]
            ),
            compute_pct=100.0 * total_kernel / denom,
            memcpy_pct=100.0 * total_memcpy / denom,
            hot_regions=tuple(
                HotRegion(
                    name=name,
                    kernel_count=calls,
                    kernel_ns=ns,
                    wall_ns=walls.get(name, 0),
                    share_pct=100.0 * ns / share,
                )
                for name, calls, ns in _hottest(by_region, top)
            ),
            hot_kernels=tuple(
                HotKernel(
                    name=name,
                    calls=calls,
                    total_ns=ns,
                    avg_ns=ns / calls,
                    share_pct=100.0 * ns / share,
                )
                for name, calls, ns in _hottest(((k.name, k.duration_ns) for k in kernels), top)
            ),
        )

    @staticmethod
    def _region_of(start_ns: int, windows: Sequence[RegionWindow]) -> str:
        """The narrowest window holding `start_ns`, since nested regions share the outer's.

        A kernel in no window (e.g. work awaited by a `synchronize` outside any region) is
        labeled `(outside regions)`, so unattributed GPU time stays visible.
        """
        best, best_span = "(outside regions)", None
        for window in windows:
            if not window.start_ns <= start_ns < window.end_ns:
                continue
            span = window.end_ns - window.start_ns
            if best_span is None or span < best_span:
                best, best_span = window.name, span
        return best


def _hottest(timed: Iterable[tuple[str, int]], top: int) -> list[tuple[str, int, int]]:
    """`(name, calls, total_ns)` per name, the `top` largest totals first."""
    calls: Counter[str] = Counter()
    nanos: Counter[str] = Counter()
    for name, ns in timed:
        calls[name] += 1
        nanos[name] += ns
    return [(name, calls[name], ns) for name, ns in nanos.most_common(top)]
