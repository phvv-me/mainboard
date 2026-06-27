"""One-call bottleneck reporting: run a callable, say where the GPU time went.

:class:`ProfileReport` is the structured answer a kernel-optimization caller wants from
a single :func:`~mainboard.profiling.bottleneck.profile` run — the dominant kernel and
its share, whether the work is memory- or compute-bound, its launch shape (occupancy
proxy, registers/thread, shared memory), the total kernel time, and the full per-kernel
breakdown. It is built from an immutable :class:`~mainboard.profiling.result.Profile`
plus the device's peak bandwidth, so it adds a verdict on top of the raw traces rather
than re-collecting them. ``unavailable`` records the activity kinds the device could not
trace, so a partial report on a GPU like GB10 (no ``MEMORY`` kind) reads as partial
rather than silently wrong.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING

from ..models.base import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .result import Profile
    from .trace import KernelTrace, MemcpyTrace


class Bound(Enum):
    """Whether the dominant work is limited by memory traffic or compute throughput.

    ``MEMORY`` when copies dominate the GPU time or the memory controller is the busier
    unit; ``COMPUTE`` when kernel math dominates; ``UNKNOWN`` when there was nothing to
    classify (no kernels, no copies, no utilization signal).
    """

    MEMORY = "memory"
    COMPUTE = "compute"
    UNKNOWN = "unknown"


class KernelStat(FrozenModel):
    """One kernel name's aggregate over the run: its share and representative shape.

    The shape fields come from the last-seen launch of this name (kernels of one name
    share a launch config), so the report can show occupancy/registers/shared without a
    per-call row explosion. ``occupancy_pct`` is a launch-shape proxy: threads-per-block
    over the hardware max (1024), since the base CUPTI activity record carries the launch
    config but not achieved occupancy.
    """

    name: str
    calls: int
    total_ns: int
    avg_ns: float
    share_pct: float
    grid: str
    block: str
    threads_per_block: int
    occupancy_pct: float
    registers: int
    static_shared_mem: int
    dynamic_shared_mem: int


class ProfileReport(FrozenModel):
    """Structured bottleneck verdict for one profiled callable.

    dominant_kernel/dominant_share_pct: the hottest kernel and its slice of kernel time.
    bound: the memory-vs-compute verdict (:class:`Bound`). total_kernel_ns/total_memcpy_ns:
    summed GPU time per class. achieved_bandwidth_gbps/peak_bandwidth_gbps: copy bandwidth
    measured against the device peak (the memory-bound signal). peak_memory_bytes/
    avg_memory_bytes: the device-memory high-water mark and mean over the sampled run — the
    answer to "how much HBM did this kernel need". kernels: the per-kernel breakdown,
    hottest first. unavailable: activity-kind labels the device could not trace.
    """

    device: str = ""
    iterations: int = 0
    dominant_kernel: str = ""
    dominant_share_pct: float = 0.0
    bound: Bound = Bound.UNKNOWN
    total_kernel_ns: int = 0
    total_memcpy_ns: int = 0
    total_memcpy_bytes: int = 0
    compute_pct: float = 0.0
    memcpy_pct: float = 0.0
    achieved_bandwidth_gbps: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    peak_memory_bytes: int = 0
    avg_memory_bytes: int = 0
    avg_memory_util_pct: float = 0.0
    avg_compute_util_pct: float = 0.0
    kernels: tuple[KernelStat, ...] = ()
    unavailable: tuple[str, ...] = ()

    @classmethod
    def from_profile(
        cls,
        profile: Profile,
        *,
        iterations: int,
        peak_bandwidth_gbps: float,
        supported: int | None = None,
        requested: int | None = None,
    ) -> ProfileReport:
        """Distill a :class:`Profile` into a bottleneck verdict.

        peak_bandwidth_gbps: device peak, to score copy bandwidth (0 disables the score).
        supported/requested: the :class:`Activity` kinds the device offered and the run
        asked for; their difference becomes ``unavailable`` so a partial trace is visible.
        """
        kernels = cls._kernels(profile.kernels)
        total_kernel = sum(k.duration_ns for k in profile.kernels)
        total_memcpy = sum(m.duration_ns for m in profile.memcpys)
        memcpy_bytes = sum(m.bytes_moved for m in profile.memcpys)
        achieved = cls._copy_bandwidth_gbps(profile.memcpys)
        denom = total_kernel + total_memcpy or 1
        mem_util, gpu_util = cls._utilization(profile)
        peak_memory, avg_memory = cls._memory(profile)
        return cls(
            device=profile.device,
            iterations=iterations,
            dominant_kernel=kernels[0].name if kernels else "",
            dominant_share_pct=kernels[0].share_pct if kernels else 0.0,
            bound=cls._classify(total_kernel, total_memcpy, mem_util, gpu_util),
            total_kernel_ns=total_kernel,
            total_memcpy_ns=total_memcpy,
            total_memcpy_bytes=memcpy_bytes,
            compute_pct=100.0 * total_kernel / denom,
            memcpy_pct=100.0 * total_memcpy / denom,
            achieved_bandwidth_gbps=achieved,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
            peak_memory_bytes=peak_memory,
            avg_memory_bytes=avg_memory,
            avg_memory_util_pct=mem_util,
            avg_compute_util_pct=gpu_util,
            kernels=kernels,
            unavailable=cls._unavailable(supported, requested),
        )

    @staticmethod
    def _classify(kernel_ns: int, memcpy_ns: int, mem_util: float, gpu_util: float) -> Bound:
        """Memory- vs compute-bound from the copy/compute time split and util signal.

        Copies dominating the GPU time, or the memory controller out-busying the SMs, says
        memory-bound; the reverse says compute-bound. With neither time nor a util signal
        there is nothing to judge, so the verdict is ``UNKNOWN``.
        """
        if kernel_ns or memcpy_ns:
            return Bound.MEMORY if memcpy_ns >= kernel_ns else Bound.COMPUTE
        if mem_util or gpu_util:
            return Bound.MEMORY if mem_util >= gpu_util else Bound.COMPUTE
        return Bound.UNKNOWN

    @staticmethod
    def _copy_bandwidth_gbps(memcpys: Sequence[MemcpyTrace]) -> float:
        """Achieved copy bandwidth: total bytes over total copy time (GB/s), 0 if no copy."""
        nanos = sum(m.duration_ns for m in memcpys)
        if nanos <= 0:
            return 0.0
        return sum(m.bytes_moved for m in memcpys) / nanos  # bytes/ns == GB/s

    @staticmethod
    def _utilization(profile: Profile) -> tuple[float, float]:
        """Mean sampled (memory-controller, compute) utilization across all regions."""
        summaries = profile.summaries
        if not summaries:
            return 0.0, 0.0
        compute = sum(s.avg_util_pct for s in summaries) / len(summaries)
        memory = sum(s.avg_memory_util_pct for s in summaries) / len(summaries)
        return memory, compute

    @staticmethod
    def _memory(profile: Profile) -> tuple[int, int]:
        """Device-memory (peak high-water mark, mean) in bytes over the sampled regions.

        Peak is the largest single sample (the footprint the kernel needs to fit), mean is
        averaged across regions. Zero when nothing was sampled — e.g. a kernel that finished
        between two sampler ticks, where the deep kernel trace remains the reliable signal.
        """
        summaries = profile.summaries
        if not summaries:
            return 0, 0
        peak = max(s.peak_memory_bytes for s in summaries)
        avg = sum(s.avg_memory_bytes for s in summaries) // len(summaries)
        return peak, avg

    @staticmethod
    def _kernels(kernels: Sequence[KernelTrace]) -> tuple[KernelStat, ...]:
        """Collapse kernel traces into per-name stats, hottest total time first."""
        order: list[str] = []
        shape: dict[str, KernelTrace] = {}
        counts: defaultdict[str, int] = defaultdict(int)
        nanos: defaultdict[str, int] = defaultdict(int)
        for kernel in kernels:
            if kernel.name not in shape:
                order.append(kernel.name)
            shape[kernel.name] = kernel
            counts[kernel.name] += 1
            nanos[kernel.name] += kernel.duration_ns
        total = sum(nanos.values()) or 1
        stats = [
            KernelStat(
                name=name,
                calls=counts[name],
                total_ns=nanos[name],
                avg_ns=nanos[name] / counts[name],
                share_pct=100.0 * nanos[name] / total,
                grid=shape[name].grid,
                block=shape[name].block,
                threads_per_block=shape[name].threads_per_block,
                occupancy_pct=shape[name].occupancy_pct,
                registers=shape[name].registers,
                static_shared_mem=shape[name].static_shared_mem,
                dynamic_shared_mem=shape[name].dynamic_shared_mem,
            )
            for name in order
        ]
        return tuple(sorted(stats, key=lambda k: k.total_ns, reverse=True))

    @staticmethod
    def _unavailable(supported: int | None, requested: int | None) -> tuple[str, ...]:
        """Labels for the requested activity kinds the device could not trace."""
        if supported is None or requested is None:
            return ()
        from .trace import Activity

        missing = Activity(requested) & ~Activity(supported)
        return tuple(flag.label for flag in Activity if flag in missing and flag.label != "all")

    def report(self) -> str:
        """A compact plain-text verdict and per-kernel table."""
        head = (
            f"device {self.device or 'cpu'} | {self.bound.value}-bound | "
            f"dominant {self.dominant_kernel or '(none)'} {self.dominant_share_pct:.1f}%"
        )
        if not self.kernels:
            return f"{head}\nNo kernels traced." + self._notes()
        rows = [f"{'kernel':<40}{'calls':>6}{'total ms':>10}{'share%':>8}{'regs':>6}"]
        rows += [
            f"{k.name[:40]:<40}{k.calls:>6d}{k.total_ns / 1e6:>10.3f}"
            f"{k.share_pct:>8.1f}{k.registers:>6d}"
            for k in self.kernels
        ]
        return "\n".join([head, *rows]) + self._notes()

    def _notes(self) -> str:
        """Trailing notes: peak memory, copy bandwidth vs peak, and any untraced kinds."""
        notes = []
        if self.peak_memory_bytes:
            notes.append(f"peak memory {self.peak_memory_bytes / 1e9:.3f} GB")
        if self.peak_bandwidth_gbps:
            pct = 100.0 * self.achieved_bandwidth_gbps / self.peak_bandwidth_gbps
            notes.append(
                f"copy bandwidth {self.achieved_bandwidth_gbps:.1f}/"
                f"{self.peak_bandwidth_gbps:.1f} GB/s ({pct:.0f}% of peak)"
            )
        if self.unavailable:
            notes.append(f"unavailable on this device: {', '.join(self.unavailable)}")
        return "\n" + "\n".join(notes) if notes else ""

    def __str__(self) -> str:
        return self.report()
