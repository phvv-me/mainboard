# One-call bottleneck reporting: run a callable, say where the GPU time went.

from collections.abc import Sequence
from enum import Enum

from patos import FrozenModel

from .result import DeviceEvidence, Profile
from .trace import Activity, BottleneckReport, HotKernel, KernelTrace


class Bound(Enum):
    """Whether the dominant work is limited by memory traffic or compute throughput.

    UNKNOWN when there was nothing to classify: no kernels, no copies, no utilization signal.
    """

    MEMORY = "memory"
    COMPUTE = "compute"
    UNKNOWN = "unknown"


class KernelStat(FrozenModel):
    """One kernel name's aggregate over the run: its share and representative shape.

    The shape fields come from the last-seen launch of the name, since launches of one name
    share a config. occupancy_pct: threads per block over the hardware max of 1024, a proxy,
    since the CUPTI activity record carries the launch config but not achieved occupancy.
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
    total_kernel_ns/total_memcpy_ns/device_busy_ns: summed WORK time per class and the CLOCK
        time busy with either, as on `BottleneckReport`.
    achieved_bandwidth_gbps/peak_bandwidth_gbps: copy bandwidth against the device peak, the
        memory-bound signal.
    peak_memory_bytes/avg_memory_bytes: the sampled device-memory high-water mark and mean,
        how much HBM the work needed.
    kernels: the per-kernel breakdown, hottest first.
    unavailable: activity-kind labels the device could not trace.
    device_evidence: whether device evidence was asked for and came back, so an empty verdict
        never reads like a real one.
    """

    device: str = ""
    device_evidence: DeviceEvidence = DeviceEvidence.UNSOUGHT
    iterations: int = 0
    dominant_kernel: str = ""
    dominant_share_pct: float = 0.0
    bound: Bound = Bound.UNKNOWN
    total_kernel_ns: int = 0
    total_memcpy_ns: int = 0
    total_memcpy_bytes: int = 0
    device_busy_ns: int = 0
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

    def __str__(self) -> str:
        return self.report()

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
        """Distill a `Profile` into a bottleneck verdict.

        peak_bandwidth_gbps: device peak to score copy bandwidth against, 0 disables the score.
        supported/requested: the `Activity` values the device offered and the run asked for;
            their difference becomes `unavailable`, so a partial trace is visible.
        """
        split = BottleneckReport.from_traces(
            (), profile.kernels, profile.memcpys, top=len(profile.kernels)
        )
        kernels = cls._kernels(profile.kernels, split.hot_kernels)
        mem_util, gpu_util = cls._utilization(profile)
        peak_memory, avg_memory = cls._memory(profile)
        copy_ns = split.total_memcpy_ns
        return cls(
            device=profile.device,
            device_evidence=profile.device_evidence,
            iterations=iterations,
            dominant_kernel=kernels[0].name if kernels else "",
            dominant_share_pct=kernels[0].share_pct if kernels else 0.0,
            bound=cls._classify(
                kernel_ns=split.total_kernel_ns,
                memcpy_ns=copy_ns,
                mem_util=mem_util,
                gpu_util=gpu_util,
            ),
            total_kernel_ns=split.total_kernel_ns,
            total_memcpy_ns=copy_ns,
            total_memcpy_bytes=split.total_memcpy_bytes,
            device_busy_ns=split.device_busy_ns,
            compute_pct=split.compute_pct,
            memcpy_pct=split.memcpy_pct,
            # Bytes per nanosecond equals GB/s.
            achieved_bandwidth_gbps=split.total_memcpy_bytes / copy_ns if copy_ns > 0 else 0.0,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
            peak_memory_bytes=peak_memory,
            avg_memory_bytes=avg_memory,
            avg_memory_util_pct=mem_util,
            avg_compute_util_pct=gpu_util,
            kernels=kernels,
            unavailable=cls._unavailable(supported=supported, requested=requested),
        )

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

    @staticmethod
    def _classify(*, kernel_ns: int, memcpy_ns: int, mem_util: float, gpu_util: float) -> Bound:
        """Memory- or compute-bound by the copy/kernel time split, else by utilization."""
        if kernel_ns or memcpy_ns:
            return Bound.MEMORY if memcpy_ns >= kernel_ns else Bound.COMPUTE
        if mem_util or gpu_util:
            return Bound.MEMORY if mem_util >= gpu_util else Bound.COMPUTE
        return Bound.UNKNOWN

    @staticmethod
    def _kernels(
        kernels: Sequence[KernelTrace], hottest: Sequence[HotKernel]
    ) -> tuple[KernelStat, ...]:
        """Attach each hot kernel name's last-seen launch shape."""
        shape = {kernel.name: kernel for kernel in kernels}
        return tuple(
            KernelStat(
                name=hot.name,
                calls=hot.calls,
                total_ns=hot.total_ns,
                avg_ns=hot.avg_ns,
                share_pct=hot.share_pct,
                grid=shape[hot.name].grid,
                block=shape[hot.name].block,
                threads_per_block=shape[hot.name].threads_per_block,
                occupancy_pct=shape[hot.name].occupancy_pct,
                registers=shape[hot.name].registers,
                static_shared_mem=shape[hot.name].static_shared_mem,
                dynamic_shared_mem=shape[hot.name].dynamic_shared_mem,
            )
            for hot in hottest
        )

    @staticmethod
    def _memory(profile: Profile) -> tuple[int, int]:
        """Device-memory (peak single sample, mean across regions) in bytes.

        Zero when nothing was sampled, e.g. a kernel that finished between two sampler ticks,
        where the deep kernel trace remains the reliable signal.
        """
        summaries = profile.summaries
        if not summaries:
            return 0, 0
        peak = max(s.peak_memory_bytes for s in summaries)
        avg = sum(s.avg_memory_bytes for s in summaries) // len(summaries)
        return peak, avg

    @staticmethod
    def _unavailable(*, supported: int | None, requested: int | None) -> tuple[str, ...]:
        if supported is None or requested is None:
            return ()
        missing = Activity(requested) & ~Activity(supported)
        return tuple(flag.label for flag in Activity if flag in missing and flag.label != "all")

    @staticmethod
    def _utilization(profile: Profile) -> tuple[float, float]:
        """Mean sampled (memory-controller, compute) utilization across all regions."""
        summaries = profile.summaries
        if not summaries:
            return 0.0, 0.0
        compute = sum(s.avg_util_pct for s in summaries) / len(summaries)
        memory = sum(s.avg_memory_util_pct for s in summaries) / len(summaries)
        return memory, compute

    def _notes(self) -> str:
        """Trailing notes: missing device evidence, peak memory, bandwidth, untraced kinds."""
        notes = []
        if self.device_evidence is DeviceEvidence.ABSENT:
            notes.append("no device evidence collected: asked for it and observed none")
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
