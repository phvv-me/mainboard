from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum, auto

from ..models.base import FrozenModel, FrozenSequence

_DURATION = "gpu__time_duration.sum"
_CLOCK = "gpc__cycles_elapsed.avg.per_second"
_IPC = "sm__inst_executed.avg.per_cycle_active"
_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"
_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_L1_THROUGHPUT = "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
_L2_THROUGHPUT = "lts__throughput.avg.pct_of_peak_sustained_elapsed"
_DRAM_THROUGHPUT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
_SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
_REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
_STALL_METRICS = {
    "long_scoreboard": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "wait": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "branch_resolving": "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "no_instruction": "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "short_scoreboard": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "not_selected": "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "mio_throttle": "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "lg_throttle": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
}
_VERDICT_METRICS = {
    _DURATION,
    _CLOCK,
    _IPC,
    _OCCUPANCY,
    _SM_THROUGHPUT,
    _L1_THROUGHPUT,
    _L2_THROUGHPUT,
    _DRAM_THROUGHPUT,
    *_STALL_METRICS.values(),
}
_SATURATION_PCT = 80.0
_ISSUE_UTILIZATION_PCT = 60.0
_LATENCY_STALL_PCT = 30.0
_LOW_OCCUPANCY_PCT = 25.0
_CYCLE_EQUAL_TOLERANCE = 0.03
_DURATION_EQUAL_TOLERANCE = 0.05


class KernelVerdict(StrEnum):
    """The dominant performance limit supported by one kernel capture."""

    CLOCK_BOUND = auto()
    ISSUE_BOUND = auto()
    LATENCY_BOUND = auto()
    BANDWIDTH_BOUND = auto()
    UNKNOWN = auto()


class KernelDivergence(StrEnum):
    """Why one demangled kernel differs between two captures."""

    EQUAL = auto()
    CLOCK_BOUND = auto()
    ARCHITECTURAL = auto()
    INCOMPLETE = auto()
    MISSING = auto()


class KernelCounters(FrozenModel):
    """Aggregated Nsight Compute counters for one demangled kernel.

    Raw metrics and derived evidence live together so later analysis never has to
    reconstruct cycles, issue use, coalescing, or normalized stall shares.
    """

    name: str
    demangled_name: str
    calls: int = 1
    duration_ns: float = 0.0
    achieved_clock_hz: float = 0.0
    ipc: float = 0.0
    achieved_occupancy_pct: float = 0.0
    theoretical_occupancy_pct: float = 0.0
    occupancy_limit_blocks: float = 0.0
    occupancy_limit_registers: float = 0.0
    occupancy_limit_shared_memory: float = 0.0
    occupancy_limit_warps: float = 0.0
    stall_long_scoreboard_pct: float = 0.0
    stall_wait_pct: float = 0.0
    stall_branch_resolving_pct: float = 0.0
    stall_no_instruction_pct: float = 0.0
    stall_short_scoreboard_pct: float = 0.0
    stall_not_selected_pct: float = 0.0
    stall_mio_throttle_pct: float = 0.0
    stall_lg_throttle_pct: float = 0.0
    l1_hit_rate_pct: float = 0.0
    l2_hit_rate_pct: float = 0.0
    global_load_sectors: float = 0.0
    global_load_requests: float = 0.0
    sm_throughput_pct: float = 0.0
    l1_throughput_pct: float = 0.0
    l2_throughput_pct: float = 0.0
    dram_throughput_pct: float = 0.0
    cycles: float = 0.0
    issue_utilization_pct: float = 0.0
    sectors_per_request: float = 0.0
    stall_total_pct: float = 0.0
    stall_long_scoreboard_share_pct: float = 0.0
    stall_wait_share_pct: float = 0.0
    stall_branch_resolving_share_pct: float = 0.0
    stall_no_instruction_share_pct: float = 0.0
    stall_short_scoreboard_share_pct: float = 0.0
    stall_not_selected_share_pct: float = 0.0
    stall_mio_throttle_share_pct: float = 0.0
    stall_lg_throttle_share_pct: float = 0.0
    verdict: KernelVerdict = KernelVerdict.UNKNOWN
    observed_metrics: FrozenSequence[str] = ()

    @classmethod
    def from_metrics(
        cls,
        name: str,
        metrics: Mapping[str, float],
        *,
        demangled_name: str | None = None,
    ) -> "KernelCounters":
        """Build one launch from normalized raw metric values."""
        return cls.build(
            name=name,
            demangled_name=demangled_name or name,
            calls=1,
            duration_ns=metrics.get(_DURATION, 0.0),
            achieved_clock_hz=metrics.get(_CLOCK, 0.0),
            ipc=metrics.get(_IPC, 0.0),
            achieved_occupancy_pct=metrics.get(_OCCUPANCY, 0.0),
            theoretical_occupancy_pct=metrics.get("launch__occupancy_per_block_size", 0.0),
            occupancy_limit_blocks=metrics.get("launch__occupancy_limit_blocks", 0.0),
            occupancy_limit_registers=metrics.get("launch__occupancy_limit_registers", 0.0),
            occupancy_limit_shared_memory=metrics.get("launch__occupancy_limit_shared_mem", 0.0),
            occupancy_limit_warps=metrics.get("launch__occupancy_limit_warps", 0.0),
            stalls={label: metrics.get(metric, 0.0) for label, metric in _STALL_METRICS.items()},
            l1_hit_rate_pct=metrics.get("l1tex__t_sector_hit_rate.pct", 0.0),
            l2_hit_rate_pct=metrics.get("lts__t_sector_hit_rate.pct", 0.0),
            global_load_sectors=metrics.get(_SECTORS, 0.0),
            global_load_requests=metrics.get(_REQUESTS, 0.0),
            sm_throughput_pct=metrics.get(_SM_THROUGHPUT, 0.0),
            l1_throughput_pct=metrics.get(_L1_THROUGHPUT, 0.0),
            l2_throughput_pct=metrics.get(_L2_THROUGHPUT, 0.0),
            dram_throughput_pct=metrics.get(_DRAM_THROUGHPUT, 0.0),
            observed_metrics=tuple(sorted(metrics)),
        )

    @classmethod
    def build(
        cls,
        *,
        name: str,
        demangled_name: str,
        calls: int,
        duration_ns: float,
        achieved_clock_hz: float,
        ipc: float,
        achieved_occupancy_pct: float,
        theoretical_occupancy_pct: float,
        occupancy_limit_blocks: float,
        occupancy_limit_registers: float,
        occupancy_limit_shared_memory: float,
        occupancy_limit_warps: float,
        stalls: Mapping[str, float],
        l1_hit_rate_pct: float,
        l2_hit_rate_pct: float,
        global_load_sectors: float,
        global_load_requests: float,
        sm_throughput_pct: float,
        l1_throughput_pct: float,
        l2_throughput_pct: float,
        dram_throughput_pct: float,
        observed_metrics: Sequence[str],
        cycles: float | None = None,
    ) -> "KernelCounters":
        """Construct one aggregate and derive every interpretation field."""
        stall_total = sum(stalls.values())

        def share(value: float) -> float:
            return 100.0 * value / stall_total if stall_total else 0.0

        cycle_count = duration_ns * achieved_clock_hz / 1e9 if cycles is None else cycles
        issue_utilization = 100.0 * ipc / 4.0
        sectors_per_request = (
            global_load_sectors / global_load_requests if global_load_requests else 0.0
        )
        observed = tuple(sorted(set(observed_metrics)))
        verdict = cls.classify(
            observed,
            issue_utilization_pct=issue_utilization,
            achieved_occupancy_pct=achieved_occupancy_pct,
            stall_total_pct=stall_total,
            sm_throughput_pct=sm_throughput_pct,
            l1_throughput_pct=l1_throughput_pct,
            l2_throughput_pct=l2_throughput_pct,
            dram_throughput_pct=dram_throughput_pct,
        )
        return cls(
            name=name,
            demangled_name=demangled_name,
            calls=calls,
            duration_ns=duration_ns,
            achieved_clock_hz=achieved_clock_hz,
            ipc=ipc,
            achieved_occupancy_pct=achieved_occupancy_pct,
            theoretical_occupancy_pct=theoretical_occupancy_pct,
            occupancy_limit_blocks=occupancy_limit_blocks,
            occupancy_limit_registers=occupancy_limit_registers,
            occupancy_limit_shared_memory=occupancy_limit_shared_memory,
            occupancy_limit_warps=occupancy_limit_warps,
            stall_long_scoreboard_pct=stalls.get("long_scoreboard", 0.0),
            stall_wait_pct=stalls.get("wait", 0.0),
            stall_branch_resolving_pct=stalls.get("branch_resolving", 0.0),
            stall_no_instruction_pct=stalls.get("no_instruction", 0.0),
            stall_short_scoreboard_pct=stalls.get("short_scoreboard", 0.0),
            stall_not_selected_pct=stalls.get("not_selected", 0.0),
            stall_mio_throttle_pct=stalls.get("mio_throttle", 0.0),
            stall_lg_throttle_pct=stalls.get("lg_throttle", 0.0),
            l1_hit_rate_pct=l1_hit_rate_pct,
            l2_hit_rate_pct=l2_hit_rate_pct,
            global_load_sectors=global_load_sectors,
            global_load_requests=global_load_requests,
            sm_throughput_pct=sm_throughput_pct,
            l1_throughput_pct=l1_throughput_pct,
            l2_throughput_pct=l2_throughput_pct,
            dram_throughput_pct=dram_throughput_pct,
            cycles=cycle_count,
            issue_utilization_pct=issue_utilization,
            sectors_per_request=sectors_per_request,
            stall_total_pct=stall_total,
            stall_long_scoreboard_share_pct=share(stalls.get("long_scoreboard", 0.0)),
            stall_wait_share_pct=share(stalls.get("wait", 0.0)),
            stall_branch_resolving_share_pct=share(stalls.get("branch_resolving", 0.0)),
            stall_no_instruction_share_pct=share(stalls.get("no_instruction", 0.0)),
            stall_short_scoreboard_share_pct=share(stalls.get("short_scoreboard", 0.0)),
            stall_not_selected_share_pct=share(stalls.get("not_selected", 0.0)),
            stall_mio_throttle_share_pct=share(stalls.get("mio_throttle", 0.0)),
            stall_lg_throttle_share_pct=share(stalls.get("lg_throttle", 0.0)),
            verdict=verdict,
            observed_metrics=observed,
        )

    @classmethod
    def aggregate(cls, samples: Sequence["KernelCounters"]) -> "KernelCounters":
        """Aggregate repeated launches of one demangled kernel."""
        if not samples:
            raise ValueError("at least one kernel counter sample is required")
        demangled_name = samples[0].demangled_name
        if any(sample.demangled_name != demangled_name for sample in samples):
            raise ValueError("kernel counter samples must share one demangled name")
        return cls.build(
            name=samples[0].name,
            demangled_name=demangled_name,
            calls=sum(sample.calls for sample in samples),
            duration_ns=sum(sample.duration_ns for sample in samples),
            achieved_clock_hz=cls.weighted(samples, lambda sample: sample.achieved_clock_hz),
            ipc=cls.weighted(samples, lambda sample: sample.ipc),
            achieved_occupancy_pct=cls.weighted(
                samples, lambda sample: sample.achieved_occupancy_pct
            ),
            theoretical_occupancy_pct=cls.weighted(
                samples, lambda sample: sample.theoretical_occupancy_pct
            ),
            occupancy_limit_blocks=cls.weighted(
                samples, lambda sample: sample.occupancy_limit_blocks
            ),
            occupancy_limit_registers=cls.weighted(
                samples, lambda sample: sample.occupancy_limit_registers
            ),
            occupancy_limit_shared_memory=cls.weighted(
                samples, lambda sample: sample.occupancy_limit_shared_memory
            ),
            occupancy_limit_warps=cls.weighted(
                samples, lambda sample: sample.occupancy_limit_warps
            ),
            stalls={
                "long_scoreboard": cls.weighted(
                    samples, lambda sample: sample.stall_long_scoreboard_pct
                ),
                "wait": cls.weighted(samples, lambda sample: sample.stall_wait_pct),
                "branch_resolving": cls.weighted(
                    samples, lambda sample: sample.stall_branch_resolving_pct
                ),
                "no_instruction": cls.weighted(
                    samples, lambda sample: sample.stall_no_instruction_pct
                ),
                "short_scoreboard": cls.weighted(
                    samples, lambda sample: sample.stall_short_scoreboard_pct
                ),
                "not_selected": cls.weighted(
                    samples, lambda sample: sample.stall_not_selected_pct
                ),
                "mio_throttle": cls.weighted(
                    samples, lambda sample: sample.stall_mio_throttle_pct
                ),
                "lg_throttle": cls.weighted(samples, lambda sample: sample.stall_lg_throttle_pct),
            },
            l1_hit_rate_pct=cls.weighted(samples, lambda sample: sample.l1_hit_rate_pct),
            l2_hit_rate_pct=cls.weighted(samples, lambda sample: sample.l2_hit_rate_pct),
            global_load_sectors=sum(sample.global_load_sectors for sample in samples),
            global_load_requests=sum(sample.global_load_requests for sample in samples),
            sm_throughput_pct=cls.weighted(samples, lambda sample: sample.sm_throughput_pct),
            l1_throughput_pct=cls.weighted(samples, lambda sample: sample.l1_throughput_pct),
            l2_throughput_pct=cls.weighted(samples, lambda sample: sample.l2_throughput_pct),
            dram_throughput_pct=cls.weighted(samples, lambda sample: sample.dram_throughput_pct),
            observed_metrics=tuple(
                metric for sample in samples for metric in sample.observed_metrics
            ),
            cycles=sum(sample.cycles for sample in samples),
        )

    @classmethod
    def aggregate_by_name(
        cls, samples: Sequence["KernelCounters"]
    ) -> tuple["KernelCounters", ...]:
        """Collapse launches by demangled name and rank them by duration."""
        groups: defaultdict[str, list[KernelCounters]] = defaultdict(list)
        for sample in samples:
            groups[sample.demangled_name].append(sample)
        rows = (cls.aggregate(group) for group in groups.values())
        return tuple(sorted(rows, key=lambda row: row.duration_ns, reverse=True))

    @staticmethod
    def weighted(
        samples: Sequence["KernelCounters"], value: Callable[["KernelCounters"], float]
    ) -> float:
        """Return a duration weighted metric with a call weighted fallback."""
        duration = sum(sample.duration_ns for sample in samples)
        if duration:
            return sum(value(sample) * sample.duration_ns for sample in samples) / duration
        calls = sum(sample.calls for sample in samples)
        return sum(value(sample) * sample.calls for sample in samples) / calls if calls else 0.0

    @staticmethod
    def classify(
        observed_metrics: Sequence[str],
        *,
        issue_utilization_pct: float,
        achieved_occupancy_pct: float,
        stall_total_pct: float,
        sm_throughput_pct: float,
        l1_throughput_pct: float,
        l2_throughput_pct: float,
        dram_throughput_pct: float,
    ) -> KernelVerdict:
        """Choose one dominant limit from complete counter evidence.

        Eighty percent marks a nearly saturated hardware path. Sixty percent issue use
        indicates sustained scheduler pressure because mixed pipelines rarely reach the
        four instruction ceiling. Thirty percent selected stalls or occupancy below
        twenty five percent means latency hiding is materially constrained. Complete
        evidence below those limits is clock bound because cycles then scale mainly with
        achieved clock. Bandwidth wins when multiple limits cross their thresholds
        because issue pressure commonly feeds a saturated memory path. Incomplete
        evidence stays unknown.
        """
        if not _VERDICT_METRICS.issubset(observed_metrics):
            return KernelVerdict.UNKNOWN
        if max(l1_throughput_pct, l2_throughput_pct, dram_throughput_pct) >= _SATURATION_PCT:
            return KernelVerdict.BANDWIDTH_BOUND
        if issue_utilization_pct >= _ISSUE_UTILIZATION_PCT or sm_throughput_pct >= _SATURATION_PCT:
            return KernelVerdict.ISSUE_BOUND
        if stall_total_pct >= _LATENCY_STALL_PCT or achieved_occupancy_pct < _LOW_OCCUPANCY_PCT:
            return KernelVerdict.LATENCY_BOUND
        return KernelVerdict.CLOCK_BOUND


class KernelCounterDelta(FrozenModel):
    """One demangled kernel compared across two counter captures."""

    name: str
    baseline_duration_ns: float = 0.0
    current_duration_ns: float = 0.0
    baseline_cycles: float = 0.0
    current_cycles: float = 0.0
    baseline_clock_hz: float = 0.0
    current_clock_hz: float = 0.0
    duration_change_pct: float = 0.0
    cycle_change_pct: float = 0.0
    clock_change_pct: float = 0.0
    divergence: KernelDivergence = KernelDivergence.EQUAL

    @classmethod
    def between(
        cls,
        name: str,
        baseline: KernelCounters | None,
        current: KernelCounters | None,
    ) -> "KernelCounterDelta":
        """Classify one aligned kernel using cycle and duration signatures.

        Cycle counts within three percent are equal enough to absorb replay noise. A
        duration change above five percent with equal cycles is clock bound. A cycle
        change above three percent is architectural because the GPUs did different work
        in hardware. Missing kernels remain explicit instead of being misclassified.
        """
        if baseline is None or current is None:
            return cls(
                name=name,
                baseline_duration_ns=baseline.duration_ns if baseline is not None else 0.0,
                current_duration_ns=current.duration_ns if current is not None else 0.0,
                baseline_cycles=baseline.cycles if baseline is not None else 0.0,
                current_cycles=current.cycles if current is not None else 0.0,
                baseline_clock_hz=(baseline.achieved_clock_hz if baseline is not None else 0.0),
                current_clock_hz=current.achieved_clock_hz if current is not None else 0.0,
                divergence=KernelDivergence.MISSING,
            )
        if (
            min(
                baseline.duration_ns,
                current.duration_ns,
                baseline.cycles,
                current.cycles,
                baseline.achieved_clock_hz,
                current.achieved_clock_hz,
            )
            <= 0
        ):
            return cls(
                name=name,
                baseline_duration_ns=baseline.duration_ns,
                current_duration_ns=current.duration_ns,
                baseline_cycles=baseline.cycles,
                current_cycles=current.cycles,
                baseline_clock_hz=baseline.achieved_clock_hz,
                current_clock_hz=current.achieved_clock_hz,
                divergence=KernelDivergence.INCOMPLETE,
            )
        cycle_delta = cls.relative_change(baseline.cycles, current.cycles)
        duration_delta = cls.relative_change(baseline.duration_ns, current.duration_ns)
        clock_delta = cls.relative_change(baseline.achieved_clock_hz, current.achieved_clock_hz)
        cycle_difference = cls.relative_difference(baseline.cycles, current.cycles)
        duration_difference = cls.relative_difference(baseline.duration_ns, current.duration_ns)
        divergence = KernelDivergence.EQUAL
        if cycle_difference > _CYCLE_EQUAL_TOLERANCE:
            divergence = KernelDivergence.ARCHITECTURAL
        elif duration_difference > _DURATION_EQUAL_TOLERANCE:
            divergence = KernelDivergence.CLOCK_BOUND
        return cls(
            name=name,
            baseline_duration_ns=baseline.duration_ns,
            current_duration_ns=current.duration_ns,
            baseline_cycles=baseline.cycles,
            current_cycles=current.cycles,
            baseline_clock_hz=baseline.achieved_clock_hz,
            current_clock_hz=current.achieved_clock_hz,
            duration_change_pct=duration_delta,
            cycle_change_pct=cycle_delta,
            clock_change_pct=clock_delta,
            divergence=divergence,
        )

    @staticmethod
    def relative_difference(baseline: float, current: float) -> float:
        """Return symmetric relative difference with a stable zero case."""
        scale = max(abs(baseline), abs(current))
        return abs(current - baseline) / scale if scale else 0.0

    @staticmethod
    def relative_change(baseline: float, current: float) -> float:
        """Return signed percentage change with a stable zero baseline."""
        if baseline:
            return 100.0 * (current - baseline) / abs(baseline)
        return 0.0 if current == 0 else 100.0
