# The `Profile`, the result of a profiling session, and what you do with it.

from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from .efficiency import EfficiencyReport
from .models import RegionStat, RegionSummary
from .perfetto import write_trace
from .timeline import DeviceTimeline
from .trace import ActivityRecord, BottleneckReport, KernelTrace, MemcpyTrace, RegionWindow

if TYPE_CHECKING:
    from os import PathLike


class DeviceEvidence(StrEnum):
    """What a session has to say about device evidence, so silence is never ambiguous.

    A profile that collected nothing from a GPU and a profile that was never asked to look
    at one render identically, which is how a session that quietly attached to no device at
    all reads as a run that simply did no GPU work. Saying which of the two happened is the
    difference between a result and an empty file.

    UNSOUGHT: neither device telemetry nor GPU activity was requested, so none is expected.
    ABSENT: it was requested and none came back, because no device was visible to this
        session or because the profiled code never touched the one that was.
    COLLECTED: at least one device reading, kernel, copy or activity was observed.
    """

    UNSOUGHT = auto()
    ABSENT = auto()
    COLLECTED = auto()


class RegionDelta(FrozenModel):
    """One region's change between two profiles (baseline → current)."""

    name: str
    baseline_ms: float
    current_ms: float
    delta_ms: float
    speedup: float  # baseline / current; >1 is faster


class ProfileDiff(FrozenModel):
    """Region deltas between two profiles, ranked by absolute wall-time change."""

    rows: tuple[RegionDelta, ...]
    baseline_host: str = ""
    current_host: str = ""
    baseline_device: str = ""
    current_device: str = ""

    @classmethod
    def between(cls, baseline: Profile, *, current: Profile) -> ProfileDiff:
        """Match regions by name and rank by absolute wall-time change."""
        before = {s.name: s.total_ms for s in baseline.stats()}
        after = {s.name: s.total_ms for s in current.stats()}
        rows = tuple(
            RegionDelta(
                name=name,
                baseline_ms=before.get(name, 0.0),
                current_ms=after.get(name, 0.0),
                delta_ms=after.get(name, 0.0) - before.get(name, 0.0),
                speedup=before[name] / after[name]
                if name in before and after.get(name, 0.0) > 0
                else 0.0,
            )
            for name in before.keys() | after.keys()
        )
        return cls(
            rows=tuple(sorted(rows, key=lambda r: abs(r.delta_ms), reverse=True)),
            baseline_host=baseline.host,
            current_host=current.host,
            baseline_device=baseline.device,
            current_device=current.device,
        )


class Profile(FrozenModel):
    """One immutable result containing only evidence that was observed.

    Span timings, process GPU telemetry, and native activities are independently
    optional. A detected but unused GPU never creates output, so `device_evidence`
    carries whether the silence was expected, which no absent section can say.
    """

    host: str = ""
    device: str = ""
    device_evidence: DeviceEvidence = DeviceEvidence.UNSOUGHT
    summaries: tuple[RegionSummary, ...] = ()
    windows: tuple[RegionWindow, ...] = ()
    kernels: tuple[KernelTrace, ...] = ()
    memcpys: tuple[MemcpyTrace, ...] = ()
    activities: tuple[ActivityRecord, ...] = ()  # memset/runtime/driver/sync/... when enabled
    dropped_spans: int = 0
    dropped_activities: int = 0

    def __str__(self) -> str:
        return self.report()

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Profile:
        """Load a profile saved by :meth:`save`."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """The slowest region names by total wall time."""
        return self.stats()[:top]

    def diff(self, baseline: Profile) -> ProfileDiff:
        """Compare regions against a `baseline` profile."""
        return ProfileDiff.between(baseline, current=self)

    def efficiency(
        self,
        *,
        sm_count: int,
        peak_bandwidth_gbs: float = 0.0,
        bytes_moved: int = 0,
        blocks_per_sm: int = 1,
        top: int = 12,
    ) -> EfficiencyReport:
        """Per-kernel launch shape, wave quantisation and achieved bandwidth.

        A duration ranking says which kernel is slow and the timeline says whether the
        device was idle. Neither exposes a grid that leaves most block slots empty in its
        final wave, which reads as busy while draining a handful of blocks.
        """
        return EfficiencyReport.build(
            self.kernels,
            sm_count=sm_count,
            peak_bandwidth_gbs=peak_bandwidth_gbs,
            bytes_moved=bytes_moved,
            blocks_per_sm=blocks_per_sm,
            top=top,
        )

    def perfetto(self, path: str | PathLike[str]) -> None:
        """Write a Perfetto/Chrome timeline (open at ui.perfetto.dev)."""
        write_trace(self, path)

    def report(self) -> str:
        """A plain-text report containing only populated evidence sections."""
        sections = []
        if self.summaries:
            sections.append(f"Spans\n{Profile._region_text(self.stats())}")
        if self.kernels or self.memcpys or self.activities:
            report = self.trace_report()
            sections.append(
                "GPU activity\n"
                f"{len(self.kernels)} kernels, {len(self.memcpys)} copies, "
                f"{len(self.activities)} other activities, "
                f"{report.total_kernel_ns / 1e6:.2f} ms kernel time"
            )
        if self.device_evidence is DeviceEvidence.ABSENT:
            sections.append("Device\nno device evidence collected: asked for it and observed none")
        drops = []
        if self.dropped_spans:
            drops.append(f"{self.dropped_spans} oldest spans dropped")
        if self.dropped_activities:
            drops.append(f"{self.dropped_activities} oldest GPU activities dropped")
        if drops:
            sections.append("Capture limit\n" + "\n".join(drops))
        return "\n\n".join(sections) or "No profiling data collected."

    def save(self, path: str | PathLike[str]) -> None:
        """Persist to JSON so a later run can :meth:`load` and :meth:`diff` it."""
        Path(path).write_text(self.model_dump_json(), encoding="utf-8")

    def stats(self) -> list[RegionStat]:
        """Per-name aggregates (calls/total/avg/peak), slowest total first."""
        return RegionStat.aggregate(self.summaries)

    def timeline(self, top_gaps: int = 10) -> DeviceTimeline:
        """Busy/idle accounting over observed activity, with the longest idle windows.

        A kernel ranking says which kernel costs most; this says whether the device was
        working at all. A pipeline that reads kernel-bound is often idle between launches.
        """
        return DeviceTimeline.from_traces(self.kernels, self.memcpys, top_gaps=top_gaps)

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Deep GPU-time ranking (compute/copy split, hot regions and kernels)."""
        return BottleneckReport.from_traces(self.windows, self.kernels, self.memcpys, top)

    @staticmethod
    def _region_text(stats: list[RegionStat]) -> str:
        """Plain-text per-name table, the CLI-free fallback for `Profile.report`."""
        if not stats:
            return "No regions recorded."
        rows = [f"{'region':<30}{'calls':>6}{'total ms':>10}{'avg ms':>9}{'peak MB':>10}"]
        rows += [
            f"{s.name:<30}{s.calls:>6d}{s.total_ms:>10.2f}{s.avg_ms:>9.2f}"
            f"{s.peak_memory_bytes / 1024**2:>10.1f}"
            for s in stats
        ]
        return "\n".join(rows)
