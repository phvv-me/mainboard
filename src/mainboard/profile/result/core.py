# The `Profile`, the result of a profiling session, and what you do with it.

from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ...core.constants.units import BYTES_PER_KIBIBYTE, NANOSECONDS_PER_MILLISECOND
from ..trace import ActivityRecord, BottleneckReport, KernelTrace, MemcpyTrace, RegionWindow
from .efficiency import EfficiencyReport
from .enums import DeviceEvidence
from .models import RegionStat, RegionSummary
from .perfetto import write_trace
from .timeline import DeviceTimeline

if TYPE_CHECKING:
    from os import PathLike


class Profile(FrozenModel):
    """One immutable result containing only evidence that was observed.

    Span timings, process GPU telemetry and native activities are independently optional, and
    a detected but unused GPU creates no output, so `device_evidence` says whether silence
    was expected.
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

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Profile:
        """Load a profile saved by :meth:`save`."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _region_text(stats: list[RegionStat]) -> str:
        """Plain-text per-name table for `report`."""
        if not stats:
            return "No regions recorded."
        rows = [f"{'region':<30}{'calls':>6}{'total ms':>10}{'avg ms':>9}{'peak MB':>10}"]
        rows += [
            f"{s.name:<30}{s.calls:>6d}{s.total_ms:>10.2f}{s.avg_ms:>9.2f}"
            f"{s.peak_memory_bytes / BYTES_PER_KIBIBYTE**2:>10.1f}"
            for s in stats
        ]
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.report()

    def _device_section(self) -> str:
        if self.device_evidence is DeviceEvidence.ABSENT:
            return "Device\nno device evidence collected: asked for it and observed none"
        return ""

    def _gpu_section(self) -> str:
        if not (self.kernels or self.memcpys or self.activities):
            return ""
        report = self.trace_report()
        return (
            "GPU activity\n"
            f"{len(self.kernels)} kernels, {len(self.memcpys)} copies, "
            f"{len(self.activities)} other activities, "
            f"{report.total_kernel_ns / NANOSECONDS_PER_MILLISECOND:.2f} ms kernel time"
        )

    def _limit_section(self) -> str:
        drops = []
        if self.dropped_spans:
            drops.append(f"{self.dropped_spans} oldest spans dropped")
        if self.dropped_activities:
            drops.append(f"{self.dropped_activities} GPU activities dropped")
        if not drops:
            return ""
        return "Capture limit\n" + "\n".join(drops)

    def _spans_section(self) -> str:
        if not self.summaries:
            return ""
        return f"Spans\n{Profile._region_text(self.stats())}"

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """The slowest region names by total wall time."""
        return self.stats()[:top]

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

        Exposes a grid whose final wave leaves most block slots empty, which a duration
        ranking or the timeline reads as busy.
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
        sections = [
            self._spans_section(),
            self._gpu_section(),
            self._device_section(),
            self._limit_section(),
        ]
        return (
            "\n\n".join(section for section in sections if section)
            or "No profiling data collected."
        )

    def save(self, path: str | PathLike[str]) -> None:
        """Persist to JSON so a later run can :meth:`load` it."""
        Path(path).write_text(self.model_dump_json(), encoding="utf-8", newline="\n")

    def stats(self) -> list[RegionStat]:
        """Per-name aggregates (calls/total/avg/peak), slowest total first."""
        return RegionStat.aggregate(self.summaries)

    def timeline(self, top_gaps: int = 10) -> DeviceTimeline:
        """Busy/idle accounting over observed activity, with the longest idle windows.

        A pipeline that reads kernel-bound is often idle between launches.
        """
        return DeviceTimeline.from_traces(self.kernels, self.memcpys, top_gaps=top_gaps)

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Deep GPU-time ranking (compute/copy split, hot regions and kernels)."""
        return BottleneckReport.from_traces(self.windows, self.kernels, self.memcpys, top)
