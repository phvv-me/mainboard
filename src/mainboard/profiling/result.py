"""The :class:`Profile` — the result of a profiling session, and what you do with it.

A :class:`~mainboard.profiling.profiler.Profiler` *runs*; a :class:`Profile` is the
immutable *result*. Everything you do with a measurement is a verb on this one value:
``stats`` / ``bottlenecks`` / ``trace_report`` to read it, :meth:`diff` to compare two
runs, :meth:`save` / :meth:`load` to persist, :meth:`perfetto` to export a timeline,
:meth:`show` to print it. New views (roofline, ncu/nsys ingest) are just more verbs
here, so the surface grows without new concepts.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from rich.console import RenderableType

from ..models.base import FrozenModel, FrozenSequence
from .counters import KernelCounterDelta, KernelCounters
from .efficiency import EfficiencyReport
from .models import RegionStat, RegionSummary
from .perfetto import write_trace
from .python import PythonProfile
from .render import profile_renderable, region_text, show_diff, show_profile
from .timeline import DeviceTimeline
from .trace import ActivityRecord, BottleneckReport, KernelTrace, MemcpyTrace, RegionWindow


class RegionDelta(FrozenModel):
    """One region's change between two profiles (baseline → current)."""

    name: str
    baseline_ms: float
    current_ms: float
    delta_ms: float
    speedup: float  # baseline / current; >1 is faster


class ProfileDiff(FrozenModel):
    """Region and demangled kernel deltas between two profiles."""

    rows: FrozenSequence[RegionDelta]
    kernels: FrozenSequence[KernelCounterDelta] = ()
    baseline_host: str = ""
    current_host: str = ""
    baseline_device: str = ""
    current_device: str = ""

    @classmethod
    def between(cls, baseline: Profile, current: Profile) -> ProfileDiff:
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
        baseline_kernels = {
            row.demangled_name: row for row in KernelCounters.aggregate_by_name(baseline.counters)
        }
        current_kernels = {
            row.demangled_name: row for row in KernelCounters.aggregate_by_name(current.counters)
        }
        kernel_rows = tuple(
            KernelCounterDelta.between(
                name,
                baseline_kernels.get(name),
                current_kernels.get(name),
            )
            for name in baseline_kernels.keys() | current_kernels.keys()
        )
        return cls(
            rows=tuple(sorted(rows, key=lambda r: abs(r.delta_ms), reverse=True)),
            kernels=tuple(
                sorted(
                    kernel_rows,
                    key=lambda row: abs(row.current_duration_ns - row.baseline_duration_ns),
                    reverse=True,
                )
            ),
            baseline_host=baseline.host,
            current_host=current.host,
            baseline_device=baseline.device,
            current_device=current.device,
        )

    def show(self, *, color: bool = True) -> None:
        """Print region and cross-host kernel deltas as rich tables."""
        show_diff(self, color=color)


class Profile(FrozenModel):
    """One immutable result containing only evidence that was observed.

    Python samples, span timings, process GPU telemetry, native activities, and replayed
    counters are independently optional. A detected but unused GPU never creates output.
    """

    python: PythonProfile | None = None
    host: str = ""
    device: str = ""
    summaries: FrozenSequence[RegionSummary] = ()
    windows: FrozenSequence[RegionWindow] = ()
    kernels: FrozenSequence[KernelTrace] = ()
    memcpys: FrozenSequence[MemcpyTrace] = ()
    activities: FrozenSequence[ActivityRecord] = ()  # memset/runtime/driver/sync/... when enabled
    counters: FrozenSequence[KernelCounters] = ()
    dropped_spans: int = 0
    dropped_activities: int = 0

    def stats(self) -> list[RegionStat]:
        """Per-name aggregates (calls/total/avg/peak), slowest total first."""
        return RegionStat.aggregate(self.summaries)

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """The slowest region names by total wall time."""
        return self.stats()[:top]

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Deep GPU-time ranking (compute/copy split, hot regions and kernels)."""
        return BottleneckReport.from_traces(self.windows, self.kernels, self.memcpys, top)

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

    def timeline(self, top_gaps: int = 10) -> DeviceTimeline:
        """Busy/idle accounting over observed activity, with the longest idle windows.

        A kernel ranking says which kernel costs most; this says whether the device was
        working at all. A pipeline that reads kernel-bound is often idle between launches.
        """
        return DeviceTimeline.from_traces(self.kernels, self.memcpys, top_gaps=top_gaps)

    def counter_bottlenecks(self, top: int = 10) -> tuple[KernelCounters, ...]:
        """Return the hottest demangled kernels from the counter pass."""
        return tuple(sorted(self.counters, key=lambda row: row.duration_ns, reverse=True)[:top])

    def diff(self, baseline: Profile) -> ProfileDiff:
        """Compare regions and demangled kernels against a `baseline` profile."""
        return ProfileDiff.between(baseline, self)

    def save(self, path: str | PathLike[str]) -> None:
        """Persist to JSON so a later run can :meth:`load` and :meth:`diff` it."""
        Path(path).write_text(self.model_dump_json())

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Profile:
        """Load a profile saved by :meth:`save`."""
        return cls.model_validate_json(Path(path).read_text())

    def perfetto(self, path: str | PathLike[str]) -> None:
        """Write a Perfetto/Chrome timeline (open at ui.perfetto.dev)."""
        write_trace(self, path)

    def show(self, *, color: bool = True) -> None:
        """Print a rich table of the region stats (and the deep report if traced)."""
        show_profile(self, color=color)

    def report(self) -> str:
        """A plain-text report containing only populated evidence sections."""
        sections = []
        if self.python is not None:
            sections.append(f"Python profile\n{self.python}")
        if self.summaries:
            sections.append(f"Spans\n{region_text(self.stats())}")
        if self.kernels or self.memcpys or self.activities:
            report = self.trace_report()
            sections.append(
                "GPU activity\n"
                f"{len(self.kernels)} kernels, {len(self.memcpys)} copies, "
                f"{len(self.activities)} other activities, "
                f"{report.total_kernel_ns / 1e6:.2f} ms kernel time"
            )
        if self.counters:
            rows = [
                f"{row.demangled_name} {row.duration_ns / 1e6:.3f} ms "
                f"{row.cycles:.0f} cycles {row.verdict.value}"
                for row in self.counter_bottlenecks()
            ]
            sections.append("GPU counters\n" + "\n".join(rows))
        drops = []
        if self.dropped_spans:
            drops.append(f"{self.dropped_spans} oldest spans dropped")
        if self.dropped_activities:
            drops.append(f"{self.dropped_activities} oldest GPU activities dropped")
        if drops:
            sections.append("Capture limit\n" + "\n".join(drops))
        return "\n\n".join(sections) or "No profiling data collected."

    def __str__(self) -> str:
        return self.report()

    def __rich__(self) -> RenderableType:
        """Rich renderable so ``print(profile)`` (rich/Jupyter) shows the tables."""
        return profile_renderable(self)
