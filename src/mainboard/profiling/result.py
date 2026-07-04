"""The :class:`Profile` — the result of a profiling session, and what you do with it.

A :class:`~mainboard.profiling.profiler.Profiler` *runs*; a :class:`Profile` is the
immutable *result*. Everything you do with a measurement is a verb on this one value:
``stats`` / ``bottlenecks`` / ``trace_report`` to read it, :meth:`diff` to compare two
runs, :meth:`save` / :meth:`load` to persist, :meth:`perfetto` to export a timeline,
:meth:`show` to print it. New views (roofline, ncu/nsys ingest) are just more verbs
here, so the surface grows without new concepts.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from ..models.base import FrozenModel
from .models import RegionStat, RegionSummary
from .trace import ActivityRecord, BottleneckReport, KernelTrace, MemcpyTrace, RegionWindow

if TYPE_CHECKING:
    from os import PathLike

    from rich.console import RenderableType


class RegionDelta(FrozenModel):
    """One region's change between two profiles (baseline → current)."""

    name: str
    baseline_ms: float
    current_ms: float
    delta_ms: float
    speedup: float  # baseline / current; >1 is faster


class ProfileDiff(FrozenModel):
    """Per-region deltas between two profiles, largest absolute change first."""

    rows: tuple[RegionDelta, ...]

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
        return cls(rows=tuple(sorted(rows, key=lambda r: abs(r.delta_ms), reverse=True)))

    def show(self, *, color: bool = True) -> None:
        """Print the deltas as a rich table (green = faster, red = slower)."""
        from .render import show_diff

        show_diff(self, color=color)


class Profile(FrozenModel):
    """An immutable profiling result: region telemetry + deep op traces.

    device: the sampled accelerator's name. summaries: per-occurrence region telemetry.
    windows/kernels/memcpys: the deep-trace timeline (present when ``trace=True``).
    """

    device: str = ""
    summaries: tuple[RegionSummary, ...] = ()
    windows: tuple[RegionWindow, ...] = ()
    kernels: tuple[KernelTrace, ...] = ()
    memcpys: tuple[MemcpyTrace, ...] = ()
    activities: tuple[ActivityRecord, ...] = ()  # memset/runtime/driver/sync/... when enabled

    def stats(self) -> list[RegionStat]:
        """Per-name aggregates (calls/total/avg/peak), slowest total first."""
        return RegionStat.aggregate(self.summaries)

    def bottlenecks(self, top: int = 10) -> list[RegionStat]:
        """The slowest region names by total wall time."""
        return self.stats()[:top]

    def trace_report(self, top: int = 10) -> BottleneckReport:
        """Deep GPU-time ranking (compute/copy split, hot regions and kernels)."""
        return BottleneckReport.from_traces(self.windows, self.kernels, self.memcpys, top)

    def diff(self, baseline: Profile) -> ProfileDiff:
        """Compare this run against a ``baseline`` profile, region by region."""
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
        from .perfetto import write_trace

        write_trace(self, path)

    def show(self, *, color: bool = True) -> None:
        """Print a rich table of the region stats (and the deep report if traced)."""
        from .render import show_profile

        show_profile(self, color=color)

    def report(self) -> str:
        """A plain-text per-name table — the no-rich fallback of :meth:`show`."""
        from .render import region_text

        return region_text(self.stats())

    def __str__(self) -> str:
        return self.report()

    def __rich__(self) -> RenderableType:
        """Rich renderable so ``print(profile)`` (rich/Jupyter) shows the tables."""
        from .render import profile_renderable

        return profile_renderable(self)
