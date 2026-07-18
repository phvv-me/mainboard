"""Aggregated profiling results: one row per region from its sampled snapshots."""

from collections import defaultdict
from typing import TYPE_CHECKING

from ..models.base import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..models.gpu_snapshot import GPUSnapshot


class RegionSummary(FrozenModel):
    """Wall time and aggregated device telemetry for one profiled region.

    samples: number of snapshots taken during the region. peak/avg memory are over
    those snapshots; util/power/temp are sampled means (max for temperature).
    """

    name: str
    wall_ms: float
    samples: int = 0
    peak_memory_bytes: int = 0
    avg_memory_bytes: int = 0
    avg_util_pct: float = 0.0
    avg_memory_util_pct: float = 0.0
    avg_power_w: float = 0.0
    max_temp_c: int = 0

    @classmethod
    def from_snaps(cls, name: str, wall_ms: float, snaps: Sequence[GPUSnapshot]) -> RegionSummary:
        """Aggregate the snapshots sampled during a region into one summary."""
        if not snaps:
            return cls(name=name, wall_ms=wall_ms)
        memory = [s.memory.used_bytes for s in snaps]
        return cls(
            name=name,
            wall_ms=wall_ms,
            samples=len(snaps),
            peak_memory_bytes=max(memory),
            avg_memory_bytes=sum(memory) // len(memory),
            avg_util_pct=sum(s.utilization.gpu_pct for s in snaps) / len(snaps),
            avg_memory_util_pct=sum(s.utilization.memory_pct for s in snaps) / len(snaps),
            avg_power_w=sum(s.energy.power_w for s in snaps) / len(snaps),
            max_temp_c=max(s.thermal.temperature_c for s in snaps),
        )


class RegionStat(FrozenModel):
    """One region name's aggregate across all its occurrences (calls collapsed).

    The readable unit for a profile: a region called many times becomes one row with
    its call count, total and mean wall time, and peak memory — not one row per call.
    """

    name: str
    calls: int
    total_ms: float
    avg_ms: float
    peak_memory_bytes: int
    max_util_pct: float
    max_power_w: float

    @classmethod
    def aggregate(cls, summaries: Sequence[RegionSummary]) -> list[RegionStat]:
        """Collapse per-occurrence summaries into per-name stats, slowest total first."""
        groups: defaultdict[str, list[RegionSummary]] = defaultdict(list)
        for summary in summaries:
            groups[summary.name].append(summary)
        stats = [
            cls(
                name=name,
                calls=len(rows),
                total_ms=sum(r.wall_ms for r in rows),
                avg_ms=sum(r.wall_ms for r in rows) / len(rows),
                peak_memory_bytes=max(r.peak_memory_bytes for r in rows),
                max_util_pct=max(r.avg_util_pct for r in rows),
                max_power_w=max(r.avg_power_w for r in rows),
            )
            for name, rows in groups.items()
        ]
        return sorted(stats, key=lambda s: s.total_ms, reverse=True)
