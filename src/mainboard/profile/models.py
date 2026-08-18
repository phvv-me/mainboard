# Aggregated profiling results: one row per region from its sampled snapshots.

from collections import defaultdict
from collections.abc import Sequence

from patos import FrozenModel


class ProcessReading(FrozenModel):
    """One process-scoped device telemetry sample, attributed to a profiled span.

    Built from a `DeviceSnapshot` narrowed to the profiled process's own memory; the
    other fields (utilization/power/temperature) stay whole-device signals, since a
    per-process split for those is not something a device sensor reports.

    unit_name: human-readable device name at the moment of the reading.
    memory_used_bytes: device memory used by the profiled process alone.
    gpu_util_pct: device compute utilization percent.
    memory_util_pct: device memory-controller utilization percent.
    power_w: instantaneous power draw in watts.
    temperature_c: die temperature in degrees Celsius.
    """

    unit_name: str = ""
    memory_used_bytes: int = 0
    gpu_util_pct: float = 0.0
    memory_util_pct: float = 0.0
    power_w: float = 0.0
    temperature_c: int = 0


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
    def from_snaps(
        cls, name: str, wall_ms: float, snaps: Sequence[ProcessReading]
    ) -> "RegionSummary":
        """Aggregate the snapshots sampled during a region into one summary."""
        if not snaps:
            return cls(name=name, wall_ms=wall_ms)
        memory = [s.memory_used_bytes for s in snaps]
        return cls(
            name=name,
            wall_ms=wall_ms,
            samples=len(snaps),
            peak_memory_bytes=max(memory),
            avg_memory_bytes=sum(memory) // len(memory),
            avg_util_pct=sum(s.gpu_util_pct for s in snaps) / len(snaps),
            avg_memory_util_pct=sum(s.memory_util_pct for s in snaps) / len(snaps),
            avg_power_w=sum(s.power_w for s in snaps) / len(snaps),
            max_temp_c=max(s.temperature_c for s in snaps),
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
    def aggregate(cls, summaries: Sequence[RegionSummary]) -> "list[RegionStat]":
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
