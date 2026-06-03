from __future__ import annotations

from .base import FrozenModel


class MemoryUsage(FrozenModel):
    """Memory region usage visible to a processing unit.

    scope: memory region name, e.g. `system`, `vram`, `unified`.
    total_bytes: total capacity.
    used_bytes: currently used bytes when known.
    free_bytes: currently free bytes when known.
    unified: whether CPU and accelerator share the memory pool.
    source: provider that produced the value.
    supported: whether this platform exposes the reading.
    """

    scope: str
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    unified: bool = False
    source: str = ""
    supported: bool = True

    @property
    def total_gb(self) -> float:
        """Total capacity in gibibytes."""
        return self.total_bytes / 1024**3

    @property
    def used_gb(self) -> float:
        """Used capacity in gibibytes."""
        return self.used_bytes / 1024**3

    @property
    def free_gb(self) -> float:
        """Free capacity in gibibytes."""
        return self.free_bytes / 1024**3

    @property
    def utilization_pct(self) -> float:
        """Percentage of total memory currently used."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes * 100
