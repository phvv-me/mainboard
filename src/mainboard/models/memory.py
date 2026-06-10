from __future__ import annotations

from .base import FrozenModel


class Memory(FrozenModel):
    """Memory usage for a host, unit, or memory region.

    total_bytes: total capacity.
    used_bytes: currently used bytes when known.
    free_bytes: currently free bytes when known.
    scope: region name, e.g. `system`, `vram`, `unified`.
    unified: whether CPU and accelerator share the memory pool.
    source: provider that produced the value.
    supported: whether this platform exposes the reading.
    """

    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    scope: str = ""
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
    def percent_used(self) -> float:
        """Percentage of total memory currently used; 0 when total is 0."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes * 100
