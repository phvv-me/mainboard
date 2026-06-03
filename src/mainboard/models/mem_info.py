from __future__ import annotations

from .base import FrozenModel


class MemInfo(FrozenModel):
    """GPU memory state.

    total_bytes: total installed GPU memory in bytes.
    used_bytes: currently allocated memory in bytes.
    free_bytes: unallocated memory in bytes.
    """

    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def used_mb(self) -> float:
        return self.used_bytes / (1024 * 1024)

    @property
    def free_mb(self) -> float:
        return self.free_bytes / (1024 * 1024)

    @property
    def utilization_pct(self) -> float:
        """Percentage of total memory currently used."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes * 100
