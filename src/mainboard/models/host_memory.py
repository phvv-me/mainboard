from __future__ import annotations

from functools import cached_property
from typing import Any

import psutil

from .base import FrozenModel
from .memory_card import MemoryCard


class HostMemory(FrozenModel):
    """Live system RAM snapshot.

    All properties are lazily evaluated; no data is captured at construction.
    """

    @property
    def _vm(self) -> Any:
        """Snapshot of virtual memory at first access."""
        return psutil.virtual_memory()

    @cached_property
    def _sm(self) -> Any:
        """Snapshot of swap memory at first access."""
        return psutil.swap_memory()

    @cached_property
    def cards(self) -> tuple[MemoryCard, ...]:
        """DIMM slot details from `dmidecode`; empty when unavailable."""
        return MemoryCard.all()

    @property
    def total_bytes(self) -> int:
        """Total installed RAM in bytes."""
        return self._vm.total

    @property
    def available_bytes(self) -> int:
        """Currently available RAM in bytes."""
        return self._vm.available

    @property
    def used_bytes(self) -> int:
        """RAM currently in use in bytes."""
        return self._vm.used

    @property
    def swap_total_bytes(self) -> int:
        """Total swap space in bytes."""
        return self._sm.total

    @property
    def swap_used_bytes(self) -> int:
        """Swap currently in use in bytes."""
        return self._sm.used

    @property
    def total_gb(self) -> float:
        """Total installed RAM in gibibytes."""
        return self.total_bytes / 1024**3

    @property
    def available_gb(self) -> float:
        """Currently available RAM in gibibytes."""
        return self.available_bytes / 1024**3

    @property
    def used_gb(self) -> float:
        """Currently used RAM in gibibytes."""
        return self.used_bytes / 1024**3

    @property
    def swap_total_gb(self) -> float:
        """Total swap space in gibibytes."""
        return self.swap_total_bytes / 1024**3

    @property
    def utilization_pct(self) -> float:
        """Percentage of total RAM currently in use."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes * 100

    @property
    def speed_mhz(self) -> int | None:
        """Maximum speed across populated slots; None when `cards` is unavailable."""
        speeds = [c.speed_mhz for c in self.cards if c.populated and c.speed_mhz]
        return max(speeds) if speeds else None

    @property
    def slots_total(self) -> int:
        """Total DIMM slot count; 0 when `cards` is unavailable."""
        return len(self.cards)

    @property
    def slots_used(self) -> int:
        """Number of populated DIMM slots."""
        return sum(1 for c in self.cards if c.populated)
