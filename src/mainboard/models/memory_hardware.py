from functools import cached_property

import psutil

from .base import FrozenModel
from .memory_card import MemoryCard
from .psutil_results import SwapMemory


class MemoryHardware(FrozenModel):
    """Physical DIMM slots and swap space for the host.

    All properties are lazily evaluated; no data is captured at construction.
    """

    @cached_property
    def _swap(self) -> SwapMemory:
        """Snapshot of swap memory at first access."""
        swap: SwapMemory = psutil.swap_memory()
        return swap

    @cached_property
    def cards(self) -> tuple[MemoryCard, ...]:
        """DIMM slot details from `dmidecode`; empty when unavailable."""
        return MemoryCard.all()

    @property
    def swap_total_bytes(self) -> int:
        """Total swap space in bytes."""
        return self._swap.total

    @property
    def swap_used_bytes(self) -> int:
        """Swap currently in use in bytes."""
        return self._swap.used

    @property
    def swap_total_gb(self) -> float:
        """Total swap space in gibibytes."""
        return self.swap_total_bytes / 1024**3

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
