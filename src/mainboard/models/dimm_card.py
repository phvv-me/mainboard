from __future__ import annotations

from .base import FrozenModel


class DimmCard(FrozenModel):
    """One DIMM memory slot and its installed module (if any).

    locator: slot label, e.g. `Controller0-ChannelA` or `DIMM_A1`.
    bank: bank locator string, e.g. `BANK 0`.
    size_bytes: installed module size; 0 for an empty slot.
    speed_mhz: configured speed in MT/s; None if not reported.
    memory_type: DRAM type string, e.g. `LPDDR5`, `DDR5`.
    manufacturer: module manufacturer; None if not reported.
    part_number: module part number; None if not reported.
    """

    locator: str
    bank: str = ""
    size_bytes: int = 0
    speed_mhz: int | None = None
    memory_type: str = ""
    manufacturer: str | None = None
    part_number: str | None = None

    @property
    def is_populated(self) -> bool:
        """True when a module is installed in this slot."""
        return self.size_bytes > 0

    @property
    def size_gb(self) -> float:
        """Module size in gibibytes."""
        return self.size_bytes / 1024**3
