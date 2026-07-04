import re
import subprocess
from functools import cached_property

from .base import FrozenModel

_DMI_PLACEHOLDER = frozenset({"unknown", "not specified", "not provided", "none", ""})
_NO_MODULE_PHRASES = ("No Module", "Not Installed")
_SIZE_UNITS: dict[str, int] = {"GB": 1024**3, "MB": 1024**2, "TB": 1024**4}


class MemoryCard(FrozenModel):
    """One physical DIMM slot parsed from `dmidecode -t 17`.

    section: raw dmidecode Memory Device section text.
    """

    section: str

    @classmethod
    def all(cls) -> tuple[MemoryCard, ...]:
        """Parse `dmidecode -t 17` into MemoryCard objects.

        Returns an empty tuple when dmidecode is absent, not runnable, or
        returns no Memory Device sections (e.g. no root privileges).
        """
        try:
            result = subprocess.run(
                ["dmidecode", "-t", "17"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if result.returncode != 0 and not result.stdout:
                return ()
            sections = [s for s in result.stdout.split("\n\n") if "Memory Device" in s]
            return tuple(cls(section=sec) for sec in sections)
        except FileNotFoundError, subprocess.TimeoutExpired, PermissionError:
            return ()

    @cached_property
    def slot(self) -> str:
        """Locator string, e.g. `DIMM_A1`."""
        return self._field("Locator") or "unknown"

    @cached_property
    def bank(self) -> str:
        """Bank locator string, e.g. `BANK 0`."""
        return self._field("Bank Locator") or ""

    @cached_property
    def _size_str(self) -> str | None:
        """Raw Size field from this section."""
        return self._field("Size")

    @cached_property
    def populated(self) -> bool:
        """False when the slot is empty."""
        return self._size_str is not None and not any(
            p in self._size_str for p in _NO_MODULE_PHRASES
        )

    @cached_property
    def size_bytes(self) -> int:
        """Installed module capacity; 0 on empty slots."""
        return self._parse_size(self._size_str) if self.populated else 0

    @cached_property
    def speed_mhz(self) -> int | None:
        """Rated speed in MT/s; None if not reported."""
        m = re.search(
            r"^\s+(?:Speed|Configured Memory Speed):\s+(\d+)\s+MT/s",
            self.section,
            re.MULTILINE,
        )
        return int(m.group(1)) if m else None

    @cached_property
    def memory_type(self) -> str:
        """Technology string, e.g. `LPDDR5` or `DDR4`."""
        return self._field("Type") or ""

    @cached_property
    def form_factor(self) -> str:
        """Physical form factor, e.g. `DIMM`."""
        return self._field("Form Factor") or ""

    @cached_property
    def manufacturer(self) -> str | None:
        """Module manufacturer; None if not specified."""
        return self._field("Manufacturer")

    @cached_property
    def part_number(self) -> str | None:
        """Module part number; None if not specified."""
        return self._field("Part Number")

    @property
    def size_gb(self) -> float:
        """Module capacity in gibibytes."""
        return self.size_bytes / 1024**3

    def _field(self, field: str) -> str | None:
        """Return the value of one dmidecode field within this section."""
        m = re.search(rf"^\s+{re.escape(field)}:\s+(.+)$", self.section, re.MULTILINE)
        if not m:
            return None
        value = m.group(1).strip()
        return value if value.lower() not in _DMI_PLACEHOLDER else None

    @staticmethod
    def _parse_size(size_str: str | None) -> int:
        """Parse a dmidecode Size field like `512 GB` or `32768 MB` into bytes."""
        if not size_str:
            return 0
        m = re.match(r"(\d+)\s+(GB|MB|TB)", size_str.strip(), re.IGNORECASE)
        if not m:
            return 0
        return int(m.group(1)) * _SIZE_UNITS[m.group(2).upper()]
