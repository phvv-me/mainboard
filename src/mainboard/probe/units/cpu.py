from functools import cached_property
from typing import ClassVar

from ..enums import UnitKind, Vendor
from ..facts.memory import Memory
from .unit import Unit


class CPU(Unit):
    """Host CPU package or SoC CPU cluster."""

    name_value: str
    architecture_value: str
    logical_cores: int = 0
    physical_cores: int = 0
    current_clock_mhz: float | None = None
    vendor: Vendor = Vendor.UNKNOWN
    kind: ClassVar[UnitKind] = UnitKind.CPU
    backend: str = "os"

    @cached_property
    def architecture(self) -> str:
        """CPU architecture string."""
        return self.architecture_value

    @cached_property
    def label(self) -> str:
        """CPU model name."""
        return self.name_value

    @property
    def memory(self) -> Memory:
        """System memory visible to the CPU."""
        return Memory.system()
