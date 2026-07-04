from functools import cached_property
from typing import ClassVar

from pydantic import Field

from .enums import UnitKind, Vendor
from .models.clock import Clock
from .models.memory import Memory
from .unit import Unit


class CPU(Unit):
    """Host CPU package or SoC CPU cluster."""

    name_value: str
    architecture_value: str
    logical_cores: int = 0
    physical_cores: int = 0
    current_clock_mhz: float | None = None
    vendor: Vendor = Field(default=Vendor.UNKNOWN)
    kind: ClassVar[UnitKind] = UnitKind.CPU
    backend: str = "os"

    @cached_property
    def name(self) -> str:
        """CPU model name."""
        return self.name_value

    @cached_property
    def architecture(self) -> str:
        """CPU architecture string."""
        return self.architecture_value

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """CPU clock readings from the OS."""
        return (
            Clock(
                domain="cpu",
                current_mhz=self.current_clock_mhz,
                source="psutil",
                supported=self.current_clock_mhz is not None,
            ),
        )

    @property
    def memory(self) -> Memory:
        """System memory visible to the CPU."""
        return Memory.system()
