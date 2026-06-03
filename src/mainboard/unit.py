from __future__ import annotations

from functools import cached_property
from typing import ClassVar

import psutil
from pydantic import Field

from .enums import UnitKind, Vendor
from .models.base import FrozenModel
from .models.clock import Clock
from .models.energy_reading import EnergyReading
from .models.memory_usage import MemoryUsage
from .models.thermal_state import ThermalState
from .models.unit_snapshot import UnitSnapshot
from .models.utilization import Utilization


class Unit(FrozenModel):
    """Schedulable hardware execution resource.

    A unit can be a CPU package or cluster, GPU, NPU, DSP, or other hardware
    engine that executes work over memory.
    """

    index: int = 0
    kind: ClassVar[UnitKind] = UnitKind.UNKNOWN
    vendor: Vendor = Field(default=Vendor.UNKNOWN)
    backend: str = "none"

    @cached_property
    def name(self) -> str:
        """Human-readable unit name."""
        return "unknown"

    @cached_property
    def architecture(self) -> str:
        """Human-readable architecture or generation."""
        return "unknown"

    @cached_property
    def total_memory_bytes(self) -> int:
        """Total memory most relevant to this unit."""
        return 0

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """Clock readings grouped by hardware domain."""
        return ()

    @property
    def memory_readings(self) -> tuple[MemoryUsage, ...]:
        """Memory regions visible to this unit."""
        return ()

    @property
    def utilization(self) -> Utilization:
        """Normalized utilization where available."""
        return Utilization()

    @property
    def energy(self) -> EnergyReading:
        """Power and cumulative energy where available."""
        return EnergyReading()

    @property
    def thermal(self) -> ThermalState:
        """Thermal state where available."""
        return ThermalState()

    def snapshot(self, name: str = "") -> UnitSnapshot:
        """Capture neutral telemetry for this unit."""
        return UnitSnapshot(
            name=name,
            unit_name=self.name,
            kind=self.kind,
            vendor=self.vendor,
            clocks=self.clock_readings,
            memory=self.memory_readings,
            utilization=self.utilization,
            energy=self.energy,
            thermal=self.thermal,
        )


class CPU(Unit):
    """Host CPU package or SoC CPU cluster."""

    name_value: str
    architecture_value: str
    logical_cores: int = 0
    physical_cores: int = 0
    total_memory_value_bytes: int = 0
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

    @cached_property
    def total_memory_bytes(self) -> int:
        """System memory visible to the CPU."""
        return self.total_memory_value_bytes

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
    def memory_readings(self) -> tuple[MemoryUsage, ...]:
        """System memory visible to the CPU."""
        vm = psutil.virtual_memory()
        return (
            MemoryUsage(
                scope="system",
                total_bytes=vm.total,
                used_bytes=vm.used,
                free_bytes=vm.available,
                source="psutil",
            ),
        )
