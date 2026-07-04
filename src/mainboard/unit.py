from functools import cached_property
from typing import ClassVar

from pydantic import Field

from .enums import UnitKind, Vendor
from .models.base import FrozenModel
from .models.clock import Clock
from .models.energy_reading import EnergyReading
from .models.memory import Memory
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

    @property
    def memory(self) -> Memory:
        """Memory visible to this unit."""
        return Memory()

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """Clock readings grouped by hardware domain."""
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
            memory=self.memory,
            utilization=self.utilization,
            energy=self.energy,
            thermal=self.thermal,
        )
