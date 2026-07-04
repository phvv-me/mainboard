import time

from ..enums import UnitKind, Vendor
from .base import Field, FrozenModel
from .clock import Clock
from .energy_reading import EnergyReading
from .memory import Memory
from .thermal_state import ThermalState
from .utilization import Utilization


class UnitSnapshot(FrozenModel):
    """Point-in-time neutral telemetry for one unit.

    name: caller-provided label, e.g. a profiling region.
    unit_name: human-readable unit name.
    kind: unit category.
    vendor: hardware vendor.
    timestamp_ns: monotonic timestamp set at construction.
    clocks: clock readings by hardware domain.
    memory: memory visible to the unit.
    utilization: normalized utilization when available.
    energy: instantaneous power and cumulative energy when available.
    thermal: thermal reading when available.
    """

    name: str = ""
    unit_name: str = ""
    kind: UnitKind = Field(default=UnitKind.UNKNOWN)
    vendor: Vendor = Field(default=Vendor.UNKNOWN)
    timestamp_ns: int = Field(default_factory=time.perf_counter_ns)
    clocks: tuple[Clock, ...] = ()
    memory: Memory = Memory()
    utilization: Utilization = Utilization()
    energy: EnergyReading = EnergyReading()
    thermal: ThermalState = ThermalState()
