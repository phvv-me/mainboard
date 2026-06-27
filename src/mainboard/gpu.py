from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

from patos import Registry
from pydantic import Field

from .enums import UnitKind, Vendor
from .models.clock import Clock
from .models.clock_info import ClockInfo
from .models.energy_reading import EnergyReading
from .models.gpu_snapshot import GPUSnapshot
from .models.memory import Memory
from .models.pcie_info import PcieInfo
from .models.thermal_state import ThermalState
from .models.utilization import Utilization
from .unit import Unit

if TYPE_CHECKING:
    from .models.process_info import ProcessInfo

logger = logging.getLogger(__name__)


class GPU(Unit, Registry):
    """GPU with telemetry and legacy profiling sensor accessors.

    Registry root: concrete vendor providers self-register on import, and
    `all` fans out over them, concatenating each provider's own probe.
    """

    index: int = 0
    kind: ClassVar[UnitKind] = UnitKind.GPU
    vendor: Vendor = Field(default=Vendor.UNKNOWN)
    backend: str = "none"

    @classmethod
    def all(cls) -> tuple[GPU, ...]:
        """Return GPUs visible across every registered provider.

        Probing is best-effort per provider: a backend whose `all` raises (a
        binding that loads but then throws, an unexpected NVML error) is logged
        and skipped so one broken vendor never sinks the whole machine probe.
        """
        return tuple(gpu for provider in cls.implementations() for gpu in cls.probe(provider))

    @classmethod
    def probe(cls, provider: type[GPU]) -> tuple[GPU, ...]:
        """One provider's devices, or an empty tuple when its probe fails."""
        try:
            return tuple(provider.all())
        except Exception:
            logger.warning(
                "GPU provider %s failed to probe; skipping", provider.__name__, exc_info=True
            )
            return ()

    @cached_property
    def name(self) -> str:
        """Human-readable GPU name."""
        return "unknown"

    @cached_property
    def uuid(self) -> str:
        """Stable GPU identifier when the provider exposes one."""
        return ""

    @cached_property
    def architecture(self) -> str:
        """Human-readable architecture or generation name."""
        return "unknown"

    @cached_property
    def arch_key(self) -> str:
        """A stable, machine-friendly architecture id for per-arch dispatch.

        The key to look this device up in an arch-keyed table (see
        `mainboard.profiling.arch_config`): vendor backends return a precise,
        dot-free target such as `sm_90` (NVIDIA) so tile sizes and kernel
        configs can be pinned per generation. The base falls back to the
        lowercased human architecture name.
        """
        return self.architecture.lower()

    @cached_property
    def peak_bandwidth_gbs(self) -> float:
        """Theoretical peak memory bandwidth in GB/s when known."""
        return 0.0

    @cached_property
    def driver_version(self) -> tuple[int, int] | None:
        """Driver or runtime version as `(major, minor)` when known."""
        return None

    @property
    def memory(self) -> Memory:
        """Current accelerator memory state."""
        return Memory(scope="device", source=self.backend, supported=False)

    @property
    def clocks(self) -> ClockInfo:
        """Current compute and memory clocks."""
        return ClockInfo()

    @property
    def clock_readings(self) -> tuple[Clock, ...]:
        """Clock readings grouped by hardware domain."""
        clocks = self.clocks
        return (
            Clock(
                domain="gpu_compute",
                current_mhz=clocks.sm_mhz or None,
                source=self.backend,
                supported=clocks.sm_mhz > 0,
            ),
            Clock(
                domain="memory",
                current_mhz=clocks.memory_mhz or None,
                source=self.backend,
                supported=clocks.memory_mhz > 0,
            ),
        )

    @property
    def utilization(self) -> Utilization:
        """Current compute and memory-controller utilization."""
        return Utilization()

    @property
    def thermal(self) -> ThermalState:
        """Current thermal state."""
        return ThermalState()

    @property
    def energy(self) -> EnergyReading:
        """Current power and cumulative energy reading."""
        return EnergyReading()

    @property
    def pcie(self) -> PcieInfo:
        """Current host interconnect throughput."""
        return PcieInfo()

    @property
    def fan_speed_pct(self) -> int:
        """Fan speed percentage, or zero when unavailable."""
        return 0

    @property
    def temperature_c(self) -> int:
        """Current GPU temperature in Celsius."""
        return self.thermal.temperature_c

    @property
    def gpu_util_pct(self) -> int:
        """Current compute utilization percentage."""
        return self.utilization.gpu_pct

    @property
    def processes(self) -> list[ProcessInfo]:
        """Processes using this accelerator."""
        return []

    def snapshot(self, name: str = "") -> GPUSnapshot:
        """Point-in-time reading of all common sensor properties.

        name: profiling region label to embed in the snapshot.
        """
        return GPUSnapshot(
            name=name,
            unit_name=self.name,
            kind=self.kind,
            vendor=self.vendor,
            memory=self.memory,
            clocks=self.clock_readings,
            utilization=self.utilization,
            energy=self.energy,
            thermal=self.thermal,
            pcie=self.pcie,
            fan_speed_pct=self.fan_speed_pct,
            processes=self.processes,
        )
