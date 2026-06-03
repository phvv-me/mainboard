from __future__ import annotations

import time

from ..enums import UnitKind, Vendor
from .base import Field, FrozenModel
from .board import Board
from .environment import Environment
from .gpu_snapshot import GPUSnapshot
from .memory_usage import MemoryUsage
from .unit_snapshot import UnitSnapshot


class CpuSnapshot(FrozenModel):
    """Host CPU identity and capacity.

    name: CPU model name.
    architecture: CPU architecture string, e.g. `arm64` or `x86_64`.
    vendor: CPU core vendor.
    logical_cores: logical CPU threads including hyperthreading.
    physical_cores: physical CPU cores.
    total_memory_bytes: system memory visible to the CPU.
    current_clock_mhz: current CPU frequency in MHz when the platform reports it.
    """

    name: str = ""
    architecture: str = ""
    vendor: Vendor = Field(default=Vendor.UNKNOWN)
    logical_cores: int = 0
    physical_cores: int = 0
    total_memory_bytes: int = 0
    current_clock_mhz: float | None = None


class MachineSnapshot(FrozenModel):
    """One-call JSON-serializable probe of a host's compute resources.

    timestamp_ns: monotonic timestamp set at construction.
    hostname: network name of the probed host.
    cpu: host CPU identity and capacity.
    memory: system RAM usage at probe time.
    gpus: detected GPUs with per-device telemetry, empty when none are present.
    npus: detected neural processing units, empty when none are present.
    environment: the user, group(s), and job scheduler available on the host.
    board: the host's motherboard and firmware identity.
    """

    timestamp_ns: int = Field(default_factory=time.perf_counter_ns)
    hostname: str = ""
    cpu: CpuSnapshot = CpuSnapshot()
    memory: MemoryUsage = MemoryUsage(scope="system")
    environment: Environment = Environment()
    board: Board = Board()
    gpus: tuple[GPUSnapshot, ...] = ()
    npus: tuple[UnitSnapshot, ...] = ()

    @property
    def unit_count(self) -> int:
        """Total schedulable units: CPU plus every GPU and NPU."""
        return 1 + len(self.gpus) + len(self.npus)

    @property
    def kinds(self) -> tuple[UnitKind, ...]:
        """Distinct unit kinds present on the host."""
        present = {UnitKind.CPU}
        present.update(UnitKind.GPU for _ in self.gpus)
        present.update(UnitKind.NPU for _ in self.npus)
        return tuple(present)
