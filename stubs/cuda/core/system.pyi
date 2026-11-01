# Stub for `cuda.core.system` (cuda-core 1.2): the NVML device view the NVIDIA probe reads.
# The wheel merges `cu12/` and `cu13/` at import, with no `py.typed`, so no checker resolves it.

from collections.abc import Iterable

from cuda.bindings.nvml import NotSupportedError as NotSupportedError

class DeviceArch:
    name: str

class MemoryInfo:
    free: int
    total: int
    used: int

class Utilization:
    gpu: int
    memory: int

class Device:
    @classmethod
    def get_all_devices(cls) -> Iterable[Device]: ...
    @property
    def arch(self) -> DeviceArch: ...
    @property
    def cuda_compute_capability(self) -> tuple[int, int]: ...
    @property
    def memory_info(self) -> MemoryInfo: ...
    @property
    def name(self) -> str: ...
    @property
    def pci_bus_id(self) -> str: ...
    @property
    def utilization(self) -> Utilization: ...
    @property
    def uuid(self) -> str: ...
    def __init__(self, *, index: int) -> None: ...
