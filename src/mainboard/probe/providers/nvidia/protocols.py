from collections.abc import Sequence
from typing import Protocol


class MemoryInfo(Protocol):
    """A device memory snapshot carrying total, used, and free bytes."""

    total: int
    used: int
    free: int


class UtilizationInfo(Protocol):
    """GPU and memory-controller activity percentages."""

    gpu: int
    memory: int


class CudaError(Protocol):
    """The `cudaError_t` enum, used only to read its `cudaSuccess` member."""

    @property
    def cudaSuccess(self) -> int: ...


class DeviceAttr(Protocol):
    """The `cudaDeviceAttr` members read to probe memory coherence (see `NvidiaGPU.coherent`).

    Pageable access: the GPU reads host pageable memory directly. Concurrent managed access: CPU
    and GPU may touch managed pages at once. Host-native atomics: device atomics on host memory
    are native, which only a coherent fabric (GH200, GB10) gives.
    """

    cudaDevAttrPageableMemoryAccess: int
    cudaDevAttrConcurrentManagedAccess: int
    cudaDevAttrHostNativeAtomicSupported: int


class CudaRuntime(Protocol):
    """The `cuda.bindings.runtime` functions and enums the provider calls.

    Every call returns `(error, *values)`, the error compared against `cudaError_t.cudaSuccess`.
    """

    cudaError_t: CudaError
    cudaDeviceAttr: DeviceAttr

    def cudaDeviceGetAttribute(self, attr: int, index: int) -> tuple[int, int]: ...
    def cudaDeviceGetPCIBusId(self, length: int, index: int) -> tuple[int, bytes]: ...
    def cudaRuntimeGetVersion(self) -> tuple[int, int]: ...
    def cudaGetDevice(self) -> tuple[int, int]: ...
    def cudaGetDeviceCount(self) -> tuple[int, int]: ...
    def cudaMemGetInfo(self) -> tuple[int, int, int]: ...
    def cudaSetDevice(self, index: int) -> tuple[int]: ...


class ClockDomain(Protocol):
    """The `nvmlClockType_t` enum, read for the SM and memory clock domains."""

    CLOCK_SM: int
    CLOCK_MEM: int


class TemperatureSensor(Protocol):
    """The `nvmlTemperatureSensors_t` enum, read only for the die sensor."""

    TEMPERATURE_GPU: int


class DriverModel(Protocol):
    """NVIDIA Windows driver models relevant to process attribution."""

    DRIVER_WDDM: int
    DRIVER_WDM: int
    DRIVER_MCDM: int


class ClocksEvent(Protocol):
    """The `nvmlClocksEventReasons` bits that cost real performance.

    The benign members (an idle device, an applied clock setting) are left out, so the provider
    cannot read a healthy device as a throttled one.
    """

    EVENT_REASON_SW_POWER_CAP: int
    EVENT_REASON_SW_THERMAL_SLOWDOWN: int
    EVENT_REASON_SYNC_BOOST: int
    THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN: int
    THROTTLE_REASON_HW_SLOWDOWN: int
    THROTTLE_REASON_HW_THERMAL_SLOWDOWN: int


class ProcessInfo(Protocol):
    """One NVML compute-context entry: the process and what it holds on the device."""

    pid: int
    used_gpu_memory: int


class PciInfo(Protocol):
    """PCI identity returned by NVML for one physical device."""

    bus_id: bytes | str


class Nvml(Protocol):
    """The NVML functions the provider calls (snake_case `cuda.bindings._nvml`)."""

    ClockType: ClockDomain
    ClocksEventReasons: ClocksEvent
    DriverModel: DriverModel
    TemperatureSensors: TemperatureSensor

    def device_get_count_v2(self) -> int: ...
    def device_get_compute_running_processes_v3(
        self, handle: NvmlHandle
    ) -> Sequence[ProcessInfo]: ...

    def device_get_cuda_compute_capability(self, handle: NvmlHandle) -> tuple[int, int]: ...
    def device_get_driver_model_v2(self, handle: NvmlHandle) -> tuple[int, int]: ...
    def device_get_current_clocks_event_reasons(self, handle: NvmlHandle) -> int: ...
    def device_get_handle_by_pci_bus_id_v2(self, pci_bus_id: str) -> NvmlHandle: ...
    def device_get_handle_by_index_v2(self, index: int) -> NvmlHandle: ...
    def device_get_handle_by_uuid(self, uuid: bytes) -> NvmlHandle: ...
    def device_get_max_clock_info(self, handle: NvmlHandle, clock: int) -> int: ...
    def device_get_memory_bus_width(self, handle: NvmlHandle) -> int: ...
    def device_get_memory_info_v2(self, handle: NvmlHandle) -> MemoryInfo: ...
    def device_get_name(self, handle: NvmlHandle) -> bytes | str: ...
    def device_get_power_usage(self, handle: NvmlHandle) -> int: ...
    def device_get_pci_info_v3(self, handle: NvmlHandle) -> PciInfo: ...
    def device_get_temperature_v(self, handle: NvmlHandle, sensor: int) -> int: ...
    def device_get_utilization_rates(self, handle: NvmlHandle) -> UtilizationInfo: ...
    def device_get_uuid(self, handle: NvmlHandle) -> bytes | str: ...
    def init_v2(self) -> None: ...
    def system_get_driver_version(self) -> bytes | str: ...


class NvmlHandle(Protocol):
    """An opaque NVML device handle, only ever threaded back into NVML calls."""


class ArchToken(Protocol):
    """A `cuda.core` architecture token whose `name` is the readable label, e.g. `ADA`."""

    name: str


class SystemDevice(Protocol):
    """The `cuda.core.system.Device` fields the provider reads for identity and memory."""

    name: bytes | str
    uuid: bytes | str
    index: int
    pci_bus_id: bytes | str
    cuda_compute_capability: tuple[int, int]
    arch: ArchToken
    memory_info: MemoryInfo
    utilization: UtilizationInfo


class SystemDeviceType(Protocol):
    """The `cuda.core.system.Device` class, built by physical index; its enumeration of every
    physical device ignores `CUDA_VISIBLE_DEVICES`."""

    def __call__(self, *, index: int) -> SystemDevice: ...
    def get_all_devices(self) -> Sequence[SystemDevice]: ...


class CoreSystem(Protocol):
    """The `cuda.core.system` module, a device factory plus its unsupported-feature error."""

    NotSupportedError: type[Exception]
    Device: SystemDeviceType


class CoreDevice(Protocol):
    """An opaque `cuda.core.Device` instance, only used to gate the optional layer."""


class CoreDeviceType(Protocol):
    """The `cuda.core.Device` class, called with a visible index to build a device."""

    def __call__(self, index: int) -> CoreDevice: ...
