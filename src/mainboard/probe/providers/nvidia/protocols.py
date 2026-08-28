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
    """The `cudaDeviceAttr` enum members read to probe memory coherence.

    `cudaDevAttrPageableMemoryAccess` is set when the GPU can read host pageable memory
    directly, and `cudaDevAttrConcurrentManagedAccess` when CPU and GPU may touch managed
    pages concurrently. Both true is the Grace-Hopper / GB10 coherent-pool signature, where
    host RAM is a peer NUMA node of HBM rather than a PCIe copy away.
    """

    cudaDevAttrPageableMemoryAccess: int
    cudaDevAttrConcurrentManagedAccess: int


class CudaRuntime(Protocol):
    """The `cuda.bindings.runtime` functions and enums the provider calls.

    Every call returns the `(error, *values)` tuple the CUDA Runtime uses, where the
    error is the opaque `cudaError_t` member compared against `cudaError_t.cudaSuccess`.
    """

    cudaError_t: CudaError
    cudaDeviceAttr: DeviceAttr

    def cudaDeviceGetAttribute(self, attr: int, index: int) -> tuple[int, int]: ...

    def cudaDeviceGetPCIBusId(self, length: int, index: int) -> tuple[int, bytes]: ...

    def cudaDriverGetVersion(self) -> tuple[int, int]: ...

    def cudaGetDevice(self) -> tuple[int, int]: ...

    def cudaGetDeviceCount(self) -> tuple[int, int]: ...
    def cudaMemGetInfo(self) -> tuple[int, int, int]: ...

    def cudaSetDevice(self, index: int) -> tuple[int]: ...


class ClockDomain(Protocol):
    """The `nvmlClockType_t` enum, read only for the memory-clock domain."""

    CLOCK_MEM: int


class TemperatureSensor(Protocol):
    """The `nvmlTemperatureSensors_t` enum, read only for the die sensor."""

    TEMPERATURE_GPU: int


class ClocksEvent(Protocol):
    """The `nvmlClocksEventReasons` bits that mean the device is really being held back.

    The enum carries benign members too (an idle device, an applied clock setting); only
    the ones that cost real performance are named here, so the provider cannot read a
    healthy device as a throttled one.
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


class Nvml(Protocol):
    """The NVML functions the provider calls (snake_case `cuda.bindings._nvml`).

    Handles are opaque device tokens threaded back into later calls, typed as the
    dedicated `NvmlHandle` protocol below rather than inspected.
    """

    ClockType: ClockDomain
    ClocksEventReasons: ClocksEvent
    TemperatureSensors: TemperatureSensor

    def device_get_compute_running_processes_v3(
        self, handle: NvmlHandle
    ) -> Sequence[ProcessInfo]: ...

    def device_get_cuda_compute_capability(self, handle: NvmlHandle) -> tuple[int, int]: ...

    def device_get_current_clocks_event_reasons(self, handle: NvmlHandle) -> int: ...

    def device_get_handle_by_pci_bus_id_v2(self, pci_bus_id: str) -> NvmlHandle: ...

    def device_get_max_clock_info(self, handle: NvmlHandle, clock: int) -> int: ...

    def device_get_memory_bus_width(self, handle: NvmlHandle) -> int: ...

    def device_get_memory_info_v2(self, handle: NvmlHandle) -> MemoryInfo: ...

    def device_get_name(self, handle: NvmlHandle) -> bytes | str: ...

    def device_get_power_usage(self, handle: NvmlHandle) -> int: ...

    def device_get_temperature_v(self, handle: NvmlHandle, sensor: int) -> int: ...

    def device_get_utilization_rates(self, handle: NvmlHandle) -> UtilizationInfo: ...

    def device_get_uuid(self, handle: NvmlHandle) -> bytes | str: ...

    def init_v2(self) -> None: ...


class NvmlHandle(Protocol):
    """An opaque NVML device handle, only ever threaded back into NVML calls."""


class ArchToken(Protocol):
    """A `cuda.core` architecture token whose `name` is the readable label, e.g. `ADA`."""

    name: str


class SystemDevice(Protocol):
    """The `cuda.core.system.Device` fields the provider reads for identity and memory."""

    name: bytes | str
    uuid: bytes | str
    cuda_compute_capability: tuple[int, int]
    arch: ArchToken
    memory_info: MemoryInfo
    utilization: UtilizationInfo


class CoreSystem(Protocol):
    """The `cuda.core.system` module, a device factory plus its unsupported-feature error."""

    NotSupportedError: type[Exception]

    def Device(self, index: int) -> SystemDevice: ...


class CoreDevice(Protocol):
    """An opaque `cuda.core.Device` instance, only used to gate the optional layer."""


class CoreDeviceType(Protocol):
    """The `cuda.core.Device` class, called with a visible index to build a device."""

    def __call__(self, index: int) -> CoreDevice: ...
