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


class Nvml(Protocol):
    """The NVML functions the provider calls (snake_case `cuda.bindings._nvml`).

    Handles are opaque device tokens threaded back into later calls, typed as the
    dedicated `NvmlHandle` protocol below rather than inspected.
    """

    def device_get_cuda_compute_capability(self, handle: NvmlHandle) -> tuple[int, int]: ...

    def device_get_handle_by_pci_bus_id_v2(self, pci_bus_id: str) -> NvmlHandle: ...

    def device_get_memory_info_v2(self, handle: NvmlHandle) -> MemoryInfo: ...

    def device_get_name(self, handle: NvmlHandle) -> bytes | str: ...

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
