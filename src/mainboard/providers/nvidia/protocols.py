"""Structural contracts for the untyped CUDA runtime, NVML, and `cuda.core` bindings.

`cuda.bindings` and `cuda.core` ship no type stubs, so every call into them is otherwise
opaque. These Protocols pin down exactly the functions and result fields mainboard reads,
turning the provider into fully-typed code over a precise seam instead of `Any`. Each
Protocol lists only what mainboard actually touches; the bindings expose far more.

The CUDA Runtime functions follow the `(error, *values)` tuple convention; the error is an
opaque enum member compared against `cudaError_t.cudaSuccess`, so it is typed `int` here
(it compares and formats like one) and the value fields keep their real shapes.
"""

from collections.abc import Callable
from typing import Protocol

from ...profiling.protocols import RawActivity


class ActivityKind(Protocol):
    """The `cupti.ActivityKind` enum: members are selected by name with `getattr`."""

    RUNTIME: int
    DRIVER: int


class CallbackDomain(Protocol):
    """The `cupti.CallbackDomain` enum members mainboard routes records and callbacks by."""

    RUNTIME_API: int
    DRIVER_API: int


class ApiCallbackSite(Protocol):
    """The `cupti.ApiCallbackSite` enum: distinguishes the API-enter from the API-exit edge."""

    API_ENTER: int


class CallbackData(Protocol):
    """The CUPTI callback payload mainboard reads: which edge fired and the API name."""

    callback_site: int
    function_name: str


class Cupti(Protocol):
    """The `cupti.cupti` functions and enums mainboard's Activity + Callback collectors use.

    `cupti-python` ships no stubs; this pins the asynchronous Activity API surface (register,
    enable/disable, flush) and the synchronous Callback API surface (subscribe, enable_domain)
    that the tracer drives. `subscribe` returns an opaque subscriber token, threaded back into
    `enable_domain`/`unsubscribe` and never inspected.
    """

    ActivityKind: ActivityKind
    CallbackDomain: CallbackDomain
    ApiCallbackSite: ApiCallbackSite

    def get_timestamp(self) -> int: ...
    def get_callback_name(self, domain: int, cbid: int) -> str: ...
    def activity_enable(self, kind: int) -> None: ...
    def activity_disable(self, kind: int) -> None: ...
    def activity_flush_all(self, flag: int) -> None: ...
    def activity_register_callbacks(
        self,
        on_requested: Callable[[], tuple[int, int]],
        on_completed: Callable[[list[RawActivity]], None],
    ) -> None: ...
    def subscribe(
        self, callback: Callable[[None, int, int, CallbackData], None], userdata: None
    ) -> Subscriber: ...
    def unsubscribe(self, subscriber: Subscriber) -> None: ...
    def enable_domain(self, enable: int, subscriber: Subscriber, domain: int) -> None: ...


class Subscriber(Protocol):
    """An opaque CUPTI callback-subscriber token, only threaded back into CUPTI calls."""


class Nvtx(Protocol):
    """The `nvtx` annotation surface mainboard emits: range push/pop and instant marks."""

    def push_range(self, message: str) -> None: ...
    def pop_range(self) -> None: ...
    def mark(self, message: str) -> None: ...


class MemoryInfo(Protocol):
    """A device memory snapshot: total, used, and free bytes."""

    total: int
    used: int
    free: int


class UtilizationRates(Protocol):
    """NVML utilization sample: compute and memory-controller busy percentages."""

    gpu: int
    memory: int


class ProcessSample(Protocol):
    """One NVML compute process: its PID and resident GPU memory in bytes."""

    pid: int
    used_gpu_memory: int


class DeviceProperties(Protocol):
    """`cudaDeviceProp` fields mainboard reads for SM count, clocks, and bandwidth."""

    multiProcessorCount: int  # noqa: N815 — mirrors the CUDA Runtime struct field name
    memoryBusWidth: int  # noqa: N815
    clock_rate: int
    memory_clock_rate: int
    global_memory_bus_width: int


class CudaError(Protocol):
    """The `cudaError_t` enum, used only to read its `cudaSuccess` member."""

    @property
    def cudaSuccess(self) -> int:  # noqa: N802 — mirrors the CUDA enum member name
        ...


class ClockType(Protocol):
    """NVML `ClockType` enum members mainboard selects (SM and memory domains)."""

    CLOCK_SM: int
    CLOCK_MEM: int


class TemperatureSensors(Protocol):
    """NVML `TemperatureSensors` enum: the GPU-die sensor selector."""

    TEMPERATURE_GPU: int


class TemperatureThresholds(Protocol):
    """NVML `TemperatureThresholds` enum: the shutdown and slowdown selectors."""

    TEMPERATURE_THRESHOLD_SHUTDOWN: int
    TEMPERATURE_THRESHOLD_SLOWDOWN: int


class DeviceAttr(Protocol):
    """The `cudaDeviceAttr` enum members mainboard reads to probe memory coherence.

    `cudaDevAttrPageableMemoryAccess` is set when the GPU can read host pageable memory
    directly, and `cudaDevAttrConcurrentManagedAccess` when CPU and GPU may touch managed
    pages concurrently. Both true is the Grace-Hopper / GB10 coherent-pool signature, where
    host RAM is a peer NUMA node of HBM rather than a PCIe copy away.
    """

    cudaDevAttrPageableMemoryAccess: int  # noqa: N815 — mirrors the CUDA enum member name
    cudaDevAttrConcurrentManagedAccess: int  # noqa: N815


class CudaRuntime(Protocol):
    """The `cuda.bindings.runtime` functions and enums mainboard calls.

    Every call returns the `(error, *values)` tuple the CUDA Runtime uses; the error is
    the opaque `cudaError_t` member compared against `cudaError_t.cudaSuccess`.
    """

    cudaError_t: CudaError  # noqa: N815 — the binding exposes the enum under this exact name
    cudaDeviceAttr: DeviceAttr  # noqa: N815 — the coherence-probe attribute enum

    def cudaGetDeviceCount(self) -> tuple[int, int]: ...  # noqa: N802
    def cudaGetDevice(self) -> tuple[int, int]: ...  # noqa: N802
    def cudaSetDevice(self, index: int) -> tuple[int]: ...  # noqa: N802
    def cudaDeviceSynchronize(self) -> tuple[int]: ...  # noqa: N802
    def cudaMemGetInfo(self) -> tuple[int, int, int]: ...  # noqa: N802
    def cudaDriverGetVersion(self) -> tuple[int, int]: ...  # noqa: N802
    def cudaGetDeviceProperties(self, index: int) -> tuple[int, DeviceProperties]: ...  # noqa: N802
    def cudaDeviceGetPCIBusId(self, length: int, index: int) -> tuple[int, bytes]: ...  # noqa: N802
    def cudaDeviceGetAttribute(self, attr: int, index: int) -> tuple[int, int]: ...  # noqa: N802


class Nvml(Protocol):
    """The NVML functions and enums mainboard calls (snake_case `cuda.bindings._nvml`).

    Handles are opaque device tokens threaded back into later calls, so they are typed
    `object` only at the binding boundary is avoided: each is an `NvmlHandle` alias the
    provider never inspects, only passes through.
    """

    ClockType: ClockType
    TemperatureSensors: TemperatureSensors
    TemperatureThresholds: TemperatureThresholds

    def init_v2(self) -> None: ...
    def device_get_handle_by_pci_bus_id_v2(self, pci_bus_id: str) -> NvmlHandle: ...
    def device_get_name(self, handle: NvmlHandle) -> bytes | str: ...
    def device_get_uuid(self, handle: NvmlHandle) -> bytes | str: ...
    def device_get_cuda_compute_capability(self, handle: NvmlHandle) -> tuple[int, int]: ...
    def device_get_memory_info_v2(self, handle: NvmlHandle) -> MemoryInfo: ...
    def device_get_clock_info(self, handle: NvmlHandle, clock: int) -> int: ...
    def device_get_max_clock_info(self, handle: NvmlHandle, clock: int) -> int: ...
    def device_get_utilization_rates(self, handle: NvmlHandle) -> UtilizationRates: ...
    def device_get_current_clocks_event_reasons(self, handle: NvmlHandle) -> int: ...
    def device_get_temperature_v(self, handle: NvmlHandle, sensor: int) -> int: ...
    def device_get_temperature_threshold(self, handle: NvmlHandle, threshold: int) -> int: ...
    def device_get_power_usage(self, handle: NvmlHandle) -> int: ...
    def device_get_total_energy_consumption(self, handle: NvmlHandle) -> int: ...
    def device_get_pcie_throughput(self, handle: NvmlHandle, counter: int) -> int: ...
    def device_get_fan_speed(self, handle: NvmlHandle) -> int: ...
    def device_get_compute_running_processes_v3(
        self, handle: NvmlHandle
    ) -> list[ProcessSample]: ...


class NvmlHandle(Protocol):
    """An opaque NVML device handle, only ever threaded back into NVML calls."""


class Clock(Protocol):
    """A `cuda.core` clock domain handle exposing its current frequency."""

    def get_current_mhz(self) -> float: ...


class ArchToken(Protocol):
    """A `cuda.core` architecture token whose `name` is the readable label, e.g. `ADA`."""

    name: str


class SystemDevice(Protocol):
    """The `cuda.core.system.Device` fields mainboard reads for identity and live sensors."""

    name: bytes | str
    uuid: bytes | str
    cuda_compute_capability: tuple[int, int]
    arch: ArchToken
    memory_info: MemoryInfo
    utilization: UtilizationRates

    def get_clock(self, domain: str) -> Clock: ...


class CoreDeviceProperties(Protocol):
    """`cuda.core.Device.properties` fields mainboard reads for SM count and bandwidth."""

    multiprocessor_count: int
    memory_clock_rate: int
    global_memory_bus_width: int
    clock_rate: int


class CoreDevice(Protocol):
    """The `cuda.core.Device` surface mainboard reads (static device properties)."""

    @property
    def properties(self) -> CoreDeviceProperties: ...


class CoreSystem(Protocol):
    """The `cuda.core.system` module: device factory plus its unsupported-feature error."""

    NotSupportedError: type[Exception]

    def Device(self, index: int) -> SystemDevice: ...  # noqa: N802 — binding's class name


class CoreDeviceType(Protocol):
    """The `cuda.core.Device` class, called with a visible index to build a device."""

    def __call__(self, index: int) -> CoreDevice: ...
