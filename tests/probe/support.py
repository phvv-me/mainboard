from typing import NoReturn, Protocol


class FakeError(Exception):
    """Shared NVML/system error type for the fake CUDA stack."""


def raise_unsupported(*args, **kwargs) -> NoReturn:
    """Raise the fake NVML `NotSupportedError`, modelling a sensor a device lacks."""
    raise FakeError


class CudaErrorT:
    """Mimic `cudaError_t` with a single success sentinel."""

    cudaSuccess = 0


class CudaDeviceAttr:
    """Mimic `cudaDeviceAttr` with the two coherence-probe members read here."""

    cudaDevAttrPageableMemoryAccess = 24
    cudaDevAttrConcurrentManagedAccess = 89


class FakeRuntime:
    """Minimal `cuda.bindings.runtime` returning two visible devices.

    `coherent` toggles the two `cudaDeviceGetAttribute` coherence flags so a test can
    model a discrete card (the default 4090, both flags off) or a Grace-Hopper-style
    coherent pool (both flags on, the `unified=True` signal).
    """

    cudaError_t = CudaErrorT
    cudaDeviceAttr = CudaDeviceAttr

    def __init__(self, count: int = 2, coherent: bool = False) -> None:
        self.count = count
        self.coherent = coherent

    def cudaDeviceGetAttribute(self, attr: int, index: int) -> tuple[int, int]:
        return (CudaErrorT.cudaSuccess, 1 if self.coherent else 0)

    def cudaDeviceGetPCIBusId(self, length: int, index: int) -> tuple[int, bytes]:
        return (CudaErrorT.cudaSuccess, f"0000:0{index}:00.0\x00".encode())

    def cudaRuntimeGetVersion(self) -> tuple[int, int]:
        return (CudaErrorT.cudaSuccess, 13010)

    def cudaGetDevice(self) -> tuple[int, int]:
        return (CudaErrorT.cudaSuccess, 0)

    def cudaGetDeviceCount(self) -> tuple[int, int]:
        return (CudaErrorT.cudaSuccess, self.count)

    def cudaMemGetInfo(self) -> tuple[int, int, int]:
        return (CudaErrorT.cudaSuccess, 8 * 1024**3, 24 * 1024**3)

    def cudaSetDevice(self, index: int) -> None:
        return None


class MemoryInfo(Protocol):
    total: int
    used: int
    free: int


class FakeMemoryInfo:
    def __init__(self, total: int, *, used: int, free: int) -> None:
        self.total = total
        self.used = used
        self.free = free


class FakeSystemDevice:
    """Mimic `cuda.core.system.Device` NVML-backed reads."""

    name = b"NVIDIA GeForce RTX 4090"
    uuid = "GPU-deadbeef"
    cuda_compute_capability = (8, 9)
    arch = type("Arch", (), {"name": "ADA"})()

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.memory_info = FakeMemoryInfo(24 * 1024**3, used=6 * 1024**3, free=18 * 1024**3)
        self.utilization = FakeUtilizationReading(gpu=61, memory=37)


class FakeUtilizationReading:
    """A gpu/memory percentage pair as both device layers shape it."""

    def __init__(self, *, gpu: int, memory: int) -> None:
        self.gpu = gpu
        self.memory = memory


class FakeClockType:
    """Mimic `nvmlClockType_t`, read only for the memory-clock domain."""

    CLOCK_MEM = 2


class FakeTemperatureSensors:
    """Mimic `nvmlTemperatureSensors_t`, read only for the die sensor."""

    TEMPERATURE_GPU = 0


class FakeClocksEventReasons:
    """Mimic the `nvmlClocksEventReasons` bitmask, benign bits included.

    `EVENT_REASON_GPU_IDLE` is the one every idle device sets and the provider must ignore,
    so it is here even though nothing reads it by name.
    """

    EVENT_REASON_GPU_IDLE = 0x1
    EVENT_REASON_APPLICATIONS_CLOCKS_SETTING = 0x2
    EVENT_REASON_SW_POWER_CAP = 0x4
    THROTTLE_REASON_HW_SLOWDOWN = 0x8
    EVENT_REASON_SYNC_BOOST = 0x10
    EVENT_REASON_SW_THERMAL_SLOWDOWN = 0x20
    THROTTLE_REASON_HW_THERMAL_SLOWDOWN = 0x40
    THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN = 0x80


class FakeProcessInfo:
    """One NVML compute-context entry: the process and what it holds on the device."""

    def __init__(self, *, pid: int, used_gpu_memory: int) -> None:
        self.pid = pid
        self.used_gpu_memory = used_gpu_memory


class FakeDriverModel:
    """NVML's Windows driver-model enum."""

    DRIVER_WDDM = 0
    DRIVER_WDM = 1
    DRIVER_MCDM = 2


class FakePciInfo:
    """PCI identity in the normalized snake-case shape Mainboard consumes."""

    def __init__(self, bus_id: str) -> None:
        self.bus_id = bus_id


class FakeNvml:
    """Minimal `cuda.bindings.nvml` surface used by the provider.

    The sensor readings are a 4090 at idle: 1008 GB/s of peak bandwidth from a 384-bit bus
    at 10501 MHz, 17.6 W, 42 C, and a clocks-event mask carrying only the benign idle bit.
    """

    NotSupportedError = FakeError
    NoPermissionError = FakeError
    UnknownError = FakeError
    GpuIsLostError = FakeError
    ClockType = FakeClockType
    ClocksEventReasons = FakeClocksEventReasons
    DriverModel = FakeDriverModel
    TemperatureSensors = FakeTemperatureSensors

    def __init__(
        self,
        clocks_event_reasons: int = FakeClocksEventReasons.EVENT_REASON_GPU_IDLE,
        device_count: int = 2,
    ) -> None:
        self.clocks_event_reasons = clocks_event_reasons
        self.count = device_count

    def device_get_compute_running_processes_v3(self, handle: str) -> tuple[FakeProcessInfo, ...]:
        return (FakeProcessInfo(pid=4242, used_gpu_memory=2 * 1024**3),)

    def device_get_count_v2(self) -> int:
        return self.count

    def device_get_cuda_compute_capability(self, handle: str) -> tuple[int, int]:
        return (8, 9)

    def device_get_driver_model_v2(self, handle: str) -> tuple[int, int]:
        return (self.DriverModel.DRIVER_WDM, self.DriverModel.DRIVER_WDM)

    def device_get_current_clocks_event_reasons(self, handle: str) -> int:
        return self.clocks_event_reasons

    def device_get_handle_by_pci_bus_id_v2(self, bus_id: str) -> str:
        return f"handle:{bus_id}"

    def device_get_handle_by_index_v2(self, index: int) -> str:
        return f"handle:0000:0{index}:00.0"

    def device_get_max_clock_info(self, handle: str, clock: int) -> int:
        return 10501

    def device_get_memory_bus_width(self, handle: str) -> int:
        return 384

    def device_get_memory_info_v2(self, handle: str) -> FakeMemoryInfo:
        return FakeMemoryInfo(24 * 1024**3, used=6 * 1024**3, free=18 * 1024**3)

    def device_get_name(self, handle: str) -> str:
        return "NVIDIA GeForce RTX 4090"

    def device_get_power_usage(self, handle: str) -> int:
        return 17_647

    def device_get_pci_info_v3(self, handle: str) -> FakePciInfo:
        return FakePciInfo(handle.removeprefix("handle:"))

    def device_get_temperature_v(self, handle: str, sensor: int) -> int:
        return 42

    def device_get_utilization_rates(self, handle: str) -> FakeUtilizationReading:
        return FakeUtilizationReading(gpu=48, memory=22)

    def device_get_uuid(self, handle: str) -> str:
        return "GPU-deadbeef"

    def init_v2(self) -> None:
        return None

    def system_get_driver_version(self) -> str:
        return "580.65.06"


class FakeSystem:
    """Mimic `cuda.core.system` module: Device factory plus NotSupportedError."""

    NotSupportedError = FakeError

    def __init__(self) -> None:
        self.Device = FakeSystemDevice


class FakeNvidiaApis:
    """Drop-in replacement for `NvidiaApis` wired to the fakes above.

    `has_cuda_core=False` drops `cuda.core` to exercise the NVML/runtime-only
    paths the provider uses on hosts where the optional layer fails to load.
    """

    def __init__(
        self, device_count: int = 2, *, has_cuda_core: bool = True, coherent: bool = False
    ) -> None:
        self.runtime = FakeRuntime(device_count, coherent=coherent)
        self.system = FakeSystem() if has_cuda_core else None
        self.nvml = FakeNvml(device_count=device_count)
        self.cuda_device_type = (lambda index: object()) if has_cuda_core else None
        self.nvml_errors = (FakeError,)

    @property
    def has_cuda_core(self) -> bool:
        """Whether the optional `cuda.core` layer is wired in this fake."""
        return self.cuda_device_type is not None


class FakeSensorlessDevice:
    """A `cuda.core` device whose memory sensor refuses, the shape a device without it reports."""

    memory_info = property(raise_unsupported)


class InstallNvidiaStack(Protocol):
    """Install a fake CUDA/NVML stack for this test and hand it back."""

    def __call__(
        self, *, device_count: int = 2, has_cuda_core: bool = True, coherent: bool = False
    ) -> FakeNvidiaApis: ...
