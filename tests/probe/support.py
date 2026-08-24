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

    def cudaDriverGetVersion(self) -> tuple[int, int]:
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


class FakeNvml:
    """Minimal `cuda.bindings.nvml` surface used by the provider."""

    NotSupportedError = FakeError
    NoPermissionError = FakeError
    UnknownError = FakeError
    GpuIsLostError = FakeError

    def device_get_cuda_compute_capability(self, handle: str) -> tuple[int, int]:
        return (8, 9)

    def device_get_handle_by_pci_bus_id_v2(self, bus_id: str) -> str:
        return f"handle:{bus_id}"

    def device_get_memory_info_v2(self, handle: str) -> FakeMemoryInfo:
        return FakeMemoryInfo(24 * 1024**3, used=6 * 1024**3, free=18 * 1024**3)

    def device_get_name(self, handle: str) -> str:
        return "NVIDIA GeForce RTX 4090"

    def device_get_utilization_rates(self, handle: str) -> FakeUtilizationReading:
        return FakeUtilizationReading(gpu=48, memory=22)

    def device_get_uuid(self, handle: str) -> str:
        return "GPU-deadbeef"

    def init_v2(self) -> None:
        return None


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
        self.nvml = FakeNvml()
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
