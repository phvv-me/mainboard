from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import strategies as st

from mainboard import shell
from mainboard.providers import apple
from mainboard.providers.apple import gpu as apple_gpu
from mainboard.providers.apple import npu as apple_npu
from mainboard.providers.apple import profile as apple_profile
from mainboard.providers.nvidia import apis as nvidia_apis_module

APPLE_PROFILE: dict[str, Any] = {
    "SPHardwareDataType": [
        {
            "chip_type": "Apple M4 Pro",
            "machine_model": "Mac16,8",
            "platform_UUID": "00000000-1111-2222-3333-444455556666",
            "physical_memory": "48 GB",
        }
    ],
    "SPDisplaysDataType": [
        {
            "_name": "Apple M4 Pro",
            "sppci_device_type": "spdisplays_gpu",
            "sppci_model": "Apple M4 Pro",
            "sppci_cores": "20",
            "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
            "spdisplays_vendor": "sppci_vendor_Apple",
        },
        {"sppci_device_type": "spdisplays_display"},
    ],
}


def reset_apple_caches() -> None:
    """Drop every cached macOS profiler result so a fresh fixture takes effect."""
    shell.system_profiler.cache_clear()
    apple.AppleGPU.gpu_records.cache_clear()


def reset_nvidia_cache() -> None:
    """Drop the cached CUDA/NVML import stack."""
    nvidia_apis_module.nvidia_apis.cache_clear()


def reset_machine_singleton() -> None:
    """Drop the cached `Machine` so each test builds a fresh, isolated instance."""
    from mainboard.machine import Machine

    # SingletonMeta caches the instance on the class itself, so dropping the
    # class attribute is the reset.
    Machine.__dict__.get("singleton_instance") and delattr(Machine, "singleton_instance")


@pytest.fixture(autouse=True)
def reset_global_caches() -> Iterator[None]:
    """Keep tests hermetic by clearing every module-level cache around each test."""
    reset_machine_singleton()
    reset_nvidia_cache()
    reset_apple_caches()
    yield
    reset_machine_singleton()
    reset_nvidia_cache()


@pytest.fixture(autouse=True)
def isolate_unit_registries() -> Iterator[None]:
    """Undo any `Unit` subclass a test defines, since `Registry.__init_subclass__`
    appends it to the global GPU/NPU root list and an inherited `all` would then
    recurse forever once the test's monkeypatches are torn down."""
    from mainboard.gpu import GPU
    from mainboard.npu import NPU

    # `registry()` returns the nearest root's live list; snapshot a copy, restore in place.
    saved_gpu = list(GPU.registry())
    saved_npu = list(NPU.registry())
    yield
    GPU.registry()[:] = saved_gpu
    NPU.registry()[:] = saved_npu


@pytest.fixture
def apple_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend the host is an Apple Silicon Mac with the canonical profiler payload."""
    reset_apple_caches()
    monkeypatch.setattr(apple_gpu.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(apple_gpu.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(apple_npu.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(apple_npu.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(apple_profile, "apple_system_profile", lambda: APPLE_PROFILE)
    apple.AppleGPU.gpu_records.cache_clear()
    yield


class FakeVirtualMemory:
    """Stand-in for `psutil.virtual_memory()` with a fixed layout."""

    total = 48 * 1024**3
    used = 16 * 1024**3
    available = 32 * 1024**3


@pytest.fixture
def fake_psutil_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin psutil's virtual-memory reading so unified-memory paths are deterministic."""
    from mainboard.models import memory as memory_mod

    monkeypatch.setattr(memory_mod.psutil, "virtual_memory", lambda: FakeVirtualMemory())


class FakeError(Exception):
    """Shared NVML/system error type for the fake CUDA stack."""


class CudaErrorT:
    """Mimic `cudaError_t` with a single success sentinel."""

    cudaSuccess = 0


class FakeRuntime:
    """Minimal `cuda.bindings.runtime` returning two visible devices."""

    cudaError_t = CudaErrorT

    def __init__(self, count: int = 2) -> None:
        self.count = count

    def cudaGetDeviceCount(self) -> tuple[int, int]:
        return (CudaErrorT.cudaSuccess, self.count)

    def cudaGetDeviceProperties(self, index: int) -> tuple[int, Any]:
        props = type(
            "Props",
            (),
            {
                "totalGlobalMem": 24 * 1024**3,
                "multiProcessorCount": 128,
                "memoryBusWidth": 384,
            },
        )()
        return (CudaErrorT.cudaSuccess, props)

    def cudaDeviceGetPCIBusId(self, length: int, index: int) -> tuple[int, bytes]:
        return (CudaErrorT.cudaSuccess, f"0000:0{index}:00.0\x00".encode())

    def cudaDriverGetVersion(self) -> tuple[int, int]:
        return (CudaErrorT.cudaSuccess, 13010)

    def cudaGetDevice(self) -> tuple[int, int]:
        return (CudaErrorT.cudaSuccess, 0)

    def cudaSetDevice(self, index: int) -> None:
        return None

    def cudaMemGetInfo(self) -> tuple[int, int, int]:
        return (CudaErrorT.cudaSuccess, 8 * 1024**3, 24 * 1024**3)


class FakeClock:
    def __init__(self, mhz: int) -> None:
        self.mhz = mhz

    def get_current_mhz(self) -> int:
        return self.mhz


class FakeSystemDevice:
    """Mimic `cuda.core.system.Device` NVML-backed reads."""

    name = b"NVIDIA GeForce RTX 4090"
    uuid = "GPU-deadbeef"
    cuda_compute_capability = (8, 9)
    arch = type("Arch", (), {"name": "ADA"})()

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.memory_info = type(
            "Mem", (), {"total": 24 * 1024**3, "used": 6 * 1024**3, "free": 18 * 1024**3}
        )()
        self.utilization = type("Util", (), {"gpu": 42, "memory": 17})()

    def get_clock(self, domain: str) -> FakeClock:
        return FakeClock(2520 if domain == "sm" else 10501)


class FakeCudaDevice:
    """Mimic `cuda.core.Device` with PCI id and static properties."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.pci_bus_id = f"0000:0{index}:00.0"
        self.properties = type(
            "Props",
            (),
            {
                "multiprocessor_count": 128,
                "memory_clock_rate": 10501000,
                "global_memory_bus_width": 384,
                "clock_rate": 2520000,
            },
        )()


class FakeNvml:
    """Minimal `cuda.bindings.nvml` surface used by the provider."""

    class TemperatureSensors:
        TEMPERATURE_GPU = 0

    class TemperatureThresholds:
        TEMPERATURE_THRESHOLD_SHUTDOWN = 0
        TEMPERATURE_THRESHOLD_SLOWDOWN = 1

    class ClockType:
        CLOCK_SM = 1
        CLOCK_MEM = 2

    NotSupportedError = FakeError
    NoPermissionError = FakeError
    UnknownError = FakeError
    GpuIsLostError = FakeError

    def init_v2(self) -> None:
        return None

    def device_get_handle_by_pci_bus_id_v2(self, bus_id: str) -> str:
        return f"handle:{bus_id}"

    def device_get_name(self, handle: str) -> str:
        return "NVIDIA GeForce RTX 4090"

    def device_get_uuid(self, handle: str) -> str:
        return "GPU-deadbeef"

    def device_get_cuda_compute_capability(self, handle: str) -> tuple[int, int]:
        return (8, 9)

    def device_get_memory_info_v2(self, handle: str) -> Any:
        return type(
            "Mem", (), {"total": 24 * 1024**3, "used": 6 * 1024**3, "free": 18 * 1024**3}
        )()

    def device_get_clock_info(self, handle: str, clock: int) -> int:
        return 2520 if clock == FakeNvml.ClockType.CLOCK_SM else 10501

    def device_get_max_clock_info(self, handle: str, clock: int) -> int:
        return 10501

    def device_get_utilization_rates(self, handle: str) -> Any:
        return type("Util", (), {"gpu": 42, "memory": 17})()

    def device_get_current_clocks_event_reasons(self, handle: str) -> int:
        return 0x08  # HW_SLOWDOWN

    def device_get_temperature_v(self, handle: str, sensor: int) -> int:
        return 65

    def device_get_temperature_threshold(self, handle: str, threshold: int) -> int:
        return 95 if threshold == 0 else 83

    def device_get_power_usage(self, handle: str) -> int:
        return 120_000

    def device_get_total_energy_consumption(self, handle: str) -> int:
        return 9_000_000

    def device_get_pcie_throughput(self, handle: str, counter: int) -> int:
        return 1024 if counter == 0 else 2048

    def device_get_fan_speed(self, handle: str) -> int:
        return 30

    def device_get_compute_running_processes_v3(self, handle: str) -> list[Any]:
        return [type("Proc", (), {"pid": 4321, "used_gpu_memory": 2 * 1024**3})()]


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

    def __init__(self, device_count: int = 2, has_cuda_core: bool = True) -> None:
        self.runtime = FakeRuntime(device_count)
        self.system = FakeSystem() if has_cuda_core else None
        self.nvml = FakeNvml()
        self.cuda_device_type = FakeCudaDevice if has_cuda_core else None
        self.nvml_errors = (FakeError,)

    @property
    def has_cuda_core(self) -> bool:
        """Whether the optional `cuda.core` layer is wired in this fake."""
        return self.cuda_device_type is not None


@pytest.fixture
def nvidia_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeNvidiaApis]:
    """Replace the cached CUDA/NVML stack with deterministic fakes."""
    apis = FakeNvidiaApis()
    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: apis)
    yield apis


@pytest.fixture
def nvidia_host_no_cuda_core(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeNvidiaApis]:
    """A CUDA stack without the optional `cuda.core` layer, exercising NVML-only paths."""
    apis = FakeNvidiaApis(has_cuda_core=False)
    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: apis)
    yield apis


def text_strategy() -> st.SearchStrategy[str]:
    """Printable text without control characters for name-like fields."""
    return st.text(st.characters(blacklist_categories=("Cs", "Cc")), max_size=40)


class FakeGPU:
    """A GPU stand-in for the profiling/contention helpers (no real device needed).

    Carries just the telemetry those helpers read: a peak bandwidth to score copies
    against, and live `utilization`/`memory` so `gpu_busy` has something to judge.
    """

    def __init__(self, *, gpu_pct: int = 0, memory_pct: int = 0, used_pct: float = 0.0) -> None:
        from mainboard.enums import Vendor
        from mainboard.models.memory import Memory
        from mainboard.models.utilization import Utilization

        self.vendor = Vendor.UNKNOWN  # so `Tracer.detect` picks the no-op base
        self.peak_bandwidth_gbs = 900.0
        self.utilization = Utilization(gpu_pct=gpu_pct, memory_pct=memory_pct)
        total = 1000
        self.memory = Memory(total_bytes=total, used_bytes=int(total * used_pct / 100.0))


class FakeTracer:
    """A no-op `Tracer` whose deep-trace support is fixed to `KERNEL | MEMCPY`."""

    def supported(self) -> Any:
        from mainboard.profiling.trace import Activity

        return Activity.KERNEL | Activity.MEMCPY

    def push(self, name: str) -> None:
        return None

    def pop(self) -> None:
        return None


class RecordingProfiler:
    """A `Profiler` stand-in that ignores tracing and returns a fixed kernel `Profile`.

    Stands in for the CUPTI-backed profiler so the orchestration in `bottleneck.profile`
    is testable without CUDA; its `result()` carries one hot `gemm` kernel.
    """

    def __init__(self, **_: Any) -> None:
        pass

    def __enter__(self) -> RecordingProfiler:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def result(self) -> Any:
        from mainboard.profiling.result import Profile
        from mainboard.profiling.trace import KernelTrace

        return Profile(device="fake", kernels=(KernelTrace(name="gemm", start_ns=0, end_ns=1000),))


def _patch_gpu_all(monkeypatch: pytest.MonkeyPatch, gpu: FakeGPU | None) -> None:
    """Point `GPU.all` at a single fake GPU (or none) for every reader."""
    from mainboard.gpu import GPU

    devices = (gpu,) if gpu is not None else ()
    monkeypatch.setattr(GPU, "all", classmethod(lambda cls: devices))


@pytest.fixture
def gpu_profiling_host(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """Pretend a traceable GPU is present, with a fake tracer and recording profiler."""
    from mainboard.profiling import annotate, bottleneck

    gpu = FakeGPU()
    tracer = FakeTracer()
    _patch_gpu_all(monkeypatch, gpu)
    monkeypatch.setattr(bottleneck, "tracer", lambda: tracer)
    monkeypatch.setattr(annotate, "tracer", lambda: tracer)  # region() reads this one
    monkeypatch.setattr(bottleneck, "Profiler", RecordingProfiler)
    return gpu


@pytest.fixture
def idle_gpu_host(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """One GPU sitting idle, below both the utilization and memory thresholds."""
    gpu = FakeGPU(gpu_pct=2, used_pct=10.0)
    _patch_gpu_all(monkeypatch, gpu)
    return gpu


@pytest.fixture
def busy_gpu_host(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """One GPU under heavy compute load (above the utilization threshold)."""
    gpu = FakeGPU(gpu_pct=85, used_pct=40.0)
    _patch_gpu_all(monkeypatch, gpu)
    return gpu


@pytest.fixture
def memory_pressure_gpu_host(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    """One GPU idle on compute but nearly full on memory."""
    gpu = FakeGPU(gpu_pct=1, used_pct=95.0)
    _patch_gpu_all(monkeypatch, gpu)
    return gpu


@pytest.fixture
def cpu_only_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the host has no GPU at all (the contention/profile helpers degrade)."""
    _patch_gpu_all(monkeypatch, None)
