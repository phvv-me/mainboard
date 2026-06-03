from __future__ import annotations

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


@pytest.fixture(autouse=True)
def reset_global_caches() -> Iterator[None]:
    """Keep tests hermetic by clearing every module-level cache around each test."""
    from mainboard.machine import Machine

    Machine.__new__.cache_clear()
    reset_nvidia_cache()
    reset_apple_caches()
    yield
    Machine.__new__.cache_clear()
    reset_nvidia_cache()


@pytest.fixture(autouse=True)
def isolate_unit_registries() -> Iterator[None]:
    """Undo any `Unit` subclass a test defines, since `Registry.__init_subclass__`
    appends it to the global GPU/NPU root list and an inherited `all` would then
    recurse forever once the test's monkeypatches are torn down."""
    from mainboard.gpu import GPU
    from mainboard.npu import NPU

    roots = [next(b for b in cls.__mro__ if "_registry" in b.__dict__) for cls in (GPU, NPU)]
    saved = {root: list(root._registry) for root in roots}
    yield
    for root, members in saved.items():
        root._registry[:] = members


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
    monkeypatch.setattr(apple_gpu.psutil, "virtual_memory", lambda: FakeVirtualMemory())


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
        return (CudaErrorT.cudaSuccess, type("Props", (), {"totalGlobalMem": 24 * 1024**3})())

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
    """Drop-in replacement for `NvidiaApis` wired to the fakes above."""

    def __init__(self, device_count: int = 2) -> None:
        self.runtime = FakeRuntime(device_count)
        self.system = FakeSystem()
        self.nvml = FakeNvml()
        self.cuda_device_type = FakeCudaDevice
        self.nvml_errors = (FakeError,)


@pytest.fixture
def nvidia_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeNvidiaApis]:
    """Replace the cached CUDA/NVML stack with deterministic fakes."""
    apis = FakeNvidiaApis()
    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: apis)
    yield apis


def text_strategy() -> st.SearchStrategy[str]:
    """Printable text without control characters for name-like fields."""
    return st.text(st.characters(blacklist_categories=("Cs", "Cc")), max_size=40)
