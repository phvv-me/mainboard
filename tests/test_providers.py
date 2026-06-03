from __future__ import annotations

import pytest

from mainboard import (
    AMDGPU,
    GPUSnapshot,
    IntelGPU,
    IntelNPU,
    NvidiaGPU,
    QualcommGPU,
    QualcommNPU,
)
from mainboard.enums import CudaPythonVariant, UnitKind, Vendor
from mainboard.providers import apple, nvidia
from mainboard.providers.apple import AppleGPU, AppleNPU

from .conftest import FakeNvidiaApis


@pytest.mark.parametrize(
    ("provider", "vendor", "kind"),
    [
        (AMDGPU, Vendor.AMD, UnitKind.GPU),
        (IntelGPU, Vendor.INTEL, UnitKind.GPU),
        (IntelNPU, Vendor.INTEL, UnitKind.NPU),
        (QualcommGPU, Vendor.QUALCOMM, UnitKind.GPU),
        (QualcommNPU, Vendor.QUALCOMM, UnitKind.NPU),
    ],
)
def test_future_provider_stubs_are_inert_with_static_identity(
    provider: type, vendor: Vendor, kind: UnitKind
) -> None:
    """Every stub provider is import-safe, unavailable, detects no devices, and resolves
    its vendor/kind identity without importing any backend."""
    assert provider.is_available() is False
    assert provider.all() == ()
    unit = provider(index=0)
    assert unit.vendor == vendor
    assert unit.kind == kind


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_gpu_reads_system_profiler() -> None:
    """The Apple GPU exposes profiler identity and unified host memory."""
    assert AppleGPU.is_available() is True
    gpus = AppleGPU.all()
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.vendor == Vendor.APPLE
    assert gpu.name == "Apple M4 Pro"
    assert gpu.architecture == "Apple M4 Pro"
    assert gpu.core_count == 20
    assert gpu.metal_support == "spdisplays_metal4"
    assert gpu.uuid == "00000000-1111-2222-3333-444455556666"
    assert gpu.total_memory_bytes == 48 * 1024**3
    reading = gpu.memory_readings[0]
    assert reading.unified is True
    assert reading.total_bytes == 48 * 1024**3
    assert [c.domain for c in gpu.clock_readings] == ["gpu_compute", "memory"]
    assert all(not c.supported for c in gpu.clock_readings)


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_gpu_snapshot_is_a_gpu_snapshot() -> None:
    """An Apple GPU snapshot is a `GPUSnapshot` carrying unified memory."""
    snapshot = AppleGPU.all()[0].snapshot("region")
    assert isinstance(snapshot, GPUSnapshot)
    assert snapshot.name == "region"
    assert snapshot.gpu_memory.total_bytes == 48 * 1024**3


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_npu_names_itself_from_the_chip() -> None:
    """The Apple NPU derives its name and memory from the SoC profile."""
    assert AppleNPU.is_available() is True
    npu = AppleNPU.all()[0]
    assert npu.vendor == Vendor.APPLE
    assert npu.name == "Apple M4 Pro Neural Engine"
    assert npu.architecture == "Apple M4 Pro"
    assert npu.total_memory_bytes == 48 * 1024**3
    assert npu.memory_readings[0].unified is True
    assert npu.clock_readings[0].domain == "npu"


def test_apple_core_count_tolerates_non_numeric(
    apple_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-integer core count degrades to zero instead of raising."""
    records = ({"sppci_device_type": "spdisplays_gpu", "sppci_cores": "many"},)
    monkeypatch.setattr(AppleGPU, "gpu_records", classmethod(lambda cls: records))
    assert AppleGPU(index=0).core_count == 0


def test_apple_unavailable_off_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-Darwin or non-arm64 hosts the Apple providers report nothing."""
    monkeypatch.setattr(apple.platform, "system", lambda: "Linux")
    apple.AppleGPU.gpu_records.cache_clear()
    assert AppleGPU.is_available() is False
    assert AppleGPU.gpu_records() == ()
    assert AppleNPU.is_available() is False
    assert AppleNPU.all() == ()


def test_apple_system_profile_reads_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """`apple_system_profile` returns the shell-parsed profiler payload."""
    profile = {"SPHardwareDataType": [{"chip_type": "Apple M4 Pro"}]}
    monkeypatch.setattr(apple.shell, "system_profiler", lambda *types: profile)
    assert apple.apple_system_profile()["SPHardwareDataType"][0]["chip_type"] == "Apple M4 Pro"


def test_apple_gpu_records_tolerate_profiler_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty profiler payload yields no Apple GPU records instead of raising."""
    monkeypatch.setattr(apple.platform, "system", lambda: "Darwin")
    apple.AppleGPU.gpu_records.cache_clear()
    monkeypatch.setattr(apple, "apple_system_profile", lambda: {})
    assert apple.AppleGPU.gpu_records() == ()


def test_nvidia_detects_and_describes_devices(nvidia_host: object) -> None:
    """The NVIDIA provider reads identity, capability, and memory from the fakes."""
    assert NvidiaGPU.is_available() is True
    gpus = NvidiaGPU.all()
    assert len(gpus) == 2
    gpu = gpus[0]
    assert gpu.vendor == Vendor.NVIDIA
    assert gpu.name == "NVIDIA GeForce RTX 4090"
    assert gpu.uuid == "GPU-deadbeef"
    assert str(gpu.cuda_architecture) == "8.9"
    assert gpu.architecture == "Ada"
    assert gpu.sm_count == 128
    assert gpu.total_memory_bytes == 24 * 1024**3
    assert gpu.driver_version == (13, 1)
    assert gpu.peak_bandwidth_gbs == pytest.approx(10501000 * 2 * 384 / 8 / 1e6)


def test_nvidia_cuda_python_variant_follows_driver(nvidia_host: object) -> None:
    """A driver major of 13 selects the CU13 CUDA Python stack."""
    assert NvidiaGPU(index=0).cuda_python.variant == CudaPythonVariant.CU13


def test_nvidia_live_sensors(nvidia_host: object) -> None:
    """Live NVML sensors flow through to memory, clocks, thermal, energy, and pcie."""
    gpu = NvidiaGPU(index=0)
    mem = gpu.memory_readings[0]
    assert (mem.total_bytes, mem.used_bytes, mem.free_bytes) == (
        24 * 1024**3,
        6 * 1024**3,
        18 * 1024**3,
    )
    assert (gpu.clocks.sm_mhz, gpu.clocks.memory_mhz) == (2520, 10501)
    assert (gpu.utilization.gpu_pct, gpu.utilization.memory_pct) == (42, 17)
    assert gpu.thermal.temperature_c == 65
    assert gpu.thermal.slowdown_threshold_c == 83
    assert gpu.energy.power_mw == 120_000
    assert gpu.pcie.tx_kbs == 1024
    assert gpu.fan_speed_pct == 30
    assert gpu.processes[0].pid == 4321


def test_nvidia_snapshot_round_trips(nvidia_host: object) -> None:
    """A live NVIDIA snapshot serializes and rebuilds equally."""
    snapshot = NvidiaGPU(index=0).snapshot("k")
    assert isinstance(snapshot, GPUSnapshot)
    assert GPUSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot


def test_nvidia_runtime_props_error_raises(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-success `cudaGetDeviceProperties` surfaces a clear runtime error."""

    def failing(index: int) -> tuple[int, object]:
        return (99, None)

    monkeypatch.setattr(nvidia_host.runtime, "cudaGetDeviceProperties", failing)
    with pytest.raises(RuntimeError, match="cudaGetDeviceProperties"):
        _ = NvidiaGPU(index=0).total_memory_bytes


def test_nvidia_runtime_mem_info_error_raises(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `cudaMemGetInfo` failure during the runtime fallback raises."""

    class Unsupported:
        NotSupportedError = nvidia_host.system.NotSupportedError

        @property
        def memory_info(self) -> object:
            raise self.NotSupportedError

    monkeypatch.setattr(nvidia_host.runtime, "cudaGetDevice", lambda: (99, 0))
    monkeypatch.setattr(nvidia_host.runtime, "cudaMemGetInfo", lambda: (99, 0, 0))
    gpu = NvidiaGPU(index=0)
    monkeypatch.setattr(type(gpu), "system_device", Unsupported())
    with pytest.raises(RuntimeError, match="cudaMemGetInfo"):
        _ = gpu.mem_info


def test_nvidia_sensors_degrade_on_nvml_errors(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fan speed, energy, pcie, and processes degrade to defaults on NVML errors."""

    def boom(*args: object, **kwargs: object) -> object:
        raise FakeNvidiaApis().nvml.NotSupportedError

    for method in (
        "device_get_fan_speed",
        "device_get_power_usage",
        "device_get_pcie_throughput",
        "device_get_compute_running_processes_v3",
    ):
        monkeypatch.setattr(nvidia_host.nvml, method, boom)
    gpu = NvidiaGPU(index=0)
    assert gpu.fan_speed_pct == 0
    assert gpu.energy.power_mw == 0
    assert gpu.pcie.tx_kbs == 0
    assert gpu.processes == []


def test_nvidia_falls_back_to_runtime_when_nvml_memory_unsupported(
    nvidia_host: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When NVML memory is unsupported, memory and clocks fall back to the runtime."""

    class Unsupported:
        NotSupportedError = FakeNvidiaApis().system.NotSupportedError

        @property
        def memory_info(self) -> object:
            raise self.NotSupportedError

        def get_clock(self, domain: str) -> object:
            raise self.NotSupportedError

    gpu = NvidiaGPU(index=0)
    monkeypatch.setattr(type(gpu), "system_device", Unsupported())
    mem = gpu.memory_readings[0]
    assert mem.total_bytes == 24 * 1024**3  # from cudaMemGetInfo
    assert mem.used_bytes == 24 * 1024**3 - 8 * 1024**3
    clocks = gpu.clocks
    assert clocks.sm_mhz == round(2520000 / 1000)
    assert clocks.memory_mhz == round(10501000 / 1000)


def test_nvidia_apis_imports_real_module_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """`NvidiaApis` wires runtime/system/nvml/core handles from `import_module`."""
    fake_nvml = FakeNvidiaApis().nvml
    fake_device = FakeNvidiaApis().cuda_device_type
    modules = {
        "cuda.bindings.runtime": FakeNvidiaApis().runtime,
        "cuda.core.system": FakeNvidiaApis().system,
        "cuda.bindings._nvml": fake_nvml,
        "cuda.core": type("Core", (), {"Device": fake_device}),
    }
    monkeypatch.setattr(nvidia, "import_module", lambda name: modules[name])
    apis = nvidia.NvidiaApis()
    assert apis.cuda_device_type is fake_device
    assert apis.nvml is fake_nvml


def test_nvidia_apis_falls_back_to_public_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing private `_nvml` module falls back to the public `nvml` binding."""
    fake_nvml = FakeNvidiaApis().nvml

    def loader(name: str) -> object:
        if name == "cuda.bindings._nvml":
            raise ModuleNotFoundError(name)
        return {
            "cuda.bindings.runtime": FakeNvidiaApis().runtime,
            "cuda.core.system": FakeNvidiaApis().system,
            "cuda.bindings.nvml": fake_nvml,
            "cuda.core": type("Core", (), {"Device": FakeNvidiaApis().cuda_device_type}),
        }[name]

    monkeypatch.setattr(nvidia, "import_module", loader)
    apis = nvidia.NvidiaApis()
    assert apis.nvml is fake_nvml


def test_nvidia_apis_cache_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """`nvidia_apis` caches a single `NvidiaApis` per process."""
    built: list[int] = []

    class Marker:
        def __init__(self) -> None:
            built.append(1)

    nvidia.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia, "NvidiaApis", Marker)
    first = nvidia.nvidia_apis()
    second = nvidia.nvidia_apis()
    assert first is second
    assert built == [1]
    nvidia.nvidia_apis.cache_clear()


def test_nvidia_unavailable_when_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero device count makes the provider report unavailable and empty."""
    nvidia.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia, "nvidia_apis", lambda: FakeNvidiaApis(device_count=0))
    assert NvidiaGPU.is_available() is False
    assert NvidiaGPU.all() == ()


def test_nvidia_unavailable_when_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing CUDA modules make `is_available` return False without raising."""

    def boom() -> object:
        raise ModuleNotFoundError("no cuda")

    nvidia.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia, "nvidia_apis", boom)
    assert NvidiaGPU.is_available() is False
