from collections.abc import Callable

import pytest

from mainboard import (
    AMDGPU,
    GPU,
    GPUSnapshot,
    IntelGPU,
    IntelNPU,
    Machine,
    NvidiaGPU,
    QualcommGPU,
    QualcommNPU,
)
from mainboard.enums import CudaPythonVariant, UnitKind, Vendor
from mainboard.providers.apple import AppleGPU, AppleNPU
from mainboard.providers.apple import gpu as apple_gpu
from mainboard.providers.apple import profile as apple_profile
from mainboard.providers.nvidia import apis as nvidia_apis_module

from .conftest import FakeError, FakeNvidiaApis, raise_unsupported


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
    assert gpu.memory.total_bytes == 48 * 1024**3
    assert gpu.memory.unified is True
    assert [c.domain for c in gpu.clock_readings] == ["gpu_compute", "memory"]
    assert all(not c.supported for c in gpu.clock_readings)


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_gpu_snapshot_is_a_gpu_snapshot() -> None:
    """An Apple GPU snapshot is a `GPUSnapshot` carrying unified memory."""
    snapshot = AppleGPU.all()[0].snapshot("region")
    assert isinstance(snapshot, GPUSnapshot)
    assert snapshot.name == "region"
    assert snapshot.memory.total_bytes == 48 * 1024**3


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_npu_names_itself_from_the_chip() -> None:
    """The Apple NPU derives its name and memory from the SoC profile."""
    assert AppleNPU.is_available() is True
    npu = AppleNPU.all()[0]
    assert npu.vendor == Vendor.APPLE
    assert npu.name == "Apple M4 Pro Neural Engine"
    assert npu.architecture == "Apple M4 Pro"
    assert npu.memory.total_bytes == 48 * 1024**3
    assert npu.memory.unified is True
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
    monkeypatch.setattr(apple_gpu.platform, "system", lambda: "Linux")
    AppleGPU.gpu_records.cache_clear()
    assert AppleGPU.is_available() is False
    assert AppleGPU.gpu_records() == ()
    assert AppleNPU.is_available() is False
    assert AppleNPU.all() == ()


def test_apple_system_profile_reads_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """`apple_system_profile` returns the shell-parsed profiler payload."""
    profile = {"SPHardwareDataType": [{"chip_type": "Apple M4 Pro"}]}
    monkeypatch.setattr(apple_profile.shell, "system_profiler", lambda *types: profile)
    payload = apple_profile.apple_system_profile()
    assert payload["SPHardwareDataType"][0]["chip_type"] == "Apple M4 Pro"


def test_apple_gpu_records_tolerate_profiler_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty profiler payload yields no Apple GPU records instead of raising."""
    monkeypatch.setattr(apple_gpu.platform, "system", lambda: "Darwin")
    AppleGPU.gpu_records.cache_clear()
    monkeypatch.setattr(apple_profile, "apple_system_profile", lambda: {})
    assert AppleGPU.gpu_records() == ()


def test_apple_units_tolerate_empty_hardware_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty `SPHardwareDataType` list degrades identity fields instead of raising."""
    records = ({"sppci_device_type": "spdisplays_gpu", "sppci_model": "Apple M4 Pro"},)
    monkeypatch.setattr(AppleGPU, "gpu_records", classmethod(lambda cls: records))
    monkeypatch.setattr(apple_profile, "apple_system_profile", lambda: {"SPHardwareDataType": []})
    gpu = AppleGPU(index=0)
    assert gpu.uuid == ""
    assert gpu.architecture == "Apple M4 Pro"  # falls back to the GPU name
    npu = AppleNPU()
    assert npu.architecture == "Apple Silicon"
    assert npu.name == "Apple Silicon Neural Engine"


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
    assert gpu.memory.total_bytes == 24 * 1024**3
    assert gpu.driver_version == (13, 1)
    assert gpu.peak_bandwidth_gbs == pytest.approx(10501000 * 2 * 384 / 8 / 1e6)


def test_nvidia_cuda_python_variant_follows_driver(nvidia_host: object) -> None:
    """A driver major of 13 selects the CU13 CUDA Python stack."""
    assert NvidiaGPU(index=0).cuda_python.variant == CudaPythonVariant.CU13


def test_nvidia_live_sensors(nvidia_host: object) -> None:
    """Live NVML sensors flow through to memory, clocks, thermal, energy, and pcie."""
    gpu = NvidiaGPU(index=0)
    mem = gpu.memory
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


def test_nvidia_nvml_only_paths(nvidia_host_no_cuda_core: FakeNvidiaApis) -> None:
    """When `cuda.core` is absent (e.g. a torch wheel poisons `libstdc++`), discovery,
    identity, capability, memory, clocks, utilization, and bandwidth all read through
    `cuda.bindings` (runtime + NVML) alone."""
    assert nvidia_host_no_cuda_core.has_cuda_core is False
    assert NvidiaGPU.is_available() is True
    gpu = NvidiaGPU(index=0)
    assert gpu.name == "NVIDIA GeForce RTX 4090"
    assert gpu.uuid == "GPU-deadbeef"
    assert str(gpu.cuda_architecture) == "8.9"
    assert gpu.architecture == "Ada"  # from the compute-capability table (cc 8.9)
    assert gpu.sm_count == 128
    mem = gpu.memory
    assert (mem.total_bytes, mem.used_bytes, mem.source) == (24 * 1024**3, 6 * 1024**3, "nvml")
    assert (gpu.clocks.sm_mhz, gpu.clocks.memory_mhz) == (2520, 10501)
    assert (gpu.utilization.gpu_pct, gpu.utilization.memory_pct) == (42, 17)
    expected_bw = 10501 * 1e3 * 2 * 384 / 8 / 1e6
    assert gpu.peak_bandwidth_gbs == pytest.approx(expected_bw)


def test_nvidia_pci_bus_id_error_raises(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `cudaDeviceGetPCIBusId` surfaces as a clear runtime error."""
    monkeypatch.setattr(
        nvidia_host.runtime, "cudaDeviceGetPCIBusId", lambda length, index: (99, b"")
    )
    with pytest.raises(RuntimeError, match="cudaDeviceGetPCIBusId"):
        _ = NvidiaGPU(index=0).pci_bus_id


def test_nvidia_device_properties_error_raises(
    nvidia_host_no_cuda_core: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `cudaGetDeviceProperties` surfaces as a clear runtime error."""
    monkeypatch.setattr(
        nvidia_host_no_cuda_core.runtime, "cudaGetDeviceProperties", lambda index: (99, None)
    )
    with pytest.raises(RuntimeError, match="cudaGetDeviceProperties"):
        _ = NvidiaGPU(index=0).sm_count


def test_nvidia_nvml_memory_unsupported_falls_back_to_runtime(
    nvidia_host_no_cuda_core: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NVML memory raising drops to the CUDA-runtime `cudaMemGetInfo` reading."""
    monkeypatch.setattr(
        nvidia_host_no_cuda_core.nvml, "device_get_memory_info_v2", raise_unsupported
    )
    mem = NvidiaGPU(index=0).memory
    assert mem.source == "cuda-runtime"
    assert mem.total_bytes == 24 * 1024**3


def test_nvidia_runtime_memory_error_raises(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `cudaMemGetInfo` failure during the runtime fallback raises."""

    class Unsupported:
        NotSupportedError = FakeError

        @property
        def memory_info(self) -> object:
            raise self.NotSupportedError

    monkeypatch.setattr(nvidia_host.runtime, "cudaGetDevice", lambda: (99, 0))
    monkeypatch.setattr(nvidia_host.runtime, "cudaMemGetInfo", lambda: (99, 0, 0))
    gpu = NvidiaGPU(index=0)
    monkeypatch.setattr(type(gpu), "system_device", Unsupported())
    with pytest.raises(RuntimeError, match="cudaMemGetInfo"):
        _ = gpu.memory


@pytest.mark.parametrize(
    ("fixture", "broken", "expected"),
    [
        pytest.param(
            "nvidia_host",
            (
                "device_get_fan_speed",
                "device_get_power_usage",
                "device_get_pcie_throughput",
                "device_get_compute_running_processes_v3",
            ),
            lambda gpu: (gpu.fan_speed_pct, gpu.energy.power_mw, gpu.pcie.tx_kbs, gpu.processes)
            == (0, 0, 0, []),
            id="sensors-degrade",
        ),
        pytest.param(
            "nvidia_host_no_cuda_core",
            (
                "device_get_temperature_v",
                "device_get_temperature_threshold",
                "device_get_current_clocks_event_reasons",
                "device_get_utilization_rates",
            ),
            lambda gpu: (
                gpu.thermal.temperature_c,
                gpu.thermal.slowdown_threshold_c,
                gpu.thermal.throttle_reasons,
                gpu.utilization.gpu_pct,
                gpu.utilization.memory_pct,
            )
            == (0, 0, 0, 0, 0),
            id="thermal-and-util-degrade",
        ),
        pytest.param(
            "nvidia_host",
            ("device_get_temperature_threshold",),
            lambda gpu: (gpu.thermal.temperature_c, gpu.thermal.slowdown_threshold_c) == (65, 0),
            id="threshold-fails-temp-survives",
        ),
    ],
)
def test_nvidia_sensors_degrade_on_nvml_errors(
    fixture: str,
    broken: tuple[str, ...],
    expected: Callable[[NvidiaGPU], bool],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVML sensors that raise degrade to defaults while the rest of the read survives."""
    apis: FakeNvidiaApis = request.getfixturevalue(fixture)
    for method in broken:
        monkeypatch.setattr(apis.nvml, method, raise_unsupported)
    assert expected(NvidiaGPU(index=0))


def test_nvidia_utilization_degrades_when_cuda_core_unsupported(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `cuda.core` device that cannot report utilization yields zeros, not a crash."""

    class Unsupported:
        NotSupportedError = FakeError

        @property
        def utilization(self) -> object:
            raise self.NotSupportedError

    gpu = NvidiaGPU(index=0)
    monkeypatch.setattr(type(gpu), "system_device", Unsupported())
    assert (gpu.utilization.gpu_pct, gpu.utilization.memory_pct) == (0, 0)


def test_nvidia_falls_back_to_runtime_when_nvml_memory_unsupported(
    nvidia_host: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When NVML memory is unsupported, memory and clocks fall back to the runtime."""

    class Unsupported:
        NotSupportedError = FakeError

        @property
        def memory_info(self) -> object:
            raise self.NotSupportedError

        def get_clock(self, domain: str) -> object:
            raise self.NotSupportedError

    gpu = NvidiaGPU(index=0)
    monkeypatch.setattr(type(gpu), "system_device", Unsupported())
    mem = gpu.memory
    assert mem.total_bytes == 24 * 1024**3
    assert mem.used_bytes == 24 * 1024**3 - 8 * 1024**3
    clocks = gpu.clocks
    assert clocks.sm_mhz == round(2520000 / 1000)
    assert clocks.memory_mhz == round(10501000 / 1000)


def test_nvidia_apis_tolerates_missing_cuda_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing `cuda.core` import leaves `NvidiaApis` usable with `has_cuda_core` False."""
    fake_nvml = FakeNvidiaApis().nvml
    modules = {
        "cuda.bindings.runtime": FakeNvidiaApis().runtime,
        "cuda.bindings.nvml": fake_nvml,
    }

    def loader(name: str) -> object:
        if name in modules:
            return modules[name]
        if name in ("cuda.core", "cuda.core.system"):
            raise ImportError("CXXABI_1.3.15 not found")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(nvidia_apis_module, "import_module", loader)
    apis = nvidia_apis_module.NvidiaApis()
    assert apis.has_cuda_core is False
    assert apis.system is None
    assert apis.nvml is fake_nvml


@pytest.mark.parametrize("nvml_module", ["cuda.bindings._nvml", "cuda.bindings.nvml"])
def test_nvidia_apis_wires_module_surface_via_either_nvml(
    nvml_module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`NvidiaApis` wires runtime/system/core/nvml handles from `import_module`, taking
    the public `nvml` binding only when the private `_nvml` module is absent."""
    fake_nvml = FakeNvidiaApis().nvml
    fake_device = FakeNvidiaApis().cuda_device_type
    modules = {
        "cuda.bindings.runtime": FakeNvidiaApis().runtime,
        "cuda.core.system": FakeNvidiaApis().system,
        nvml_module: fake_nvml,
        "cuda.core": type("Core", (), {"Device": fake_device}),
    }

    def loader(name: str) -> object:
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    monkeypatch.setattr(nvidia_apis_module, "import_module", loader)
    apis = nvidia_apis_module.NvidiaApis()
    assert apis.cuda_device_type is fake_device
    assert apis.nvml is fake_nvml


def test_nvidia_apis_cache_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """`nvidia_apis` caches a single `NvidiaApis` per process."""
    built: list[int] = []

    class Marker:
        def __init__(self) -> None:
            built.append(1)

    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "NvidiaApis", Marker)
    first = nvidia_apis_module.nvidia_apis()
    second = nvidia_apis_module.nvidia_apis()
    assert first is second
    assert built == [1]
    nvidia_apis_module.nvidia_apis.cache_clear()


def test_nvidia_unavailable_when_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero device count makes the provider report unavailable and empty."""
    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: FakeNvidiaApis(device_count=0))
    assert NvidiaGPU.is_available() is False
    assert NvidiaGPU.all() == ()


def test_nvidia_unavailable_when_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing CUDA modules make `is_available` return False without raising."""

    def boom() -> object:
        raise ModuleNotFoundError("no cuda")

    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", boom)
    assert NvidiaGPU.is_available() is False


@pytest.mark.usefixtures("apple_host")
def test_machine_degrades_without_cuda_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A base install without the `[cuda]` extra reports no NVIDIA devices.

    Simulates the bindings being absent at the import seam itself (the real
    `NvidiaApis` constructor runs and fails to import `cuda.bindings.runtime`),
    so `GPU.all` and `Machine` detection degrade instead of raising."""

    def absent(name: str) -> object:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(nvidia_apis_module, "import_module", absent)
    nvidia_apis_module.nvidia_apis.cache_clear()
    assert NvidiaGPU.is_available() is False
    assert NvidiaGPU.all() == ()
    assert all(gpu.vendor is not Vendor.NVIDIA for gpu in GPU.all())
    assert all(gpu.vendor is not Vendor.NVIDIA for gpu in Machine().gpus)
