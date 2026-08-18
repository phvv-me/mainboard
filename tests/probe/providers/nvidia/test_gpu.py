from typing import NoReturn

import pytest

from mainboard import Machine
from mainboard.probe import GPU, Memory, NvidiaGPU, Vendor
from mainboard.probe.providers.nvidia import apis as nvidia_apis_module

from ...conftest import FakeError, FakeNvidiaApis, raise_unsupported


def test_nvidia_detects_and_describes_devices(nvidia_host: FakeNvidiaApis) -> None:
    """The NVIDIA provider reads identity, capability, and memory from the fakes."""
    assert NvidiaGPU.is_available() is True
    gpus = NvidiaGPU.all()
    assert len(gpus) == 2
    gpu = gpus[0]
    assert gpu.vendor == Vendor.NVIDIA
    assert gpu.label == "NVIDIA GeForce RTX 4090"
    assert gpu.uuid == "GPU-deadbeef"
    assert str(gpu.cuda_architecture) == "8.9"
    assert gpu.architecture == "Ada"
    assert gpu.arch_key == "sm_89"
    assert gpu.memory.total_bytes == 24 * 1024**3
    assert gpu.driver_version == (13, 1)


def test_nvidia_discrete_gpu_is_not_coherent(nvidia_host: FakeNvidiaApis) -> None:
    """A discrete card reports neither coherence attribute, so its memory is not unified."""
    gpu = NvidiaGPU(index=0)
    assert gpu.coherent is False
    assert gpu.memory.unified is False


def test_nvidia_coherent_pool_sets_unified(nvidia_coherent_host: FakeNvidiaApis) -> None:
    """A device reporting both coherence attributes flags its memory as unified."""
    gpu = NvidiaGPU(index=0)
    assert gpu.coherent is True
    assert gpu.memory.unified is True


def test_nvidia_coherent_pool_sets_unified_without_cuda_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unified flag flows through the NVML-only memory path too (no `cuda.core`)."""
    apis = FakeNvidiaApis(has_cuda_core=False, coherent=True)
    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: apis)
    assert NvidiaGPU(index=0).memory.unified is True


def test_nvidia_coherence_probe_degrades_when_attribute_query_absent(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A binding without `cudaDeviceGetAttribute` degrades to not-coherent, not a crash."""

    def absent(*args) -> NoReturn:
        raise AttributeError("module 'cuda.bindings.runtime' has no cudaDeviceGetAttribute")

    monkeypatch.setattr(nvidia_host.runtime, "cudaDeviceGetAttribute", absent)
    gpu = NvidiaGPU(index=0)
    assert gpu.coherent is False
    assert gpu.memory.unified is False


def test_nvidia_memory_cuda_core_path(nvidia_host: FakeNvidiaApis) -> None:
    """Tier one: `cuda.core.system.Device.memory_info` when the optional layer loaded."""
    mem = NvidiaGPU(index=0).memory
    assert (mem.total_bytes, mem.used_bytes, mem.free_bytes) == (
        24 * 1024**3,
        6 * 1024**3,
        18 * 1024**3,
    )
    assert mem.source == "cuda-core-system"


def test_nvidia_snapshot_free_survives_json(nvidia_host: FakeNvidiaApis) -> None:
    """The live NVIDIA reading is JSON round-trippable through `Memory`."""

    mem = NvidiaGPU(index=0).memory
    assert Memory.model_validate_json(mem.model_dump_json()) == mem


def test_nvidia_nvml_only_paths(nvidia_host_no_cuda_core: FakeNvidiaApis) -> None:
    """Tier two: with `cuda.core` absent, identity, capability, and memory read through
    `cuda.bindings` (runtime + NVML) alone."""
    assert nvidia_host_no_cuda_core.has_cuda_core is False
    assert NvidiaGPU.is_available() is True
    gpu = NvidiaGPU(index=0)
    assert gpu.label == "NVIDIA GeForce RTX 4090"
    assert gpu.uuid == "GPU-deadbeef"
    assert str(gpu.cuda_architecture) == "8.9"
    assert gpu.architecture == "Ada"  # from the compute-capability table (cc 8.9)
    mem = gpu.memory
    assert (mem.total_bytes, mem.used_bytes, mem.source) == (24 * 1024**3, 6 * 1024**3, "nvml")


def test_nvidia_pci_bus_id_error_raises(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `cudaDeviceGetPCIBusId` surfaces as a clear runtime error."""
    monkeypatch.setattr(
        nvidia_host.runtime, "cudaDeviceGetPCIBusId", lambda length, index: (99, b"")
    )
    with pytest.raises(RuntimeError, match="cudaDeviceGetPCIBusId"):
        _ = NvidiaGPU(index=0).pci_bus_id


def test_nvidia_nvml_memory_unsupported_falls_back_to_runtime(
    nvidia_host_no_cuda_core: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier three: NVML memory raising drops to the CUDA-runtime `cudaMemGetInfo` reading."""
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
        def memory_info(self) -> NoReturn:
            raise self.NotSupportedError

    monkeypatch.setattr(nvidia_host.runtime, "cudaGetDevice", lambda: (99, 0))
    monkeypatch.setattr(nvidia_host.runtime, "cudaMemGetInfo", lambda: (99, 0, 0))
    gpu = NvidiaGPU(index=0)
    monkeypatch.setattr(type(gpu), "system_device", Unsupported())
    with pytest.raises(RuntimeError, match="cudaMemGetInfo"):
        _ = gpu.memory


def test_nvidia_falls_back_to_runtime_when_cuda_core_memory_unsupported(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `cuda.core` device that cannot report memory falls to the runtime tier."""

    class Unsupported:
        NotSupportedError = FakeError

        @property
        def memory_info(self) -> NoReturn:
            raise self.NotSupportedError

    gpu = NvidiaGPU(index=0)
    monkeypatch.setattr(type(gpu), "system_device", Unsupported())
    mem = gpu.memory
    assert mem.total_bytes == 24 * 1024**3
    assert mem.used_bytes == 24 * 1024**3 - 8 * 1024**3
    assert mem.source == "cuda-runtime"


def test_nvidia_apis_property_is_cached_per_instance(nvidia_host: FakeNvidiaApis) -> None:
    """`GPU.apis` is a `cached_property` that reads the process-wide cached stack."""
    gpu = NvidiaGPU(index=0)
    assert gpu.apis is nvidia_apis_module.nvidia_apis()


def test_nvidia_unavailable_when_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero device count makes the provider report unavailable and empty."""
    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: FakeNvidiaApis(device_count=0))
    assert NvidiaGPU.is_available() is False
    assert NvidiaGPU.all() == ()


def test_nvidia_unavailable_when_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing CUDA modules make `is_available` return False without raising."""

    def boom() -> NoReturn:
        raise ModuleNotFoundError("no cuda")

    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", boom)
    assert NvidiaGPU.is_available() is False


def test_machine_degrades_without_cuda_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A base install without the `[cuda]` extra reports no NVIDIA devices.

    Simulates the bindings being absent at the import seam itself (the real
    `NvidiaApis` constructor runs and fails to import `cuda.bindings.runtime`),
    so `GPU.all` and `Machine` detection degrade instead of raising."""

    def absent(name: str) -> NoReturn:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(nvidia_apis_module, "import_module", absent)
    nvidia_apis_module.nvidia_apis.cache_clear()
    assert NvidiaGPU.is_available() is False
    assert NvidiaGPU.all() == ()
    assert all(gpu.vendor is not Vendor.NVIDIA for gpu in GPU.all())
    assert all(gpu.vendor is not Vendor.NVIDIA for gpu in Machine().gpus)


def test_system_api_raises_when_cuda_core_unavailable(
    nvidia_host_no_cuda_core: FakeNvidiaApis,
) -> None:
    """`system_api` refuses rather than returning `None` when `cuda.core` never loaded."""
    with pytest.raises(RuntimeError, match="is unavailable"):
        _ = NvidiaGPU(index=0).system_api


def test_utilization_reads_through_cuda_core(nvidia_host: FakeNvidiaApis) -> None:
    assert nvidia_host.has_cuda_core is True
    reading = NvidiaGPU(index=0).utilization
    assert (reading.gpu_pct, reading.memory_pct) == (61, 37)


def test_utilization_falls_back_to_nvml(nvidia_host_no_cuda_core: FakeNvidiaApis) -> None:
    assert nvidia_host_no_cuda_core.has_cuda_core is False
    reading = NvidiaGPU(index=0).utilization
    assert (reading.gpu_pct, reading.memory_pct) == (48, 22)


def test_utilization_degrades_to_empty_when_both_layers_refuse(
    nvidia_host_no_cuda_core: FakeNvidiaApis,
) -> None:
    def refuse(handle: str) -> object:
        raise FakeError("counters unavailable")

    nvidia_host_no_cuda_core.nvml.device_get_utilization_rates = refuse
    reading = NvidiaGPU(index=0).utilization
    assert (reading.gpu_pct, reading.memory_pct) == (0, 0)
