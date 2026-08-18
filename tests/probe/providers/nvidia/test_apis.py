import pytest

from mainboard.probe.providers.nvidia import apis as nvidia_apis_module

from ...conftest import FakeNvidiaApis, FakeNvml, FakeRuntime, FakeSystem


def test_text_decodes_bytes_and_passes_through_str() -> None:
    """`text` decodes NVML/CUDA byte strings and leaves plain strings untouched."""
    assert nvidia_apis_module.text(b"NVIDIA") == "NVIDIA"
    assert nvidia_apis_module.text("NVIDIA") == "NVIDIA"


def test_apis_tolerates_missing_cuda_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing `cuda.core` import leaves `NvidiaApis` usable with `has_cuda_core` False."""
    fake_nvml = FakeNvidiaApis().nvml
    modules = {
        "cuda.bindings.runtime": FakeNvidiaApis().runtime,
        "cuda.bindings.nvml": fake_nvml,
    }

    def loader(name: str) -> FakeRuntime | FakeNvml:
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
def test_apis_wires_module_surface_via_either_nvml(
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

    def loader(name: str) -> FakeRuntime | FakeSystem | FakeNvml | type:
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    monkeypatch.setattr(nvidia_apis_module, "import_module", loader)
    apis = nvidia_apis_module.NvidiaApis()
    assert apis.cuda_device_type is fake_device
    assert apis.nvml is fake_nvml
    assert apis.has_cuda_core is True


def test_apis_cache_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
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
