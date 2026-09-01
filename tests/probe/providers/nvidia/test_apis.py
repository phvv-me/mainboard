from types import ModuleType

import pytest

from mainboard.probe.providers.nvidia import apis as nvidia_apis_module

from ...support import FakeError, FakeNvidiaApis, FakeNvml, FakeRuntime, FakeSystem

type Loaded = FakeRuntime | FakeNvml | FakeSystem | ModuleType


@pytest.mark.parametrize("has_cuda_core", [True, False], ids=["cuda-core", "no-cuda-core"])
def test_a_unix_stack_binds_public_nvml_and_each_optional_cuda_layer(
    has_cuda_core: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional CUDA Core failure leaves public NVML and CUDA Runtime usable."""
    stack = FakeNvidiaApis()
    core = ModuleType("cuda.core")
    core.Device = stack.cuda_device_type
    modules = {
        "cuda.bindings.nvml": stack.nvml,
        "cuda.bindings.runtime": stack.runtime,
    }
    if has_cuda_core:
        modules |= {"cuda.core.system": stack.system, "cuda.core": core}

    def loader(name: str) -> Loaded:
        if name in modules:
            return modules[name]
        if name.startswith("cuda.core"):
            raise ImportError("CXXABI_1.3.15 not found")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(nvidia_apis_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(nvidia_apis_module, "import_module", loader)
    apis = nvidia_apis_module.NvidiaApis()
    assert apis.runtime is stack.runtime
    assert apis.nvml is stack.nvml
    assert apis.has_cuda_core is has_cuda_core
    assert (apis.system is stack.system) is has_cuda_core
    assert set(apis.nvml_errors) == {FakeError}


def test_windows_discovery_imports_only_public_cuda_python_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A facts read does not pull runtime or compiler extensions into the control process."""
    stack = FakeNvidiaApis()
    loaded: list[str] = []

    def loader(name: str) -> Loaded:
        loaded.append(name)
        if name == "cuda.bindings.nvml":
            return stack.nvml
        raise AssertionError(f"unexpected Windows CUDA import: {name}")

    monkeypatch.setattr(nvidia_apis_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(nvidia_apis_module, "import_module", loader)
    apis = nvidia_apis_module.NvidiaApis()
    assert loaded == ["cuda.bindings.nvml"]
    assert apis.nvml is stack.nvml
    assert apis.runtime is None
    assert apis.has_cuda_core is False


def test_the_stack_is_imported_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The accessor caches one stack for every device."""
    built: list[int] = []

    class Marker:
        def __init__(self) -> None:
            built.append(1)

    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "NvidiaApis", Marker)
    assert nvidia_apis_module.nvidia_apis() is nvidia_apis_module.nvidia_apis()
    assert built == [1]


def test_a_binding_that_names_no_error_types_suppresses_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stripped public NVML module yields an empty suppression tuple."""
    monkeypatch.setattr(nvidia_apis_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(nvidia_apis_module, "import_module", lambda name: ModuleType("nvml"))
    assert nvidia_apis_module.NvidiaApis().nvml_errors == ()
