from types import ModuleType

import pytest

from mainboard.probe.providers.nvidia import apis as nvidia_apis_module

from ...support import FakeError, FakeNvidiaApis, FakeNvml, FakeRuntime, FakeSystem

# What a faked `import_module` hands back, one stand-in per layer the real stack imports.
type Loaded = FakeRuntime | FakeNvml | FakeSystem | ModuleType


@pytest.mark.parametrize(
    ("nvml_name", "has_cuda_core"),
    [
        pytest.param("cuda.bindings._nvml", True, id="private-nvml"),
        pytest.param("cuda.bindings.nvml", True, id="public-nvml"),
        pytest.param("cuda.bindings._nvml", False, id="no-cuda-core"),
    ],
)
def test_the_stack_binds_whichever_nvml_and_optional_layer_the_host_offers(
    nvml_name: str, has_cuda_core: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional `cuda.core` layer may fail without taking the stack down.

    `cuda.bindings` is required and either NVML spelling satisfies it, while `cuda.core` is
    optional and its compiled extensions can fail to load behind another library's
    `libstdc++`, so an `ImportError` there leaves the stack usable with the layer off.
    """
    stack = FakeNvidiaApis()
    core = ModuleType("cuda.core")
    core.Device = stack.cuda_device_type
    modules = {"cuda.bindings.runtime": stack.runtime, nvml_name: stack.nvml}
    if has_cuda_core:
        modules |= {"cuda.core.system": stack.system, "cuda.core": core}

    def loader(name: str) -> Loaded:
        if name in modules:
            return modules[name]
        if name.startswith("cuda.core"):
            raise ImportError("CXXABI_1.3.15 not found")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(nvidia_apis_module, "import_module", loader)
    apis = nvidia_apis_module.NvidiaApis()
    assert apis.runtime is stack.runtime
    assert apis.nvml is stack.nvml
    assert apis.has_cuda_core is has_cuda_core
    assert (apis.system is stack.system) is has_cuda_core
    assert set(apis.nvml_errors) == {FakeError}  # every error type the binding exposes


def test_the_stack_is_imported_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The accessor caches one stack for every device.

    Loading NVML is expensive and every GPU instance reads the same handles.
    """
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
    """The suppression set is read off what the binding actually exposes.

    A stripped NVML module yields an empty tuple instead of an `AttributeError` while it is
    built.
    """
    stack = FakeNvidiaApis()

    def loader(name: str) -> Loaded:
        if name == "cuda.bindings.runtime":
            return stack.runtime
        if name.startswith("cuda.bindings"):
            return ModuleType("nvml")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(nvidia_apis_module, "import_module", loader)
    assert nvidia_apis_module.NvidiaApis().nvml_errors == ()
