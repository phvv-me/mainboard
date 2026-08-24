from collections.abc import Iterator

import pytest

from mainboard import Machine
from mainboard.probe import GPU, NPU
from mainboard.probe.facts import memory as memory_mod
from mainboard.probe.providers.apple import gpu as apple_gpu_mod
from mainboard.probe.providers.apple import npu as apple_npu_mod
from mainboard.probe.providers.nvidia import apis as nvidia_apis_module

from .support import FakeNvidiaApis, InstallNvidiaStack


def reset_nvidia_cache() -> None:
    """Drop the cached CUDA/NVML import stack."""
    nvidia_apis_module.nvidia_apis.cache_clear()


def reset_machine_singleton() -> None:
    """Drop the cached `Machine` so each test builds a fresh, isolated instance."""
    # SingletonMeta caches the instance on the class itself, so dropping the
    # class attribute is the reset.
    Machine.__dict__.get("singleton_instance") and delattr(Machine, "singleton_instance")


@pytest.fixture(autouse=True)
def reset_global_caches() -> Iterator[None]:
    """Keep tests hermetic by clearing every module-level cache around each test."""
    reset_machine_singleton()
    reset_nvidia_cache()
    yield
    reset_machine_singleton()
    reset_nvidia_cache()


@pytest.fixture(autouse=True)
def isolate_unit_registries() -> Iterator[None]:
    """Undo any `Unit` subclass a test defines.

    `Registry.__init_subclass__` appends it to the global GPU/NPU root list, and a leaked
    subclass would leak into every later test's `GPU.all()`/`NPU.all()`.
    """

    # `registry()` returns the nearest root's live list, so snapshot a copy and restore in place.
    saved_gpu = list(GPU.registry())
    saved_npu = list(NPU.registry())
    yield
    GPU.registry()[:] = saved_gpu
    NPU.registry()[:] = saved_npu


@pytest.fixture
def apple_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend the host is an Apple Silicon Mac reporting a fixed chip name."""
    monkeypatch.setattr(apple_gpu_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(apple_gpu_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(apple_npu_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(apple_npu_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(apple_gpu_mod, "sysctl", lambda name: "Apple M4 Pro")
    monkeypatch.setattr(apple_npu_mod, "sysctl", lambda name: "Apple M4 Pro")
    yield


class FakeVirtualMemory:
    """Stand-in for `psutil.virtual_memory()` with a fixed layout."""

    total = 48 * 1024**3
    used = 16 * 1024**3
    available = 32 * 1024**3


@pytest.fixture
def fake_psutil_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin psutil's virtual-memory reading so unified-memory paths are deterministic."""

    monkeypatch.setattr(memory_mod.psutil, "virtual_memory", lambda: FakeVirtualMemory())


@pytest.fixture
def install_nvidia_stack(monkeypatch: pytest.MonkeyPatch) -> InstallNvidiaStack:
    """Build the fake CUDA/NVML stack a test asks for and put it behind the cached accessor.

    The three axes are the ones the provider branches on, how many devices are visible, whether
    the optional `cuda.core` layer loaded, and whether the device reports a coherent pool.
    """

    def install(
        *, device_count: int = 2, has_cuda_core: bool = True, coherent: bool = False
    ) -> FakeNvidiaApis:
        apis = FakeNvidiaApis(device_count, has_cuda_core=has_cuda_core, coherent=coherent)
        nvidia_apis_module.nvidia_apis.cache_clear()
        monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: apis)
        return apis

    return install


@pytest.fixture
def nvidia_host(install_nvidia_stack: InstallNvidiaStack) -> FakeNvidiaApis:
    """The default fake stack, two discrete devices with the optional `cuda.core` layer."""
    return install_nvidia_stack()
