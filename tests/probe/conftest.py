import sys
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from mainboard import Machine
from mainboard.probe import GPU, NPU
from mainboard.probe.facts import memory as memory_mod
from mainboard.probe.providers.nvidia import apis as nvidia_apis_module

from .support import FakeNvidiaApis, FakeTorch, InstallNvidiaStack


def reset_global_caches() -> None:
    """Drop the cached `Machine` and the cached CUDA/NVML import stack."""
    # SingletonMeta caches the instance on the class itself, so dropping the attribute resets it.
    Machine.__dict__.get("singleton_instance") and delattr(Machine, "singleton_instance")
    nvidia_apis_module.nvidia_apis.cache_clear()


@pytest.fixture(autouse=True)
def hermetic_caches() -> Iterator[None]:
    """Keep tests hermetic by clearing every module-level cache around each test."""
    reset_global_caches()
    yield
    reset_global_caches()


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
def fake_psutil_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin psutil's virtual-memory reading at 48 GiB, 16 used and 32 available."""
    ram = SimpleNamespace(total=48 * 1024**3, used=16 * 1024**3, available=32 * 1024**3)
    monkeypatch.setattr(memory_mod.psutil, "virtual_memory", lambda: ram)


@pytest.fixture
def install_nvidia_stack(monkeypatch: pytest.MonkeyPatch) -> InstallNvidiaStack:
    """Build the fake CUDA/NVML stack a test asks for and put it behind the cached accessor.

    Its axes are the ones the provider branches on: how many devices are visible, whether the
    optional `cuda.core` layer loaded, and whether the device reports a coherent pool (or only
    the partial one heterogeneous memory management gives a discrete card).
    """
    # The workspace exports a CUDA mask for every run; the fakes enumerate every device.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def install(**shape: int | bool) -> FakeNvidiaApis:
        apis = FakeNvidiaApis(**shape)
        monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", lambda: apis)
        return apis

    return install


@pytest.fixture
def nvidia_host(install_nvidia_stack: InstallNvidiaStack) -> FakeNvidiaApis:
    """The default fake stack, two discrete devices with the optional `cuda.core` layer."""
    return install_nvidia_stack()


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> FakeTorch:
    """Stand PyTorch in for the stress probe, which imports it only once a measurement runs."""
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch
