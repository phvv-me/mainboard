import pytest
from mainboard.probe import AppleGPU, UnitKind, Vendor
from mainboard.probe.providers.apple import gpu as apple_gpu_mod


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_gpu_reads_identity_and_unified_memory() -> None:
    """The Apple GPU derives its name from `sysctl` and reports unified host memory."""
    assert AppleGPU.is_available() is True
    gpus = AppleGPU.all()
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.vendor == Vendor.APPLE
    assert gpu.kind == UnitKind.GPU
    assert gpu.architecture == "Apple M4 Pro"
    assert gpu.label == "Apple M4 Pro GPU"
    assert gpu.memory.total_bytes == 48 * 1024**3
    assert gpu.memory.unified is True


def test_apple_unavailable_off_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-Darwin or non-arm64 hosts the Apple GPU provider reports nothing."""
    monkeypatch.setattr(apple_gpu_mod.platform, "system", lambda: "Linux")
    assert AppleGPU.is_available() is False
    assert AppleGPU.all() == ()


def test_apple_architecture_falls_back_when_sysctl_is_empty(
    apple_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty `sysctl` reading (e.g. no permission) degrades to a generic label."""
    monkeypatch.setattr(apple_gpu_mod, "sysctl", lambda name: "")
    gpu = AppleGPU(index=0)
    assert gpu.architecture == "Apple Silicon"
    assert gpu.label == "Apple Silicon GPU"
