import pytest

from mainboard.probe import AppleNPU, UnitKind, Vendor
from mainboard.probe.providers.apple import npu as apple_npu_mod


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
def test_apple_npu_reads_identity_and_unified_memory() -> None:
    """The Apple Neural Engine derives its name from `sysctl` and unified host memory."""
    assert AppleNPU.is_available() is True
    npus = AppleNPU.all()
    assert len(npus) == 1
    npu = npus[0]
    assert npu.vendor == Vendor.APPLE
    assert npu.kind == UnitKind.NPU
    assert npu.architecture == "Apple M4 Pro"
    assert npu.label == "Apple M4 Pro Neural Engine"
    assert npu.memory.total_bytes == 48 * 1024**3
    assert npu.memory.unified is True


def test_apple_npu_unavailable_off_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-Darwin or non-arm64 hosts the Apple NPU provider reports nothing."""
    monkeypatch.setattr(apple_npu_mod.platform, "system", lambda: "Linux")
    assert AppleNPU.is_available() is False
    assert AppleNPU.all() == ()


def test_apple_npu_architecture_falls_back_when_sysctl_is_empty(
    apple_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty `sysctl` reading degrades to a generic label."""
    monkeypatch.setattr(apple_npu_mod, "sysctl", lambda name: "")
    npu = AppleNPU()
    assert npu.architecture == "Apple Silicon"
    assert npu.label == "Apple Silicon Neural Engine"
