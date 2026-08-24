from types import ModuleType

import pytest

from mainboard.probe import AppleGPU, AppleNPU, Unit, UnitKind, Vendor
from mainboard.probe.providers.apple import gpu as apple_gpu_mod
from mainboard.probe.providers.apple import npu as apple_npu_mod

# Apple Silicon has no separate GPU or Neural Engine model string on the command line, so both
# engines take their identity from the same SoC brand string and differ only by their suffix.
_ENGINES = [
    pytest.param(apple_gpu_mod, AppleGPU, "GPU", id="gpu"),
    pytest.param(apple_npu_mod, AppleNPU, "Neural Engine", id="npu"),
]
_IDENTITIES = [
    pytest.param(AppleGPU, UnitKind.GPU, "metal", "GPU", id="gpu"),
    pytest.param(AppleNPU, UnitKind.NPU, "coreml", "Neural Engine", id="npu"),
]


@pytest.mark.usefixtures("apple_host", "fake_psutil_memory")
@pytest.mark.parametrize(("engine", "kind", "backend", "suffix"), _IDENTITIES)
def test_an_apple_engine_names_itself_from_the_soc_and_reports_unified_memory(
    engine: type[Unit], kind: UnitKind, backend: str, suffix: str
) -> None:
    """Both engines share the one memory pool the CPU sees, so capacity is host RAM and the
    reading carries the unified flag rather than a device-local capacity."""
    assert engine.is_available() is True
    (unit,) = engine.all()
    assert unit.vendor is Vendor.APPLE
    assert unit.kind is kind
    assert unit.backend == backend
    assert unit.architecture == "Apple M4 Pro"
    assert unit.label == f"Apple M4 Pro {suffix}"
    assert unit.memory.total_bytes == 48 * 1024**3
    assert unit.memory.unified is True


@pytest.mark.usefixtures("apple_host")
@pytest.mark.parametrize(("module", "engine", "suffix"), _ENGINES)
def test_an_unreadable_sysctl_degrades_to_a_generic_apple_silicon_label(
    module: ModuleType, engine: type[Unit], suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `sysctl` that answers nothing (no permission, a stripped image) still names the engine."""
    monkeypatch.setattr(module, "sysctl", lambda name: "")
    unit = engine()
    assert unit.architecture == "Apple Silicon"
    assert unit.label == f"Apple Silicon {suffix}"


@pytest.mark.parametrize(("module", "engine", "suffix"), _ENGINES)
def test_off_apple_silicon_the_provider_reports_nothing(
    module: ModuleType, engine: type[Unit], suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Linux box or an Intel Mac has no Metal or Core ML engine to enumerate."""
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    assert engine.is_available() is False
    assert engine.all() == ()
