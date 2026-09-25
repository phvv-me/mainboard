import pytest

from mainboard.probe import AppleGPU, AppleNPU, Unit, UnitKind, Vendor
from mainboard.probe.providers.apple import silicon


@pytest.mark.usefixtures("fake_psutil_memory")
@pytest.mark.parametrize(
    ("engine", "kind", "backend", "suffix"),
    [
        pytest.param(AppleGPU, UnitKind.GPU, "metal", "GPU", id="gpu"),
        pytest.param(AppleNPU, UnitKind.NPU, "coreml", "Neural Engine", id="npu"),
    ],
)
@pytest.mark.parametrize(
    ("brand", "soc"),
    [
        pytest.param("Apple M4 Pro", "Apple M4 Pro", id="named-soc"),
        pytest.param("", "Apple Silicon", id="unreadable-sysctl"),
    ],
)
def test_an_apple_engine_names_itself_from_the_soc_and_reports_unified_memory(
    engine: type[Unit],
    kind: UnitKind,
    backend: str,
    suffix: str,
    brand: str,
    soc: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both engines take their identity from the one SoC brand string and share its RAM.

    A `sysctl` that answers nothing (no permission, a stripped image) still names the engine,
    and capacity is the unified host pool the CPU sees, not a device capacity.
    """
    monkeypatch.setattr(silicon.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(silicon.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(silicon, "sysctl", lambda name: brand)
    (unit,) = engine.all()
    assert (unit.vendor, unit.kind, unit.backend) == (Vendor.APPLE, kind, backend)
    assert (unit.architecture, unit.label) == (soc, f"{soc} {suffix}")
    assert unit.memory.total_bytes == 48 * 1024**3
    assert unit.memory.unified is True


def test_off_apple_silicon_the_providers_report_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Linux box or an Intel Mac has no Metal or Core ML engine to enumerate."""
    monkeypatch.setattr(silicon.platform, "system", lambda: "Linux")
    assert (AppleGPU.all(), AppleNPU.all()) == ((), ())
