import pytest

import mainboard
from mainboard import CPU, GPU, GPUSnapshot, UnitSnapshot
from mainboard.enums import UnitKind, Vendor
from mainboard.unit import Unit


def test_base_unit_neutral_defaults() -> None:
    """The base `Unit` reports unknown identity and empty telemetry."""
    unit = Unit()
    assert unit.kind == UnitKind.UNKNOWN
    assert unit.vendor == Vendor.UNKNOWN
    assert unit.label == "unknown"
    assert unit.architecture == "unknown"
    assert unit.memory.total_bytes == 0
    assert unit.clock_readings == ()
    assert unit.memory.supported is True
    snapshot = unit.snapshot("region")
    assert isinstance(snapshot, UnitSnapshot)
    assert snapshot.name == "region"


def test_base_gpu_defaults_and_snapshot() -> None:
    """A bare `GPU` exposes zeroed sensors and a `GPUSnapshot`."""
    gpu = GPU()
    assert gpu.kind == UnitKind.GPU
    assert gpu.label == "unknown"
    assert gpu.uuid == ""
    assert gpu.architecture == "unknown"
    assert gpu.arch_key == "unknown"  # base key is the lowercased architecture name
    assert gpu.memory.total_bytes == 0
    assert gpu.memory.supported is False
    assert gpu.peak_bandwidth_gbs == 0.0
    assert gpu.driver_version is None
    assert gpu.temperature_c == 0
    assert gpu.gpu_util_pct == 0
    assert gpu.fan_speed_pct == 0
    assert gpu.processes == []
    snapshot = gpu.snapshot("kernel")
    assert isinstance(snapshot, GPUSnapshot)
    assert snapshot.memory.total_bytes == 0


def test_cpu_exposes_identity_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `CPU` surfaces its identity fields and a system memory reading."""
    vm = type("VM", (), {"total": 64 * 1024**3, "used": 8 * 1024**3, "available": 56 * 1024**3})()
    monkeypatch.setattr("mainboard.models.memory.psutil.virtual_memory", lambda: vm)
    cpu = CPU(
        name_value="Apple M4 Pro",
        architecture_value="arm64",
        logical_cores=14,
        physical_cores=14,
        current_clock_mhz=3200.0,
        vendor=Vendor.APPLE,
    )
    assert cpu.kind == UnitKind.CPU
    assert cpu.label == "Apple M4 Pro"
    assert cpu.architecture == "arm64"
    assert cpu.clock_readings[0].current_mhz == 3200.0
    assert cpu.memory.scope == "system"
    assert cpu.memory.total_bytes == 64 * 1024**3


def test_cpu_clock_reading_unsupported_when_unknown() -> None:
    """A CPU with no frequency reports an unsupported clock reading."""
    cpu = CPU(name_value="x", architecture_value="x", current_clock_mhz=None)
    assert cpu.clock_readings[0].supported is False


@pytest.mark.parametrize(
    ("provider", "kind", "vendor"),
    [
        (mainboard.AppleGPU, UnitKind.GPU, Vendor.APPLE),
        (mainboard.AppleNPU, UnitKind.NPU, Vendor.APPLE),
        (mainboard.NvidiaGPU, UnitKind.GPU, Vendor.NVIDIA),
    ],
)
def test_provider_identity_without_backend_imports(
    provider: type, kind: UnitKind, vendor: Vendor
) -> None:
    """Provider identity fields resolve without touching any backend module."""
    unit = provider(index=0)
    assert unit.kind == kind
    assert unit.vendor == vendor
