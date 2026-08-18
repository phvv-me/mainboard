import pytest
from mainboard.probe import CPU, UnitKind, Vendor
from mainboard.probe.facts import memory as memory_mod


class FakeVirtualMemory:
    total = 64 * 1024**3
    used = 8 * 1024**3
    available = 56 * 1024**3


def test_cpu_exposes_identity_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `CPU` surfaces its identity fields and a system memory reading."""
    monkeypatch.setattr(memory_mod.psutil, "virtual_memory", lambda: FakeVirtualMemory())
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
    assert cpu.current_clock_mhz == 3200.0
    assert cpu.memory.scope == "system"
    assert cpu.memory.total_bytes == 64 * 1024**3


def test_cpu_current_clock_defaults_to_none() -> None:
    """A CPU with no reported frequency keeps `current_clock_mhz` as `None`."""
    cpu = CPU(name_value="x", architecture_value="x", current_clock_mhz=None)
    assert cpu.current_clock_mhz is None
