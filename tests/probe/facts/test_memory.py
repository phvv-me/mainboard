import pytest
from mainboard.probe import Memory
from mainboard.probe.facts import memory as memory_mod


class FakeVirtualMemory:
    total = 32 * 1024**3
    used = 20 * 1024**3
    available = 12 * 1024**3


def test_system_samples_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Memory.system` reads total/used/free straight off `psutil.virtual_memory`."""
    monkeypatch.setattr(memory_mod.psutil, "virtual_memory", lambda: FakeVirtualMemory())
    memory = Memory.system()
    assert memory.total_bytes == 32 * 1024**3
    assert memory.used_bytes == 20 * 1024**3
    assert memory.free_bytes == 12 * 1024**3
    assert memory.scope == "system"
    assert memory.unified is False
    assert memory.source == "psutil"


def test_gb_conversions_and_percent_used() -> None:
    """The gibibyte helpers and `percent_used` divide correctly."""
    memory = Memory(total_bytes=10 * 1024**3, used_bytes=5 * 1024**3, free_bytes=5 * 1024**3)
    assert memory.total_gb == 10.0
    assert memory.used_gb == 5.0
    assert memory.free_gb == 5.0
    assert memory.percent_used == 50.0


def test_percent_used_is_zero_when_total_is_zero() -> None:
    """An empty reading never divides by zero."""
    assert Memory().percent_used == 0.0
