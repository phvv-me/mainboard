from __future__ import annotations

import platform

import pytest

import maquina
from maquina import Machine, MachineSnapshot

pytestmark = pytest.mark.integration


def test_real_probe_describes_this_host() -> None:
    """The unmocked probe returns a valid snapshot adapted to the running hardware."""
    payload = Machine().model_dump_json()
    snapshot = MachineSnapshot.model_validate_json(payload)

    assert isinstance(snapshot, MachineSnapshot)
    assert snapshot.cpu.name
    assert snapshot.cpu.physical_cores > 0
    assert snapshot.memory.total_bytes > 0
    assert snapshot.unit_count == 1 + len(snapshot.gpus) + len(snapshot.npus)

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert any(gpu.vendor == maquina.Vendor.APPLE for gpu in snapshot.gpus)
    else:
        assert isinstance(snapshot.gpus, tuple)
