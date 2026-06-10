from __future__ import annotations

import platform

import pytest

import mainboard
from mainboard import Machine, MachineSnapshot

pytestmark = pytest.mark.integration


def test_real_probe_describes_this_host() -> None:
    """The unmocked probe returns a valid snapshot adapted to the running hardware.

    Apple Silicon exposes only Apple GPUs, yet a headless or virtualized macOS host may
    expose none, so vendor is validated without assuming any GPU is present.
    """
    payload = Machine().model_dump_json()
    snapshot = MachineSnapshot.model_validate_json(payload)

    assert isinstance(snapshot, MachineSnapshot)
    assert snapshot.cpu.name
    assert snapshot.cpu.physical_cores > 0
    assert snapshot.memory.total_bytes > 0
    assert snapshot.unit_count == 1 + len(snapshot.gpus) + len(snapshot.npus)

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert all(gpu.vendor == mainboard.Vendor.APPLE for gpu in snapshot.gpus)
    else:
        assert isinstance(snapshot.gpus, tuple)
