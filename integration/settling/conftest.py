from pathlib import Path

import pytest

from mainboard.settling.frames import FrameRelay
from mainboard.state import TelemetryLake

from ..state.telemetry.builders import lake_with
from .fakes import Connections


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> Connections:
    """The relay's ssh seam, answering from this machine and noting each host it is asked for."""
    seam = Connections()
    monkeypatch.setattr("mainboard.settling.frames.connection", seam)
    return seam


@pytest.fixture
def lake(workspace: Path) -> TelemetryLake:
    """The workspace's telemetry lake as the sink leaves it, holding nothing."""
    return lake_with(workspace, [])


@pytest.fixture
def relay(lake: TelemetryLake) -> FrameRelay:
    """The relay into the workspace's spool, over a lake that exists."""
    return FrameRelay(lake)
