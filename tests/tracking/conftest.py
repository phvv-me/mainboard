import sys

import pytest

from mainboard.manifest import Tracking

from .support import FakeWandb


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    """The tracking SDK, replaced by a stand-in for the length of one test."""
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def declared() -> Tracking:
    """The `[tracking]` table a test tunes, at the defaults a workspace that wrote none gets."""
    return Tracking()
