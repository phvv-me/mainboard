import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from mainboard.dispatch.backends import Credentials
from mainboard.dispatch.backends import hpcai as hpcai_module
from mainboard.dispatch.backends import vast as vast_module

from ..support import keypair
from .support import FakeModal


@pytest.fixture
def unsealed() -> Iterator[None]:
    """Unseal the credential loader the suite seals against the developer's `.env`.

    Loading writes straight into `os.environ`, so the environment is snapshotted and restored.
    """
    before = dict(os.environ)
    Credentials().loaded = False
    yield
    os.environ.clear()
    os.environ.update(before)
    Credentials().loaded = True


@pytest.fixture
def fake_modal() -> Iterator[FakeModal]:
    """A `FakeModal` injected into `sys.modules["modal"]`, restored after the test."""
    fake = FakeModal()
    sys.modules["modal"] = fake  # type: ignore[assignment]  reason=a hermetic double stands in for the real optional package since=2026-08-17
    yield fake
    del sys.modules["modal"]


@pytest.fixture
def home_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The standard key pair of a home directory at `tmp_path`, its private half returned."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return keypair(tmp_path)


@pytest.fixture
def answering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every rental answers ssh at its first knock."""
    for module in (hpcai_module, vast_module):
        monkeypatch.setattr(module, "reachable", lambda endpoint, *, sleeper: endpoint)
