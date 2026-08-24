import os
import sys
from collections.abc import Iterator

import pytest

from mainboard.dispatch.backends import Credentials

from .support import FakeModal


@pytest.fixture
def unsealed() -> Iterator[None]:
    """Unseal the shared credential loader for one test, then put the environment back.

    The suite seals the loader so no test reads the developer's own `.env`, and a test about the
    loader itself has to undo that. Loading writes straight into `os.environ` rather than
    through monkeypatch, so the whole environment is snapshotted here and restored on the way
    out, and the loader is sealed again behind it.
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
