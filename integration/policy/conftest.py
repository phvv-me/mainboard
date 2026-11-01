"""Fixtures of the policy tests: a home of their own, and the owner typing at a terminal."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from mainboard import Project
from mainboard.render import Audience, Terminal
from mainboard.state.storage.database import cache_home


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home of the test's own, where the grants file goes and nothing else of the owner's.

    It keeps the user's DuckDB extensions, linked, since a test may not download them, and reads
    Codex's user configuration from it too.
    """
    cached = cache_home() / Project().name
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    if (moved := cache_home() / Project().name) != cached:
        moved.parent.mkdir(parents=True, exist_ok=True)
        moved.symlink_to(cached)
    return home


@pytest.fixture
def owner() -> Iterator[Terminal]:
    """The owner typing at a terminal, the one person a write of the grants file serves."""
    with Terminal(audience=Audience.HUMAN, interactive=True).installed() as terminal:
        yield terminal
