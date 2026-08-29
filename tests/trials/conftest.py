from collections.abc import Iterator
from pathlib import Path

import pytest

from mainboard.trials import Dataset, Declaration, Session
from mainboard.trials import session as session_module

from .support import Taken, declaration


@pytest.fixture
def declared(tmp_path: Path) -> Declaration:
    """A universe rooted in a scratch directory, with two axes and three settle words."""
    (tmp_path / "alpha").mkdir()
    return declaration(tmp_path)


@pytest.fixture
def store(declared: Declaration) -> Dataset:
    """The `alpha` claim's receipt store, which every storage test writes through."""
    return declared.universe.dataset("alpha")


@pytest.fixture
def probed(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every session opened in this test stamps the same fixed provenance.

    Patched at the one seam a session reads it through, so nothing under test touches real
    silicon, a real repository or the clock of the machine running the suite.
    """
    monkeypatch.setattr(session_module, "Preflight", Taken)
    yield


@pytest.fixture
def session(declared: Declaration, probed: None) -> Session:
    """One run of the declared universe, its provenance fixed."""
    return Session(declared)
