from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest

from mainboard.dispatch.provenance import Row, Status, blob_of, listing
from mainboard.trials import Dataset, Declaration, Session
from mainboard.trials import session as session_module
from mainboard.trials.provenance import Source

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


@pytest.fixture
def research(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A research `experiments` tree whose `alpha` registration a dispatch captured.

    The working directory is the workspace root, because a registration is named relative to it,
    and every session opened in the test reads its source bundle from the fixed probe.
    """
    root = tmp_path / "experiments"
    node = root / "alpha/node.md"
    node.parent.mkdir(parents=True)
    node.write_text("committed registration\n")
    monkeypatch.chdir(tmp_path)
    closure = tmp_path / "closure.tsv"
    closure.write_text(
        listing([Row(path="experiments/alpha/node.md", blob=blob_of(node), status=Status.CLEAN)])
    )
    captured = Source(digest="a" * 64, closure=str(closure), root=tmp_path)
    monkeypatch.setattr(session_module, "Preflight", partial(Taken, source=captured))
    return root
