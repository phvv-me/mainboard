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
    (tmp_path / "alpha").mkdir()
    return declaration(tmp_path)


@pytest.fixture
def store(declared: Declaration) -> Dataset:
    return declared.universe.dataset("alpha")


@pytest.fixture
def probed(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every session opened in this test stamps the same fixed provenance."""
    monkeypatch.setattr(session_module, "Preflight", Taken)
    yield


@pytest.fixture
def session(declared: Declaration, probed: None) -> Session:
    return Session(declared)


@pytest.fixture
def research(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A research `experiments` tree whose `alpha` registration a dispatch captured, run from
    the workspace root since a registration is named relative to it."""
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
