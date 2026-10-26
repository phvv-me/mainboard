from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest

from mainboard.state import Evidence, EvidenceTree, Lake
from mainboard.trials import Artifact, Session
from mainboard.trials.artifacts import Artifacts

_PAYLOAD = b"aaaabbbbcccc"


@pytest.fixture
def closing(workspace: Path, node: str) -> tuple[Session, Artifact]:
    """A session about to close whose one store holds an object it wrote, and that object."""
    store = Artifacts(workspace, workspace / node / "evidence")
    reference = store.write(b"table", media_type="application/octet-stream")
    session = Session.__new__(Session)
    session.staged, session.leased, session.writers = Mock(), None, {}
    session.stores, session.baseline = {"law": store}, {}
    session.declared = Mock(flags=(), universe=Mock(root=workspace / node))
    return session, reference


@pytest.mark.parametrize("central", [True, False], ids=["center", "mirror"])
def test_a_closed_run_moves_into_the_center_lake(
    closing: tuple[Session, Artifact], workspace: Path, node: str, central: bool
) -> None:
    session, reference = closing
    if central:
        (workspace / ".git").mkdir()
    lake = Lake.at(workspace).ready()
    assert session.close() == ""
    assert (workspace / reference.path).exists() is not central
    assert [row.path for row in Evidence(lake).indexed(node)] == (
        [reference.path] if central else []
    )


def test_a_closed_run_moves_into_the_lake_unless_a_link_hides_it(
    kept: Evidence, tmp_path_factory: pytest.TempPathFactory, digest_of: Callable[[bytes], str]
) -> None:
    moved = kept.root / "run" / "receipt.bin"
    moved.parent.mkdir()
    moved.write_bytes(_PAYLOAD)
    volume = tmp_path_factory.mktemp("volume")
    (kept.root / "datasets").symlink_to(volume, target_is_directory=True)
    tree = EvidenceTree(kept.root)
    assert tree.keep([kept.root / "datasets" / "node.bin"]) is None
    assert tree.keep([moved.parent]) is not None
    assert not moved.parent.exists() and kept.recall(digest_of(_PAYLOAD)) == _PAYLOAD
    assert list(volume.iterdir()) == [] and kept.indexed() == kept.indexed("run")


def test_a_closed_run_written_during_collection_stays_in_the_tree(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch, digest_of: Callable[[bytes], str]
) -> None:
    active = kept.root / "active.bin"
    active.write_bytes(b"before")
    stage = kept.blobs.stage

    def mutate(connection: duckdb.DuckDBPyConnection, objects: Mapping[str, bytes | Path]) -> int:
        with active.open("r+b") as writer:
            writer.write(b"after!")
        return stage(connection, objects)

    monkeypatch.setattr(kept.blobs, "stage", mutate)
    assert EvidenceTree(kept.root).keep([active]) is not None
    assert active.read_bytes() == b"after!" and kept.recall(digest_of(b"before")) == b"before"
