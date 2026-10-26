import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard.fleet.nodes import evidence_of
from mainboard.state import Evidence, EvidenceTree
from mainboard.trials import Artifact, Dataset
from mainboard.trials.artifacts import Artifacts
from mainboard.trials.artifacts import pinned as hub_source
from mainboard.trials.store.archive import ParquetArtifacts

_PART = "part-00000.parquet"
_JSON = "application/json"
_PAYLOAD = b"aaaabbbbcccc"
_PURE = settings(database=None, deadline=None, max_examples=100)
_NAME = st.from_regex(r"[A-Za-z0-9]+([_.-][A-Za-z0-9]+)*", fullmatch=True)
_FILE = st.lists(
    st.from_regex(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", fullmatch=True), min_size=1, max_size=3
)


def _hub_file(
    *, kind: str, owner: str, repository: str, revision: str, name: Sequence[str]
) -> tuple[Path, str]:
    """A file in the hub cache and the source it was downloaded from."""
    cached = Path("/hub", f"{kind}--{owner}--{repository}", "snapshots", revision, *name)
    prefix = "" if kind == "models" else f"{kind}/"
    return cached, f"hf://{prefix}{owner}/{repository}@{revision}/{'/'.join(name)}"


_HUB_FILES = st.builds(
    _hub_file,
    kind=st.sampled_from(["models", "datasets", "spaces"]),
    owner=_NAME,
    repository=_NAME,
    revision=st.from_regex(r"[0-9a-f]{40}", fullmatch=True),
    name=_FILE,
)
_OTHER_FILES = st.lists(st.from_regex(r"[a-z]{1,8}", fullmatch=True), min_size=1, max_size=6).map(
    lambda parts: (Path("/", *parts), "")
)


@pytest.fixture
def departed(keeper: Evidence, evidence: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The node's evidence folder, ingested and then moved out of the tree."""
    keeper.ingest([evidence])
    shutil.move(evidence, tmp_path_factory.mktemp("aside") / "evidence")
    return evidence


@settings(max_examples=50)
@given(
    payload=st.binary(max_size=2000),
    damage=st.binary(min_size=1, max_size=20),
    source=st.text(max_size=30),
)
def test_artifacts_are_held_once_per_node_and_a_damaged_copy_is_replaced(
    payload: bytes, damage: bytes, source: str
) -> None:
    with TemporaryDirectory() as root:
        store = Artifacts(Path(root), Path(root) / "evidence")
        first = store.write(payload, media_type=_JSON, source=source)
        (Path(root) / first.path).write_bytes(payload + damage)
        again = store.write(payload, media_type=_JSON)
        assert first.path == f"evidence/objects/{first.sha256[:2]}/{first.sha256}"
        assert first.sha256 == hashlib.sha256(payload).hexdigest() and first.source == source
        assert (again, (Path(root) / first.path).read_bytes()) == (
            first.model_copy(update={"source": ""}),
            payload,
        )
        assert store.written == {Path(root) / first.path}
        assert [path.name for path in Path(root).rglob("*") if path.is_file()] == [first.sha256]


@_PURE
@given(download=st.one_of(_HUB_FILES, _OTHER_FILES))
def test_a_hub_cache_file_names_the_source_it_came_from_and_any_other_file_names_none(
    download: tuple[Path, str],
) -> None:
    path, source = download
    assert hub_source(path) == source


def test_settling_verifies_what_only_the_lake_holds(
    keeper: Evidence, evidence: Path, workspace: Path, node: str
) -> None:
    keeper.ingest([evidence])
    keeper.evict(keeper.finalize([evidence]))
    [row] = Dataset(evidence / "receipts").rows()
    reference = Artifact.model_validate(row["artifacts"])
    lost = reference.model_copy(update={"sha256": "1" * 64, "path": "datasets/lost"})

    def receipts(*references: Artifact) -> list[str]:
        listed = {str(at): value.model_dump() for at, value in enumerate(references)}
        return [json.dumps({"trial_receipt": {"artifacts": listed}})]

    Artifacts.verify(
        receipts(reference, reference), directory=workspace / node, boundary=workspace
    )
    with pytest.raises(FileNotFoundError):
        Artifacts.verify(receipts(lost), directory=workspace / node, boundary=workspace)


def test_receipts_and_the_artifacts_they_pin_read_back_once_the_tree_lost_them(
    departed: Path, workspace: Path, pinned: str
) -> None:
    store = Dataset(departed / "receipts")
    [row] = store.rows()
    reference = Artifact.model_validate(row["artifacts"])
    assert (row["run"], row["lane"], reference.sha256) == ("run1", "law", pinned)
    assert hashlib.sha256(reference.read(workspace / "research" / "lab")).hexdigest() == pinned
    assert [part.name for part in store.parts] == [_PART]
    assert Dataset.holding(departed) is not None


def test_a_pinned_digest_is_read_from_the_lake_once_the_tree_lost_the_file(
    keeper: Evidence, departed: Path, workspace: Path, node: str
) -> None:
    project = workspace / "research" / "lab"
    digest = next(row.sha256 for row in keeper.indexed(node) if row.path.endswith(_PART))
    receipt = departed / "receipts" / "run=run1" / _PART
    assert hashlib.sha256(EvidenceTree(project).read(receipt, digest)).hexdigest() == digest
    assert evidence_of(project, "law") == "datasets/experiments/law/evidence"
    with pytest.raises(ValueError, match="pinned digest"):
        EvidenceTree(project).read(workspace / "mb.toml", "0" * 64)
    with pytest.raises(FileNotFoundError):
        EvidenceTree(project).recall("0" * 64)


def test_the_parts_of_an_artifact_archive_stay_in_the_tree_for_readers_that_open_them(
    kept: Evidence, digest_of: Callable[[bytes], str]
) -> None:
    folder = kept.root / "datasets" / "experiments"
    member = folder / "run" / "table.bin"
    member.parent.mkdir(parents=True)
    member.write_bytes(_PAYLOAD)
    ParquetArtifacts(folder).pack([member])
    member.unlink()
    kept.ingest([folder])
    assert kept.evict(kept.finalize([folder])) == []
    reference = Artifact(path="run/table.bin", sha256=digest_of(_PAYLOAD), size=len(_PAYLOAD))
    assert reference.read(folder) == _PAYLOAD
