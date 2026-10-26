import hashlib
import shutil
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.state import Evidence
from mainboard.state.evidence import keeping as keeping_module
from mainboard.state.evidence.keeping import windows
from mainboard.state.storage.blobs import Blobs

_COUNTS = "SELECT (SELECT count(*) FROM lake.blobs), (SELECT count(*) FROM lake.evidence_log)"
_PAYLOADS = [b"same", b"same", b"same", b"diff", b"kept"]
_FOLDER = "evidence"


@pytest.fixture
def windowed(keeper: Evidence, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder of five small files, each ingest window eight bytes wide.

    Three hold the same bytes, one other bytes and one the bytes an earlier ingest kept.
    """
    monkeypatch.setattr(Blobs, "STAGED_BYTES", 8)
    previous = workspace / "previous.bin"
    previous.write_bytes(b"kept")
    keeper.ingest([previous])
    root = workspace / _FOLDER
    root.mkdir()
    for index, payload in enumerate(_PAYLOADS):
        (root / f"{index}.bin").write_bytes(payload)
    return root


@settings(max_examples=200)
@given(
    weights=st.lists(st.integers(min_value=0, max_value=100), max_size=40),
    limit=st.integers(min_value=0, max_value=300),
)
def test_windows_keep_the_order_and_each_weighs_at_most_the_limit_unless_alone(
    weights: Sequence[int], limit: int
) -> None:
    packed = list(windows(weights, lambda weight: weight, limit))
    assert [weight for window in packed for weight in window] == list(weights)
    assert all(window and (len(window) == 1 or sum(window) <= limit) for window in packed)
    assert all(sum(before) + after[0] > limit for before, after in pairwise(packed))


def test_ingest_deduplicates_within_and_across_windows(
    keeper: Evidence, windowed: Path, digest_of: Callable[[bytes], str]
) -> None:
    same, different, kept = (digest_of(payload) for payload in _PAYLOADS[2:])
    with patch.object(Blobs, "held", autospec=True, side_effect=Blobs.held) as membership:
        first, again = keeper.ingest([windowed]), keeper.ingest([windowed])
    assert [first.model_dump(), again.model_dump()] == [
        {"files": 5, "indexed": 5, "objects": 2, "size": 8},
        {"files": 5, "indexed": 0, "objects": 0, "size": 0},
    ]
    assert [call.kwargs.get("digests", call.args[-1]) for call in membership.call_args_list] == [
        {same},
        {same, different},
        {kept},
    ] * 2
    assert keeper.lake.query(_COUNTS) == [(3, 6)]
    shutil.rmtree(windowed)
    assert len(keeper.materialize([windowed])) == 5
    assert [path.read_bytes() for path in sorted(windowed.iterdir())] == _PAYLOADS
    assert keeper.verify() == []


def test_a_group_of_small_files_lands_in_one_data_file(keeper: Evidence, workspace: Path) -> None:
    root = workspace / "experiments" / "many" / _FOLDER
    root.mkdir(parents=True)
    for index in range(300):
        (root / f"{index:03}.bin").write_bytes(f"{index:03}".encode() * 1000)
    keeper.ingest([root])
    files = list(keeper.lake.data.rglob("*.parquet"))
    assert [sum(table in path.parts for path in files) for table in ("blobs", "checksums")] == [
        1,
        1,
    ]
    [(chunks, sums)] = keeper.lake.query(
        "SELECT (SELECT count(*) FROM lake.blobs), (SELECT count(*) FROM lake.checksums)"
    )
    assert chunks == sums == 300
    assert keeper.verify() == []


@pytest.mark.parametrize(
    ("target", "named"),
    [
        pytest.param("unrelated.txt", "linked", id="a-file-link"),
        pytest.param("", "", id="a-folder-holding-a-link"),
        pytest.param("", "linked/unrelated.txt", id="a-route-through-one"),
    ],
)
def test_ingest_refuses_a_link_before_keeping_anything(
    kept: Evidence, tmp_path_factory: pytest.TempPathFactory, target: str, named: str
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "unrelated.txt").write_bytes(b"outside workspace bytes")
    folder = kept.root / _FOLDER
    folder.mkdir()
    (folder / "fine.txt").write_bytes(b"fine")
    (folder / "linked").symlink_to(outside / target)
    with pytest.raises(MissionError, match="is a link"):
        kept.ingest([folder / named])
    assert kept.indexed() == []
    assert (outside / "unrelated.txt").read_bytes().startswith(b"outside")


def test_ingest_ignores_the_tools_own_aside(kept: Evidence) -> None:
    folder = kept.root / _FOLDER
    (folder / ".mb-custody-0f").mkdir(parents=True)
    (folder / ".mb-custody-0f" / "retained.bin").write_bytes(b"retained")
    (folder / "kept.bin").write_bytes(b"kept")
    assert kept.ingest([folder]).files == 1
    assert [row.path for row in kept.indexed()] == ["evidence/kept.bin"]


def test_a_file_that_changes_while_it_is_read_is_not_ingested(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = kept.root / "moving.bin"
    source.write_bytes(b"before")
    racing = SimpleNamespace(
        sha256=lambda payload: (
            source.write_bytes(b"after the read began"),
            hashlib.sha256(payload),
        )[1],
        file_digest=hashlib.file_digest,
    )
    monkeypatch.setattr(keeping_module, "hashlib", racing)
    with pytest.raises(MissionError, match="changed while reading"):
        kept.ingest([source])
    assert kept.indexed() == []
