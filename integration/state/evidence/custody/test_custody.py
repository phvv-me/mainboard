import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path

import duckdb
import pytest

from mainboard import MissionError
from mainboard.state import Evidence, Lake
from mainboard.state.evidence import keeping as keeping_module
from mainboard.state.storage.blobs import Blobs

_PAYLOAD = b"aaaabbbbcccc"
_PART = "part-00000.parquet"
_TABLES = ["blobs", "checksums", "evidence_log", "imports", "schema_log"]
_WINDOWS = pytest.mark.parametrize(
    "staged_bytes", [256 << 20, 0], ids=["in-one-query", "a-chunk-at-a-time"]
)
_DAMAGES = [
    pytest.param(["SELECT 1"], "intact", "", id="intact"),
    pytest.param(["DELETE FROM lake.checksums"], "intact", "", id="kept-before-checksums"),
    pytest.param(
        ["INSERT INTO lake.blobs SELECT * FROM lake.blobs"], "intact", "", id="written-twice"
    ),
    pytest.param(
        ["DELETE FROM lake.blobs WHERE ordinal = 1"],
        "corrupt",
        "1 of its 3 chunks are missing",
        id="a-chunk-missing",
    ),
    pytest.param(
        ["DELETE FROM lake.blobs WHERE ordinal = 2"],
        "corrupt",
        "the lake holds 8 of its 12 bytes",
        id="the-last-chunk-missing",
    ),
    pytest.param(
        ["UPDATE lake.blobs SET ordinal = 2 - ordinal"],
        "corrupt",
        "a chunk fails its checksum",
        id="chunks-reordered",
    ),
    pytest.param(
        ["INSERT INTO lake.blobs SELECT sha256, 'xxxx'::BLOB, 0 FROM lake.blobs LIMIT 1"],
        "corrupt",
        "a chunk fails its checksum",
        id="two-copies-disagreeing",
    ),
    pytest.param(
        ["UPDATE lake.blobs SET bytes = 'c'::BLOB WHERE ordinal = 2"],
        "corrupt",
        "the lake holds 9 of its 12 bytes",
        id="a-chunk-truncated",
    ),
    pytest.param(
        ["UPDATE lake.blobs SET bytes = NULL WHERE ordinal = 2"],
        "corrupt",
        "the lake holds 8 of its 12 bytes",
        id="a-cell-lost",
    ),
    pytest.param(
        ["UPDATE lake.blobs SET bytes = repeat('\\x00'::BLOB, 4) WHERE ordinal = 1"],
        "corrupt",
        "a chunk fails its checksum",
        id="a-chunk-changed",
    ),
    pytest.param(
        ["DELETE FROM lake.checksums", "UPDATE lake.blobs SET bytes = 'xxxx'::BLOB"],
        "corrupt",
        "a chunk fails its checksum",
        id="changed-without-checksums",
    ),
    pytest.param(
        ["INSERT INTO lake.blobs SELECT DISTINCT sha256, 'zzzz'::BLOB, 3 FROM lake.blobs"],
        "corrupt",
        "the lake holds 16 of its 12 bytes",
        id="an-extra-chunk",
    ),
    pytest.param(["DELETE FROM lake.blobs"], "missing", "no copy held", id="gone"),
]


def _faults(kept: Evidence) -> list[tuple[str, str, str]]:
    """Each finding of `verify` as its kind, its path and what is wrong with the object."""
    return [
        (finding.kind, finding.detail.partition(" (")[0], finding.detail.partition("): ")[2])
        for finding in kept.verify()
    ]


def _state(kept: Evidence) -> tuple[list[str], list[tuple[str, bytes, int]]]:
    """The lake's tables and every tree file but the catalog, with their bytes and times."""
    rows = [str(kept.lake.query(f"SELECT * FROM lake.{table} ORDER BY ALL")) for table in _TABLES]
    tree = [
        (path.as_posix(), path.read_bytes(), path.stat().st_mtime_ns)
        for path in kept.root.rglob("*")
        if path.is_file() and "sqlite" not in path.name
    ]
    return rows, tree


def _assert_judged(kept: Evidence, source: Path, *, condition: str, fault: str) -> None:
    """Every reader agrees on how the lake holds the object, and none of them repairs it."""
    [item] = kept.inventory([source])
    assert (item.lake, item.evictable) == (condition, condition == "intact")
    assert kept.recall(hashlib.sha256(_PAYLOAD).hexdigest()) == (
        _PAYLOAD if condition == "intact" else None
    )
    assert _faults(kept) == ([("evidence", source.name, fault)] if fault else [])


def _assert_custody(kept: Evidence, source: Path, *, whole: bool) -> None:
    """The tree copy goes only for a whole object; any other waits for ingest to repair it."""
    named = {"adopted.bin": hashlib.sha256(_PAYLOAD).hexdigest()}
    if not whole:
        with pytest.raises(MissionError, match="keeps no object"):
            kept.adopt(named)
        assert kept.evict(kept.finalize([source])) == []
        assert kept.ingest([source]).objects == 1
        assert kept.ingest([source]).objects == 0
        assert _faults(kept) == []
        assert kept.lake.query("SELECT count(*) FROM lake.blobs") == [(3,)]
    assert kept.adopt(named) == 1
    assert kept.evict(kept.finalize([source])) == [source]


@pytest.mark.parametrize(("statements", "condition", "fault"), _DAMAGES)
@_WINDOWS
def test_a_damaged_object_is_judged_alike_by_every_reader_and_only_ingest_repairs_it(
    kept: Evidence,
    stored: Callable[[str, bytes], Path],
    harm: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    statements: list[str],
    condition: str,
    fault: str,
    staged_bytes: int,
) -> None:
    source = stored("chunks.bin", _PAYLOAD)
    harm(*statements)
    monkeypatch.setattr(Blobs, "STAGED_BYTES", staged_bytes)
    _assert_judged(kept, source, condition=condition, fault=fault)
    _assert_custody(kept, source, whole=condition == "intact")
    assert kept.materialize([source]) == [source]
    assert source.read_bytes() == _PAYLOAD


@pytest.mark.parametrize(
    ("damage", "faults"),
    [
        pytest.param("SELECT 1", [], id="intact"),
        pytest.param(
            "UPDATE lake.blobs SET bytes = 'x'::BLOB WHERE sha256 = '{digest}'",
            [("evidence", "many/damaged.bin", "the lake holds 3 of its 12 bytes")],
            id="damaged",
        ),
    ],
)
def test_verify_checks_chunks_kept_before_checksums_by_digest_and_records_none(
    kept: Evidence,
    stored: Callable[[str, bytes], Path],
    harm: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    faults: list[tuple[str, str, str]],
) -> None:
    stored("many/one.bin", b"one object")
    stored("many/damaged.bin", _PAYLOAD)
    harm("DELETE FROM lake.checksums", damage.format(digest=hashlib.sha256(_PAYLOAD).hexdigest()))
    monkeypatch.setattr(keeping_module, "WINDOW_BYTES", 1)
    assert _faults(kept) == faults == _faults(kept)
    assert kept.lake.query("SELECT count(*) FROM lake.checksums") == [(0,)]


@pytest.fixture
def standings(
    stored: Callable[[str, bytes], Path],
    harm: Callable[..., None],
    digest_of: Callable[[bytes], str],
) -> Path:
    """A folder of four indexed files: one left alone, one edited, one deleted, one the lake lost.

    The lake keeps them without checksums, as it did what it kept before checksums.
    """
    changed, gone = stored("t/changed.bin", b"was"), stored("t/gone.bin", b"gone")
    stored("t/same.bin", b"same")
    stored("t/lost.bin", b"lost")
    changed.write_bytes(b"edited")
    gone.unlink()
    harm(
        f"DELETE FROM lake.blobs WHERE sha256 = '{digest_of(b'lost')}'",
        "DELETE FROM lake.checksums",
    )
    return changed.parent


def test_inventory_tells_every_standing_apart_and_changes_nothing(
    kept: Evidence, standings: Path, digest_of: Callable[[bytes], str]
) -> None:
    before = _state(kept)
    items = {item.path: item for item in kept.inventory([standings])}
    assert {path: (item.local, item.lake, item.evictable) for path, item in items.items()} == {
        "t/same.bin": ("identical", "intact", True),
        "t/changed.bin": ("changed", "intact", False),
        "t/gone.bin": ("missing", "intact", False),
        "t/lost.bin": ("identical", "missing", False),
    }
    assert items["t/changed.bin"].local_sha256 == digest_of(b"edited")
    assert len(kept.inventory([kept.root])) == len(items) and _state(kept) == before


def test_a_workspace_without_a_lake_reads_as_empty_and_creates_none(tmp_path: Path) -> None:
    bare = Evidence(Lake(root=tmp_path / "bare", home=tmp_path / "bare" / ".mb", served=""))
    assert bare.inventory([bare.root]) == [] and bare.verify() == [] and bare.indexed() == []
    assert bare.recall("0" * 64) is None and list(bare.recalled({"0" * 64: 1})) == []
    assert not (bare.root / ".mb").exists()


@pytest.mark.parametrize("moment", ["before", "after"])
def test_a_repair_that_fails_leaves_the_copy_and_the_rows_as_they_were(
    kept: Evidence,
    stored: Callable[[str, bytes], Path],
    harm: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    moment: str,
) -> None:
    source = stored("last.bin", b"last intact copy")
    harm("UPDATE lake.blobs SET bytes = 'bad'::BLOB")
    stage = Blobs.stage

    def interrupt(
        self: Blobs, connection: duckdb.DuckDBPyConnection, objects: Mapping[str, bytes | Path]
    ) -> int:
        if moment == "after":
            stage(self, connection, objects)
        raise RuntimeError("publication interrupted")

    monkeypatch.setattr(Blobs, "stage", interrupt)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        kept.ingest([source])
    assert source.read_bytes() == b"last intact copy"
    assert kept.lake.query("SELECT bytes FROM lake.blobs") == [(b"bad",)] * 4


def test_evict_deletes_only_what_the_lake_keeps_intact(
    keeper: Evidence, evidence: Path, pinned: str, files_of: Callable[[Path], dict[str, bytes]]
) -> None:
    keeper.ingest([evidence])
    (evidence / "noise.bin").write_bytes(b"changed after ingest")
    (evidence / "late.txt").write_bytes(b"never ingested")
    evicted = keeper.evict(keeper.finalize([evidence]))
    assert {path.name for path in evicted} == {"empty.txt", _PART, pinned}
    assert sorted(files_of(evidence)) == ["late.txt", "noise.bin"]
    keeper.materialize([evidence / "receipts"])
    assert (evidence / "receipts" / "run=run1" / _PART).is_file()
