"""Evidence custody: a tree copy leaves only on proof, however the lake or the tree changes."""

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO

import duckdb
import pytest

from mainboard import MissionError
from mainboard.state import Evidence, EvidenceTree, Lake
from mainboard.state.evidence import keeping as keeping_module
from mainboard.state.storage import blobs as blobs_module
from mainboard.state.storage.blobs import Audit, Blobs
from mainboard.state.storage.owned import OwnedFile
from mainboard.trials import Artifact
from mainboard.trials.store.archive import ParquetArtifacts

_PAYLOAD = b"aaaabbbbcccc"
_WHOLE = {"intact", "legacy", "duplicate"}
_DAMAGE = {
    "intact": ("SELECT 1",),
    "legacy": ("DELETE FROM lake.checksums",),
    "duplicate": ("INSERT INTO lake.blobs SELECT * FROM lake.blobs",),
    "missing": ("DELETE FROM lake.blobs WHERE ordinal = 1",),
    "reordered": ("UPDATE lake.blobs SET ordinal = 2 - ordinal",),
    "conflicting": (
        "INSERT INTO lake.blobs SELECT sha256, 'bad'::BLOB, 0 FROM lake.blobs LIMIT 1",
    ),
    "truncated": ("UPDATE lake.blobs SET bytes = 'c'::BLOB WHERE ordinal = 2",),
    "lost cell": ("UPDATE lake.blobs SET bytes = NULL WHERE ordinal = 2",),
    "unchecksummed": (
        "DELETE FROM lake.checksums",
        "UPDATE lake.blobs SET bytes = 'xxxx'::BLOB",
    ),
    "extra chunk": (
        "INSERT INTO lake.blobs SELECT DISTINCT sha256, 'zzzz'::BLOB, 3 FROM lake.blobs",
    ),
}
# A window of no bytes reads every object a chunk at a time, as one larger than it would be.
_WINDOWS = {"in one query": 256 << 20, "a chunk at a time": 0}


@pytest.fixture
def kept(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Evidence:
    """The center workspace's evidence, in chunks of four bytes."""
    (workspace / ".git").mkdir()
    monkeypatch.setattr(blobs_module, "_CHUNK_BYTES", 4)
    evidence = Evidence(Lake.at(workspace))
    monkeypatch.setattr(EvidenceTree, "keepers", property(lambda tree: [evidence]))
    return evidence


@pytest.fixture(params=_DAMAGE)
def damage(request: pytest.FixtureRequest) -> str:
    """How the lake's rows of a file are damaged."""
    return request.param


@pytest.fixture(params=_WINDOWS)
def damaged(
    kept: Evidence, damage: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Path:
    """A three-chunk file kept in the lake whose rows are then damaged, read in a window."""
    source = _stored(kept, "chunks.bin", _PAYLOAD)
    _damage(kept, *_DAMAGE[damage])
    monkeypatch.setattr(blobs_module.Blobs, "STAGED_BYTES", _WINDOWS[request.param])
    return source


def _stored(kept: Evidence, name: str, payload: bytes) -> Path:
    """A file of the workspace holding `payload`, ingested."""
    path = kept.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    kept.ingest([path])
    return path


def _damage(kept: Evidence, *statements: str) -> None:
    """Change the lake's rows behind its writers' back, the way a failing disk would."""
    with kept.lake.open(write=True) as connection:
        for statement in statements:
            connection.execute(statement)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_inventory_agrees_with_readback(kept: Evidence, damage: str, damaged: Path) -> None:
    whole = damage in _WHOLE
    [item] = kept.inventory([damaged])
    assert item.lake == ("intact" if whole else "corrupt") and item.evictable is whole
    assert kept.recall(_digest(_PAYLOAD)) == (_PAYLOAD if whole else None)
    assert (kept.verify() == []) is whole


def test_only_a_whole_object_lets_its_copy_go_and_ingest_repairs_the_rest(
    kept: Evidence, damage: str, damaged: Path
) -> None:
    whole = damage in _WHOLE
    assert kept.evict(kept.finalize([damaged])) == ([damaged] if whole else [])
    if not whole:
        assert kept.ingest([damaged]).objects == 1 and kept.verify() == []
        assert kept.evict(kept.finalize([damaged])) == [damaged]
    assert kept.materialize([damaged]) == [damaged]
    assert damaged.read_bytes() == _PAYLOAD


def test_corrupt_custody_keeps_the_last_copy_and_reingest_repairs(kept: Evidence) -> None:
    source = _stored(kept, "evidence.bin", b"irreplaceable evidence")
    digest = _digest(source.read_bytes())
    _damage(kept, "UPDATE lake.blobs SET bytes = 'corrupt'::BLOB")
    assert kept.recall(digest) is None and kept.evict(kept.finalize([source])) == []
    with pytest.raises(MissionError, match="keeps no object"):
        kept.adopt({"adopted.bin": digest})
    assert kept.ingest([source]).objects == 1 and kept.ingest([source]).objects == 0
    assert kept.recall(digest) == source.read_bytes() and kept.adopt({"adopted.bin": digest}) == 1
    assert kept.lake.query("SELECT count(*) FROM lake.blobs") == [(6,)]


def test_verify_reads_the_index_in_windows_and_records_no_checksum(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stored(kept, "many/one.bin", b"one object")
    _stored(kept, "many/damaged.bin", _PAYLOAD)
    damage = f"UPDATE lake.blobs SET bytes = 'x'::BLOB WHERE sha256 = '{_digest(_PAYLOAD)}'"
    _damage(kept, "DELETE FROM lake.checksums", damage)
    monkeypatch.setattr(keeping_module, "WINDOW_BYTES", 1)
    [finding] = kept.verify()
    assert "many/damaged.bin" in finding.detail and "3 of its 12 bytes" in finding.detail
    assert kept.lake.query("SELECT count(*) FROM lake.checksums") == [(0,)]


def test_inventory_changes_neither_the_tree_nor_the_lake(kept: Evidence, tmp_path: Path) -> None:
    bare = Evidence(Lake(root=tmp_path / "bare", home=tmp_path / "bare" / ".mb", served=""))
    assert bare.inventory([bare.root]) == [] and not (bare.root / ".mb").exists()
    source = _stored(kept, "evidence.bin", _PAYLOAD)
    _damage(kept, "DELETE FROM lake.checksums")

    def state() -> list[object]:
        tables = ["blobs", "checksums", "evidence_log", "imports", "schema_log"]
        rows = [kept.lake.query(f"SELECT * FROM lake.{table} ORDER BY ALL") for table in tables]
        tree = [(p, p.read_bytes()) for p in kept.root.rglob("*") if p.is_file()]
        return [*rows, [t for t in tree if "sqlite" not in t[0].name], source.stat().st_mtime_ns]

    before = state()
    [item] = kept.inventory([source])
    assert item.evictable and item.lake == "intact"
    assert kept.inventory([kept.root]) == [item] and kept.verify() == [] and state() == before


def test_inventory_tells_every_standing_apart(kept: Evidence) -> None:
    changed, gone = _stored(kept, "t/changed.bin", b"was"), _stored(kept, "t/gone.bin", b"gone")
    _stored(kept, "t/same.bin", b"same")
    _stored(kept, "t/lost.bin", b"lost")
    changed.write_bytes(b"edited")
    gone.unlink()
    _damage(kept, f"DELETE FROM lake.blobs WHERE sha256 = '{_digest(b'lost')}'")
    items = {item.path: item for item in kept.inventory([kept.root / "t"])}
    assert {k: (v.local, v.lake, v.evictable) for k, v in items.items()} == {
        "t/same.bin": ("identical", "intact", True),
        "t/changed.bin": ("changed", "intact", False),
        "t/gone.bin": ("missing", "intact", False),
        "t/lost.bin": ("identical", "missing", False),
    }
    assert items["t/changed.bin"].local_sha256 == _digest(b"edited")


def test_a_file_replaced_after_it_was_closed_is_put_back(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _stored(kept, "result.bin", b"finalized")
    digest = OwnedFile.digest

    def replace(self: OwnedFile) -> str:
        seen = digest(self)
        (kept.root / "replacement").write_bytes(b"successor")
        (kept.root / "replacement").replace(source)
        return seen

    monkeypatch.setattr(OwnedFile, "digest", replace)
    assert kept.evict(kept.finalize([source])) == []
    assert source.read_bytes() == b"successor" and not list(kept.root.rglob(".mb-*"))


def test_a_successor_written_while_the_lake_is_checked_is_never_unlinked(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _stored(kept, "result.bin", b"finalized")
    inspect = kept.blobs.inspect

    def check(
        connection: duckdb.DuckDBPyConnection, digest: str, sink: IO[bytes] | None = None
    ) -> Audit:
        audit = inspect(connection, digest, sink)
        source.write_bytes(b"successor")
        return audit

    monkeypatch.setattr(kept.blobs, "inspect", check)
    assert kept.evict(kept.finalize([source])) == [source]
    assert (
        source.read_bytes() == b"successor" and kept.recall(_digest(b"finalized")) == b"finalized"
    )


def test_a_failure_with_a_successor_in_place_keeps_both_versions(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _stored(kept, "result.bin", b"finalized")

    def fail(
        connection: duckdb.DuckDBPyConnection, digest: str, sink: IO[bytes] | None = None
    ) -> Audit:
        source.write_bytes(b"successor")
        raise RuntimeError("custody interrupted")

    monkeypatch.setattr(kept.blobs, "inspect", fail)
    with pytest.raises(MissionError) as refused:
        kept.evict(kept.finalize([source]))
    [candidate] = kept.root.glob(".mb-custody-*/result.bin")
    assert candidate.read_bytes() == b"finalized" and source.read_bytes() == b"successor"
    assert all(_digest(text) in str(refused.value) for text in [b"finalized", b"successor"])


def test_a_failure_before_the_unlink_puts_the_copy_back(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _stored(kept, "final.bin", b"finalized")
    unlink = os.unlink
    interrupted: list[str] = []

    def interrupt(path: str, *, dir_fd: int | None = None) -> None:
        if path == source.name and dir_fd is not None and not interrupted:
            interrupted.append(path)
            raise RuntimeError("unlink interrupted")
        unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", interrupt)
    with pytest.raises(RuntimeError, match="unlink interrupted"):
        kept.evict(kept.finalize([source]))
    assert source.read_bytes() == b"finalized" and not list(kept.root.rglob(".mb-custody-*"))


@pytest.mark.parametrize("moment", ["before", "after"])
def test_a_repair_that_fails_leaves_the_copy_and_the_rows_as_they_were(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch, moment: str
) -> None:
    source = _stored(kept, "last.bin", b"last intact copy")
    _damage(kept, "UPDATE lake.blobs SET bytes = 'bad'::BLOB")
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


def test_a_closed_run_moves_into_the_lake_unless_a_link_hides_it(
    kept: Evidence, tmp_path_factory: pytest.TempPathFactory
) -> None:
    moved = kept.root / "run" / "receipt.bin"
    moved.parent.mkdir()
    moved.write_bytes(_PAYLOAD)
    volume = tmp_path_factory.mktemp("volume")
    (kept.root / "datasets").symlink_to(volume, target_is_directory=True)
    tree = EvidenceTree(kept.root)
    assert tree.keep([kept.root / "datasets" / "node.bin"]) is None
    assert tree.keep([moved.parent]) is not None
    assert not moved.parent.exists() and kept.recall(_digest(_PAYLOAD)) == _PAYLOAD
    assert list(volume.iterdir()) == [] and kept.indexed() == kept.indexed("run")


def test_a_closed_run_written_during_collection_stays_in_the_tree(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
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
    assert active.read_bytes() == b"after!" and kept.recall(_digest(b"before")) == b"before"


def test_the_parts_of_an_artifact_archive_stay_in_the_tree_for_readers_that_open_them(
    kept: Evidence,
) -> None:
    folder = kept.root / "datasets" / "experiments"
    member = folder / "run" / "table.bin"
    member.parent.mkdir(parents=True)
    member.write_bytes(_PAYLOAD)
    ParquetArtifacts(folder).pack([member])
    member.unlink()
    kept.ingest([folder])
    assert kept.evict(kept.finalize([folder])) == []
    reference = Artifact(path="run/table.bin", sha256=_digest(_PAYLOAD), size=len(_PAYLOAD))
    assert reference.read(folder) == _PAYLOAD
