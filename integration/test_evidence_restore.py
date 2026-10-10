"""Evidence restore and ingest: neither leaves its root, follows a link or overwrites a file."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import IO

import duckdb
import pytest

from mainboard.core import MissionError
from mainboard.state import Evidence, Lake
from mainboard.state import blobs as blobs_module
from mainboard.state import evidence as evidence_module
from mainboard.state.blobs import Audit


@pytest.fixture
def kept(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Evidence:
    """The workspace's evidence, in chunks of four bytes."""
    monkeypatch.setattr(blobs_module, "CHUNK_BYTES", 4)
    return Evidence(Lake.at(workspace))


def _stored(kept: Evidence, name: str, payload: bytes) -> Path:
    """A file of the workspace holding `payload`, ingested."""
    path = kept.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    kept.ingest([path])
    return path


def _after_the_lake_is_read(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch, act: Callable[[], None]
) -> None:
    """Have `act` happen once an object is read out of the lake, as a racing process would."""
    inspect = kept.blobs.inspect

    def read(connection: duckdb.DuckDBPyConnection, digest: str, sink: IO[bytes]) -> Audit:
        audit = inspect(connection, digest, sink)
        act()
        return audit

    monkeypatch.setattr(kept.blobs, "inspect", read)


def test_materialize_refuses_a_changed_file_lists_digests_and_writes_nothing(
    kept: Evidence,
) -> None:
    changed = _stored(kept, "t/changed.txt", b"old settled result")
    gone = _stored(kept, "t/gone.txt", b"settled and then evicted")
    gone.unlink()
    changed.write_bytes(b"new uncollected result")
    with pytest.raises(MissionError) as refused:
        kept.materialize([kept.root / "t"])
    digests = [
        kept.indexed("t/changed.txt")[0].sha256,
        hashlib.sha256(changed.read_bytes()).hexdigest(),
    ]
    assert all(text in str(refused.value) for text in [str(changed), *digests])
    assert changed.read_bytes() == b"new uncollected result" and not gone.exists()


def test_materialize_writes_only_what_is_missing_into_any_root(
    kept: Evidence, tmp_path: Path
) -> None:
    same, gone = _stored(kept, "t/same.txt", b"same"), _stored(kept, "t/gone.txt", b"gone")
    gone.unlink()
    before = same.stat()
    assert kept.materialize([kept.root / "t"]) == [gone] and gone.read_bytes() == b"gone"
    assert (same.stat().st_ino, same.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    elsewhere = tmp_path / "restore" / "here"
    assert kept.materialize([same], into=elsewhere) == [elsewhere / "t/same.txt"]
    assert (elsewhere / "t/same.txt").read_bytes() == b"same"


@pytest.mark.parametrize("link", ["folder", "file"])
def test_materialize_refuses_a_link_in_the_way(
    kept: Evidence, tmp_path_factory: pytest.TempPathFactory, link: str
) -> None:
    source = _stored(kept, "sub/result.txt", b"archived")
    restore, outside = tmp_path_factory.mktemp("restore"), tmp_path_factory.mktemp("outside")
    (outside / "result.txt").write_bytes(b"unrelated file")
    if link == "folder":
        (restore / "sub").symlink_to(outside, target_is_directory=True)
    else:
        (restore / "sub").mkdir()
        (restore / "sub" / "result.txt").symlink_to(outside / "result.txt")
    with pytest.raises(MissionError, match="unsafe evidence"):
        kept.materialize([source], into=restore)
    assert (outside / "result.txt").read_bytes() == b"unrelated file"


def test_a_folder_swapped_for_a_link_during_a_restore_is_never_followed(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    source = _stored(kept, "sub/result.bin", b"archived")
    restore, outside = tmp_path_factory.mktemp("restore"), tmp_path_factory.mktemp("outside")
    (restore / "sub").mkdir()

    def swap() -> None:
        (restore / "sub").rename(restore / "original")
        (restore / "sub").symlink_to(outside, target_is_directory=True)

    _after_the_lake_is_read(kept, monkeypatch, swap)
    kept.materialize([source], into=restore)
    assert list(outside.iterdir()) == []
    assert (restore / "original" / "result.bin").read_bytes() == b"archived"


def test_a_file_appearing_during_a_restore_is_not_overwritten(
    kept: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _stored(kept, "result.bin", b"archived")
    source.unlink()
    _after_the_lake_is_read(kept, monkeypatch, lambda: source.write_bytes(b"written meanwhile"))
    with pytest.raises(MissionError, match="refusing overwrite"):
        kept.materialize([source])
    assert source.read_bytes() == b"written meanwhile" and not list(kept.root.glob(".mb-*"))


@pytest.mark.parametrize(
    "shape", ["a file link", "a folder holding a link", "a route through one"]
)
def test_ingest_refuses_a_link_before_keeping_anything(
    kept: Evidence, tmp_path_factory: pytest.TempPathFactory, shape: str
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "unrelated.txt").write_bytes(b"outside workspace bytes")
    folder = kept.root / "evidence"
    folder.mkdir()
    (folder / "fine.txt").write_bytes(b"fine")
    (folder / "linked").symlink_to(outside / "unrelated.txt" if "file" in shape else outside)
    named = {
        "a file link": folder / "linked",
        "a route through one": folder / "linked/unrelated.txt",
    }
    with pytest.raises(MissionError, match="is a link"):
        kept.ingest([named.get(shape, folder)])
    assert kept.indexed() == [] and (outside / "unrelated.txt").read_bytes().startswith(b"outside")


def test_ingest_ignores_the_tools_own_aside(kept: Evidence) -> None:
    folder = kept.root / "evidence"
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
    monkeypatch.setattr(evidence_module, "hashlib", racing)
    with pytest.raises(MissionError, match="changed while reading"):
        kept.ingest([source])
    assert kept.indexed() == []


def test_the_cache_replaces_a_stale_file_in_place(kept: Evidence) -> None:
    source = _stored(kept, "tree/result.bin", b"first")
    first = kept.cached(kept.indexed("tree")[0])
    assert first == kept.cache / "tree/result.bin" and first.read_bytes() == b"first"
    source.write_bytes(b"second")
    kept.ingest([source])
    assert kept.cached(kept.indexed("tree")[0]) == first and first.read_bytes() == b"second"


def test_the_cache_follows_no_link(
    kept: Evidence, tmp_path_factory: pytest.TempPathFactory
) -> None:
    _stored(kept, "tree/result.bin", b"cached")
    outside = tmp_path_factory.mktemp("outside")
    kept.cache.mkdir(parents=True)
    (kept.cache / "tree").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MissionError, match="unsafe evidence path"):
        kept.cached(kept.indexed("tree")[0])
    assert list(outside.iterdir()) == []
