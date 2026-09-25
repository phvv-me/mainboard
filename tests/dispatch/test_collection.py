"""Portable collection preserves evidence across retries, corruption, and racing writers."""

import os
from io import BytesIO, TextIOWrapper
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pytest

from mainboard.dispatch.collection.collector import Collector, KnownDigests
from mainboard.dispatch.collection.pack import _paths, pack


def test_collection_retries_and_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "local"
    root.mkdir()
    archive = tmp_path / "transfer.zip"
    with ZipFile(archive, "w") as packed:
        packed.writestr("data/one", b"original")
    collector = Collector(root)
    assert collector.merge(archive, path="data") == 1
    assert collector.merge(archive, path="data") == 0
    with ZipFile(archive, "w") as packed:
        packed.writestr("data/two", b"new")
        packed.writestr("data/one", b"changed")
    with pytest.raises(ValueError, match="conflicting"):
        collector.merge(archive, path="data")
    assert (root / "data/one").read_bytes() == b"original"
    assert not (root / "data/two").exists()


def test_incremental_collection_skips_equal_bytes_but_keeps_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local, remote = tmp_path / "local", tmp_path / "remote"
    for root in (local, remote):
        (root / "data").mkdir(parents=True)
        (root / "data/equal").write_bytes(b"already collected")
        (root / "data/changed").write_bytes(b"original")
    (remote / "data/changed").write_bytes(b"conflict")
    (remote / "data/new").write_bytes(b"new evidence")
    events = remote / "data/events"
    events.mkdir()
    event = b'{"offset":0,"payload":"progress"}\n'
    (events / "live.ndjson").write_bytes(event)
    collector = Collector(local)
    known = collector._known(PurePosixPath("data"))
    stream = BytesIO()
    stdout = TextIOWrapper(stream, encoding="utf-8")
    with monkeypatch.context() as changed:
        changed.setattr("sys.stdout", stdout)
        pack(str(remote), relative="data", known=known)
    with ZipFile(stream) as archive:
        assert "data/equal" not in archive.namelist()
        assert archive.read("data/changed") == b"conflict"
        assert archive.read("data/new") == b"new evidence"
        assert any("collected-" in name for name in archive.namelist())
    transfer = tmp_path / "transfer.zip"
    transfer.write_bytes(stream.getvalue())
    with pytest.raises(ValueError, match="conflicting"):
        collector.merge(transfer, path="data")
    assert (local / "data/changed").read_bytes() == b"original"
    assert not (local / "data/new").exists()


def test_a_published_file_is_hashed_once_until_its_stamp_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "data/object"
    evidence.parent.mkdir()
    evidence.write_bytes(b"published")
    collector = Collector(tmp_path)
    first = collector._known(PurePosixPath("data"))
    reads: list[Path] = []
    opened = Path.open

    def counted(self: Path, *args: str, **options: str):
        reads.append(self)
        return opened(self, *args, **options)

    monkeypatch.setattr(Path, "open", counted)
    assert collector._known(PurePosixPath("data")) == first
    assert evidence not in reads
    evidence.write_bytes(b"rewritten elsewhere")
    os.utime(evidence, ns=(1, 1))
    assert collector._known(PurePosixPath("data")) != first
    assert evidence in reads


def test_a_torn_digest_memory_is_rebuilt(tmp_path: Path) -> None:
    memory = tmp_path / "digests.json"
    memory.write_text("{", encoding="utf-8")
    assert KnownDigests(memory).held == {}


@pytest.mark.parametrize(
    "name",
    ["../escape", "C:/escape", "data/../escape", "elsewhere/file", "data/CON", "data/trailing."],
)
def test_collection_refuses_escaping_paths(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "transfer.zip"
    with ZipFile(archive, "w") as packed:
        packed.writestr(name, b"bad")
    with pytest.raises(ValueError):
        Collector(tmp_path).merge(archive, path="data")


def test_collection_preserves_a_racing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "transfer.zip"
    with ZipFile(archive, "w") as packed:
        packed.writestr("data/one", b"incoming")
    link = os.link

    def race(source: Path, target: Path) -> None:
        target.write_bytes(b"racing writer")
        link(source, target)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(ValueError, match="conflicting"):
        Collector(tmp_path).merge(archive, path="data")
    assert (tmp_path / "data/one").read_bytes() == b"racing writer"


def test_pack_snapshots_complete_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = tmp_path / "data/events"
    events.mkdir(parents=True)
    complete = b'{"offset":0,"payload":"hello"}\n'
    (events / "live.ndjson").write_bytes(complete + b'{"offset":')
    (events / "status.json").write_text('{"offset":999}')
    (events / "unfinished.tmp").write_bytes(b"temporary")
    stream = BytesIO()
    stdout = TextIOWrapper(stream, encoding="utf-8")
    with monkeypatch.context() as changed:
        changed.setattr("sys.stdout", stdout)
        pack(str(tmp_path), relative="data")
    with ZipFile(stream) as archive:
        assert archive.namelist() == [
            f"data/events/collected-{0:020d}-{len(complete):020d}.ndjson"
        ]
        assert archive.read(archive.namelist()[0]) == complete


def test_pack_refuses_linked_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "linked").symlink_to(tmp_path, target_is_directory=True)
    stdout = TextIOWrapper(BytesIO(), encoding="utf-8")
    with monkeypatch.context() as changed:
        changed.setattr("sys.stdout", stdout)
        with pytest.raises(ValueError, match="non-regular"):
            pack(str(tmp_path), relative="data")


def test_pack_excludes_unpublished_objects_before_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = tmp_path / "data/objects"
    objects.mkdir(parents=True)
    published = objects / ("a" * 64)
    published.write_bytes(b"published")
    vanished = [objects / "tmpabcdefgh", objects / "pending.tmp"]
    monkeypatch.setattr(
        "mainboard.dispatch.collection.pack._entries",
        lambda selected: iter([*vanished, published]),
    )
    assert list(_paths(tmp_path, "data")) == [published]
