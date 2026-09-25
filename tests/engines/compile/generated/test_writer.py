from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from filelock import FileLock

from mainboard import MissionError
from mainboard.engines.compile.generated import Writer


@pytest.fixture
def writer(tmp_path: Path) -> Iterator[Writer]:
    """A writer whose sync lock is held for the whole test."""
    lock = FileLock(tmp_path / ".sync.lock")
    with lock:
        yield Writer(lock)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda writer, path: writer.held(), id="the-guard-itself"),
        pytest.param(lambda writer, path: writer.write(path, "text\n"), id="a-write"),
        pytest.param(lambda writer, path: writer.remove(path), id="a-removal"),
    ],
)
def test_no_edit_survives_the_release_of_the_sync_lock(
    edit: Callable[[Writer, Path], None], tmp_path: Path
) -> None:
    writer = Writer(FileLock(tmp_path / ".sync.lock"))
    with pytest.raises(MissionError, match="no longer held"):
        edit(writer, tmp_path / "pixi.toml")


def test_new_bytes_land_as_a_fresh_inode_and_unchanged_ones_are_not_rewritten(
    writer: Writer, tmp_path: Path
) -> None:
    path = tmp_path / "job.sh"
    payload = b"#!/bin/sh\r\n# binary: \xff\r\n"
    writer.write(path, "first\n")
    first = path.stat().st_ino
    writer.write(path, payload)
    assert path.read_bytes() == payload
    assert path.stat().st_ino != first
    second = path.stat().st_ino
    writer.write(path, payload)
    assert path.stat().st_ino == second
    writer.write(path, "text\r\n")
    assert path.read_bytes() == b"text\r\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_windows_generated_files_keep_the_directorys_inherited_acl(
    writer: Writer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The atomic sibling keeps inheritance; POSIX modes never sever its Windows DACL."""
    chmod_calls: list[tuple[int, int]] = []
    replacements: list[tuple[Path, Path]] = []
    replace = Path.replace

    def record_replace(source: Path, target: Path) -> Path:
        replacements.append((source, target))
        return replace(source, target)

    monkeypatch.setattr(
        "mainboard.engines.compile.generated.writer.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        "mainboard.engines.compile.generated.writer.os.fchmod",
        lambda descriptor, mode: chmod_calls.append((descriptor, mode)),
    )
    monkeypatch.setattr(Path, "replace", record_replace)
    writer.write(tmp_path / "state.toml", "[envs]\n")

    assert chmod_calls == []
    [(staged, target)] = replacements
    assert staged.parent == target.parent == tmp_path


def test_remove_drops_a_generated_file_the_manifest_no_longer_asks_for(
    writer: Writer, tmp_path: Path
) -> None:
    target = tmp_path / "package.json"
    target.write_text("{}")
    writer.remove(target)
    writer.remove(tmp_path / "never-existed.json")
    assert not target.exists()
