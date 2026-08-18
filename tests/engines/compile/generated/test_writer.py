from typing import TYPE_CHECKING

import pytest
from filelock import FileLock

from mainboard import MissionError
from mainboard.engines.compile.generated import Writer

if TYPE_CHECKING:
    from pathlib import Path


def test_held_raises_once_the_lock_is_released(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / ".sync.lock")
    writer = Writer(lock)
    with pytest.raises(MissionError, match="no longer held"):
        writer.held()


def test_write_creates_and_replaces_a_file_atomically(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / ".sync.lock")
    target = tmp_path / "pixi.toml"
    with lock:
        writer = Writer(lock)
        writer.write(target, "first\n")
        assert target.read_text() == "first\n"
        first_inode = target.stat().st_ino

        writer.write(target, "second\n")
        assert target.read_text() == "second\n"
        assert target.stat().st_ino != first_inode


def test_write_is_a_noop_when_the_content_is_unchanged(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / ".sync.lock")
    target = tmp_path / "pixi.toml"
    with lock:
        writer = Writer(lock)
        writer.write(target, "same\n")
        inode = target.stat().st_ino

        writer.write(target, "same\n")
        assert target.stat().st_ino == inode


def test_write_refuses_once_the_lock_is_released(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / ".sync.lock")
    writer = Writer(lock)
    with pytest.raises(MissionError, match="no longer held"):
        writer.write(tmp_path / "pixi.toml", "text\n")


def test_remove_unlinks_an_existing_file(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / ".sync.lock")
    target = tmp_path / "package.json"
    target.write_text("{}")
    with lock:
        Writer(lock).remove(target)
    assert not target.exists()


def test_remove_tolerates_a_file_that_is_already_gone(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / ".sync.lock")
    with lock:
        Writer(lock).remove(tmp_path / "never-existed.json")
