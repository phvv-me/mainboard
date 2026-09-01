from collections.abc import Callable
from pathlib import Path

import pytest
from filelock import FileLock

from mainboard import MissionError
from mainboard.engines.compile.generated import Writer


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
    """A writer stashed past its block fails loudly.

    Failing beats racing whoever holds the lock now.
    """
    writer = Writer(FileLock(tmp_path / ".sync.lock"))
    with pytest.raises(MissionError, match="no longer held"):
        edit(writer, tmp_path / "pixi.toml")


def test_a_file_is_replaced_only_once_its_complete_contents_reach_disk(tmp_path: Path) -> None:
    """New content lands as a fresh inode, and unchanged content is not rewritten at all."""
    lock = FileLock(tmp_path / ".sync.lock")
    target = tmp_path / "pixi.toml"
    with lock:
        writer = Writer(lock)
        writer.write(target, "first\n")
        first = target.stat().st_ino

        writer.write(target, "second\n")
        assert target.read_text() == "second\n"
        assert target.stat().st_ino != first
        second = target.stat().st_ino

        writer.write(target, "second\n")
        assert target.stat().st_ino == second


def test_windows_generated_files_keep_the_directorys_inherited_acl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    lock = FileLock(tmp_path / ".sync.lock")
    with lock:
        Writer(lock).write(tmp_path / "state.toml", "[envs]\n")

    assert chmod_calls == []
    assert len(replacements) == 1
    staged, target = replacements[0]
    assert staged.parent == target.parent == tmp_path


def test_remove_drops_a_generated_file_the_manifest_no_longer_asks_for(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / ".sync.lock")
    target = tmp_path / "package.json"
    target.write_text("{}")
    with lock:
        Writer(lock).remove(target)
        Writer(lock).remove(tmp_path / "never-existed.json")
    assert not target.exists()
