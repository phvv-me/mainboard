"""A transient folder lands where its workspace keeps such trees, belongs to the process that made
it, and a dead process's is cleared."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from mainboard import MissionError, Project
from mainboard.core.constants.units import DISK_FLOOR
from mainboard.upkeep.disk.transient import Transient

from ...builders import survivors
from ...fakes import disks
from .builders import streaming_into, workspace_at

# A process id no process has.
_DEAD = 999_999_999


@pytest.fixture
def transient(workspace: Path) -> Transient:
    """The transient folders of a workspace that has made none yet."""
    return Transient(workspace)


def test_reaping_clears_the_folders_of_dead_processes_and_keeps_every_other_folder(
    transient: Transient,
) -> None:
    dead, copy = transient.place / f"collect-{_DEAD}-a", transient.apart / f"fix-{_DEAD}-b"
    kept = [
        transient.place / name
        for name in (f"collect-{os.getpid()}-c", "collect-x", f"other-{_DEAD}-d")
    ]
    for folder in (dead, copy, *kept):
        folder.mkdir(parents=True)

    assert sorted(transient.reap()) == sorted([dead, copy])

    assert [folder.exists() for folder in (dead, copy, *kept)] == [False, False, True, True, True]


def test_reaping_ends_the_stream_a_killed_process_left_writing_into_its_folder(
    transient: Transient,
) -> None:
    dead = transient.place / f"collect-{_DEAD}-a"
    with streaming_into(dead) as stream:
        transient.reap()

        assert (dead.exists(), survivors([stream.pid])) == (False, [])


def test_reaping_leaves_a_stream_writing_into_the_folder_of_a_live_process_alone(
    transient: Transient,
) -> None:
    live = transient.place / f"collect-{os.getpid()}-a"
    with streaming_into(live) as stream:
        assert transient.reap() == []

        assert (live.exists(), stream.poll()) == (True, None)


def test_reaping_before_any_folder_was_made_clears_nothing(transient: Transient) -> None:
    assert (transient.reap(), transient.place.exists()) == ([], False)


def test_a_fresh_folder_is_named_for_this_process_and_gone_with_its_writer(
    transient: Transient,
) -> None:
    dead = transient.place / f"collect-{_DEAD}-a"
    dead.mkdir(parents=True)

    with transient.staging(doing="collecting") as folder:
        inside = (folder.parent, folder.name.startswith(f"collect-{os.getpid()}-"), dead.exists())

    assert (inside, folder.exists()) == ((transient.place, True, False), False)


def test_a_copy_whose_bytes_would_cross_the_disk_floor_is_refused_and_never_made(
    transient: Transient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", disks(DISK_FLOOR + 10))

    with pytest.raises(MissionError, match="copying x would leave .* below the 30 GiB"):
        transient.copy_root(need=11, doing="copying x")

    assert list(transient.apart.glob(f"fix-{os.getpid()}-*")) == []


def test_a_copy_tools_run_in_lands_outside_a_workspace_that_keeps_its_trees_inside(
    transient: Transient, workspace: Path
) -> None:
    inside = transient.place.is_relative_to(workspace.resolve())

    assert (inside, transient.apart) == (True, Path(tempfile.gettempdir()).resolve())


def test_the_center_keeps_its_trees_where_the_manifest_names_and_a_mirror_ignores_it(
    tmp_path: Path,
) -> None:
    big = tmp_path / "big"
    center = workspace_at(tmp_path / "center", tmp=big)
    mirror = workspace_at(tmp_path / "mirror", tmp=big)
    (mirror / ".git").rmdir()

    assert (Transient(center).place, Transient(center).apart) == (big.resolve(), big.resolve())
    assert Transient(mirror).place == (Project().out(mirror) / "tmp").resolve()


def test_a_generated_tmp_linked_to_another_disk_is_followed_there(tmp_path: Path) -> None:
    root, disk = workspace_at(tmp_path / "w"), tmp_path / "ssd"
    disk.mkdir()
    Project().out(root).mkdir()
    (Project().out(root) / "tmp").symlink_to(disk)

    assert (Transient(root).place, Transient(root).apart) == (disk.resolve(), disk.resolve())
