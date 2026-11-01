"""The doctor's disk rows: the free room against the floor where the workspace writes, and what
dead writers left, each named with the plain command that removes it."""

import os
import shutil
import subprocess
import time
from pathlib import Path
from shlex import join

import pytest

from mainboard import Board
from mainboard.core import Verdict
from mainboard.core.constants.units import DISK_FLOOR, SECONDS_PER_DAY
from mainboard.upkeep.disk import space as space_module
from mainboard.upkeep.disk.space import Space

from ...fakes import disks, exiting

# A process id no process has.
_DEAD = 999_999_999


def _aged(path: Path) -> Path:
    """`path`, last touched two days ago."""
    then = time.time() - 2 * SECONDS_PER_DAY
    os.utime(path, (then, then))
    return path


def _left(path: Path) -> Path:
    """A file of ten bytes at `path`, last touched two days ago."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0123456789")
    return _aged(path)


@pytest.mark.parametrize(
    ("free", "verdict"),
    [
        pytest.param(2 * DISK_FLOOR, Verdict.PASS, id="twice-the-floor-free"),
        pytest.param(2 * DISK_FLOOR - 1, Verdict.WARN, id="under-twice-the-floor"),
        pytest.param(DISK_FLOOR - 1, Verdict.FAIL, id="under-the-floor"),
    ],
)
def test_the_disk_row_warns_under_twice_the_floor_and_fails_under_it(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, free: int, verdict: Verdict
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", disks(free))
    monkeypatch.setattr(subprocess, "run", exiting(0))

    assert Space(Board(workspace)).disk().verdict is verdict


def test_a_tight_disk_names_its_biggest_folders_with_their_sizes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / ".mb" / "envs").mkdir(parents=True)
    out = workspace.resolve() / ".mb"
    sized = f"2097152\t{out / 'envs'}\n1048576\t{out / 'tmp'}\n"
    monkeypatch.setattr(shutil, "disk_usage", disks(DISK_FLOOR))
    monkeypatch.setattr(subprocess, "run", exiting(0, sized))

    row = Space(Board(workspace)).disk()

    assert row.detail.endswith("biggest: .mb/envs 2.0 GiB, .mb/tmp 1.0 GiB")


def test_temporary_files_git_left_dying_are_named_with_the_command_removing_them(
    workspace: Path,
) -> None:
    git, sub = workspace / ".git", workspace / ".git" / "modules" / "packages" / "sub"
    for repository in (git, sub):
        _left(repository / "HEAD")
    _left(git / "objects" / "pack" / "tmp_pack_old")
    _left(sub / "objects" / "ab" / "tmp_obj_old")
    (git / "objects" / "pack" / "tmp_pack_writing").write_bytes(b"live")

    row = Space(Board(workspace)).git()

    assert row is not None and row.detail.startswith("2 temporary file(s)")
    assert row.fix.endswith("-path '*/objects/*' -name 'tmp_*' -type f -mmin +60 -delete")


def test_the_roots_of_finished_gate_runs_are_named_with_the_command_removing_them(
    workspace: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "system-tmp"
    dead, old = folder / f"mb-gate.{_DEAD}.abc", folder / "mb-gate.ancient"
    for root in (dead, old, *(folder / name for name in ("mb-gate.fresh", "mb-gate-report.x"))):
        root.mkdir(parents=True)
    (folder / f"mb-gate.{os.getpid()}.live").mkdir()
    _aged(old)

    row = Space(Board(workspace), gate_folder=folder).gates()

    assert row is not None and row.fix == join(["rm", "-rf", str(dead), str(old)])


def test_evidence_past_the_limit_names_its_biggest_folder_and_the_commands_that_clear_it(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "research" / "a" / "datasets").mkdir(parents=True)
    (workspace / "research" / "a" / "datasets" / "rows.bin").write_bytes(b"x" * 65536)
    (workspace / "scratch").mkdir()
    (workspace / "scratch" / "loose.parquet").write_bytes(b"y" * 1024)
    monkeypatch.setattr(space_module, "_EVIDENCE_LIMIT", 1024)

    row = Space(Board(workspace)).evidence()

    assert row is not None and row.verdict is Verdict.WARN
    assert row.fix == (
        "mb data lake ingest research/a/datasets && mb data lake evict research/a/datasets"
    )
    assert "research/a/datasets" in row.detail and "scratch/loose.parquet" in row.detail


def test_evidence_under_the_limit_passes_and_none_is_searched_in_builds_or_environments(
    workspace: Path,
) -> None:
    (workspace / "target" / "datasets").mkdir(parents=True)
    (workspace / "target" / "datasets" / "rows.bin").write_bytes(b"x" * 4096)
    (workspace / "src").mkdir()
    (workspace / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")

    row = Space(Board(workspace)).evidence()

    assert row is not None and (row.verdict, row.detail) == (Verdict.PASS, "0.0 GiB in the tree")
