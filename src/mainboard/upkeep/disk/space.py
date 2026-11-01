# The doctor's rows on disk space: the free room on each disk the workspace writes to, against the
# floor every large writer keeps (`Disk`), what dead writers left that only the owner removes, and
# the evidence still sitting in the tree that the lake should keep. Nothing here deletes: each
# leftover row names the plain command that does.

import os
import re
import subprocess
import time
from collections.abc import Collection, Iterable, Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from shlex import join, quote
from typing import TYPE_CHECKING

import psutil

from ...core import Disk, Project, Section, Verdict, abbreviated, captured
from ...core.constants.units import (
    BYTES_PER_GIBIBYTE,
    BYTES_PER_KIBIBYTE,
    DISK_FLOOR,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
)
from .transient import Transient

if TYPE_CHECKING:
    from ...fleet.board import Board

# How long `du` may take to size the folders of a tight disk before the row goes without sizes,
# and how many of the biggest it names.
_DU_SECONDS = 30
_BIGGEST = 3

# The folders that hold evidence (the lake keeps what is in them) and the folders a search for them
# never enters, since a build or an environment holds no evidence and is big to walk; and how much
# evidence the tree may hold before the row asks for it to go to the lake.
_EVIDENCE = frozenset({"datasets", "evidence"})
_UNWALKED = frozenset({".git", ".mb", ".pixi", ".venv", "node_modules", "target", ".build"})
_EVIDENCE_LIMIT = 5 * BYTES_PER_GIBIBYTE

# Where the gate (`integration/gates/run.sh`) roots a run, `mb-gate.<launcher pid>.<random>`; a
# root from a gate older than the pid in its name is `mb-gate.<random>`.
_GATES = Path("/tmp")
_GATE = re.compile(r"mb-gate\.(?:(\d+)\.)?\w+")


def _gib(size: int) -> str:
    return f"{size / BYTES_PER_GIBIBYTE:.1f} GiB"


def _git_dirs(folder: Path) -> Iterator[Path]:
    """`folder` when it is a git directory, then each submodule's beneath its `modules`."""
    if (folder / "HEAD").is_file():
        yield folder
        folder /= "modules"
    if folder.is_dir():
        for child in folder.iterdir():
            yield from _git_dirs(child)


def _evidence(root: Path) -> list[Path]:
    """The folders named for evidence under `root`, and each Parquet file outside them.

    A folder is not entered once found, and none of `_UNWALKED` is entered at all.
    """
    found: list[Path] = []
    for folder, names, files in os.walk(root):
        found += [Path(folder, name) for name in names if name in _EVIDENCE]
        found += [Path(folder, name) for name in files if name.endswith(".parquet")]
        names[:] = [name for name in names if name not in _EVIDENCE | _UNWALKED]
    return found


def _untouched(paths: Iterable[Path], since: float) -> dict[Path, int]:
    """The size of each of `paths` unmodified since `since`, leaving out one gone meanwhile."""
    found: dict[Path, int] = {}
    for path in paths:
        with suppress(FileNotFoundError):
            stat = path.stat()
            if stat.st_mtime < since:
                found[path] = stat.st_size
    return found


class Space:
    """The `disk`, `tree evidence`, `git leftovers` and `gate leftovers` rows of one doctor report.

    board: the workspace examined.
    gate_folder: the folder the gate roots its runs in.
    """

    @staticmethod
    def _by_disk(folders: Sequence[Path]) -> dict[int, Path]:
        """The first of `folders` on each disk among them, by the disk's id."""
        found: dict[int, Path] = {}
        for folder in folders:
            found.setdefault(Disk(path=folder).device, folder)
        return found

    @staticmethod
    def _measured(folders: Sequence[Path]) -> list[tuple[int, Path]]:
        """The bytes under each of `folders`, biggest first; none when `du` cannot say in time."""
        try:
            said = captured(["du", "-skx", *map(str, folders)], timeout=_DU_SECONDS).stdout
        except OSError, subprocess.TimeoutExpired:
            return []
        rows = [line.split("\t", 1) for line in said.splitlines()]
        sized = [(int(kib) * BYTES_PER_KIBIBYTE, Path(path)) for kib, path in rows]
        return sorted(sized, reverse=True)

    @staticmethod
    def _over(root: Path) -> bool:
        """Whether the gate run that made `root` is over.

        It is once the launcher the name records is gone, or, for a root that records none, once
        the root sat untouched for a day.
        """
        named = _GATE.fullmatch(root.name)
        if named is None or not root.is_dir():
            return False
        if named[1]:
            return not psutil.pid_exists(int(named[1]))
        return bool(_untouched([root], time.time() - SECONDS_PER_DAY))

    def __init__(self, board: Board, *, gate_folder: Path = _GATES) -> None:
        self.board = board
        self.gate_folder = gate_folder

    def _biggest(self, devices: Collection[int], *, out: Path, place: Path) -> str:
        """The biggest folders on `devices` among the generated folder's own and `place`."""
        folders = dict.fromkeys(
            folder.resolve()
            for folder in (*out.glob("*"), place)
            if folder.exists() and not folder.is_symlink() and Disk(path=folder).device in devices
        )
        sized = Space._measured([*folders])[:_BIGGEST]
        return abbreviated([f"{self._named(path)} {_gib(size)}" for size, path in sized])

    def _named(self, path: Path) -> Path:
        """`path` relative to the workspace when it lies inside it."""
        root = self.board.root.resolve()
        return path.relative_to(root) if path.is_relative_to(root) else path

    def disk(self) -> Section:
        """Free space on each disk the workspace writes to, against the floor.

        Those are its generated folder's, its lake's and its transient place's
        (`Transient.place`). Below twice the floor it warns, below the floor (where large writers
        refuse) it fails, and either names the biggest folders there, which only a tight disk
        pays to size.
        """
        root = self.board.root
        out, place = Project().out(root), Transient(root).place
        disks = Space._by_disk([out, self.board.manifest.workspace.lake_home(root), place])
        free = {device: Disk(path=folder).free for device, folder in disks.items()}
        detail = ", ".join(
            f"{_gib(free[device])} free at {where}" for device, where in disks.items()
        )
        tight = {device for device, room in free.items() if room < 2 * DISK_FLOOR}
        if not tight:
            return Section(
                section="disk", verdict=Verdict.PASS, detail=f"{detail}; floor {_gib(DISK_FLOOR)}"
            )
        biggest = self._biggest(tight, out=out, place=place)
        failing = any(free[device] < DISK_FLOOR for device in tight)
        return Section(
            section="disk",
            verdict=Verdict.FAIL if failing else Verdict.WARN,
            detail=(
                f"{detail}; large writers refuse below {_gib(DISK_FLOOR)}"
                + (f"; biggest: {biggest}" if biggest else "")
            ),
            fix="free space on that disk, or point [workspace] tmp at a bigger one",
        )

    def evidence(self) -> Section | None:
        """The evidence still sitting in the tree: `datasets` and `evidence` folders and Parquet.

        The lake is where evidence lives and the tree only caches it, so past a few gibibytes the
        row warns and names the two plain commands that keep the biggest folder and then give
        the disk back; `evict` removes only what the lake holds intact. None when `du` cannot
        size them in time.
        """
        folders = _evidence(self.board.root)
        found = Space._measured(folders) if folders else []
        if folders and not found:
            return None
        total = sum(size for size, _ in found)
        if total <= _EVIDENCE_LIMIT:
            return Section(
                section="tree evidence", verdict=Verdict.PASS, detail=f"{_gib(total)} in the tree"
            )
        biggest = abbreviated(
            [f"{self._named(path)} {_gib(size)}" for size, path in found[:_BIGGEST]]
        )
        folder = quote(str(self._named(found[0][1])))
        return Section(
            section="tree evidence",
            verdict=Verdict.WARN,
            detail=(
                f"{_gib(total)} of evidence sits in the tree, biggest: {biggest}; the lake keeps "
                "what `ingest` took, and `evict` gives the disk back for what it holds intact"
            ),
            fix=f"mb data lake ingest {folder} && mb data lake evict {folder}",
        )

    def gates(self) -> Section | None:
        """The gate roots whose run is over but whose folder stayed (a killed gate)."""
        over = sorted(root for root in self.gate_folder.glob("mb-gate.*") if Space._over(root))
        if not over:
            return None
        return Section(
            section="gate leftovers",
            verdict=Verdict.WARN,
            detail=f"{len(over)} root(s) of finished gate runs: {abbreviated([*map(str, over)])}",
            fix=join(["rm", "-rf", *map(str, over)]),
        )

    def git(self) -> Section | None:
        """The temporary files git left in the workspace's repositories when it died writing.

        One untouched for an hour belongs to no running git.
        """
        repository = self.board.root / ".git"
        files = (
            file
            for directory in _git_dirs(repository)
            for file in (
                *(directory / "objects" / "pack").glob("tmp_*"),
                *(directory / "objects").glob("*/tmp_obj_*"),
            )
        )
        if not (left := _untouched(files, time.time() - SECONDS_PER_HOUR)):
            return None
        return Section(
            section="git leftovers",
            verdict=Verdict.WARN,
            detail=(
                f"{len(left)} temporary file(s), {_gib(sum(left.values()))}, that git left "
                f"under {repository} when it died"
            ),
            fix=(
                f"find {quote(str(repository))} -path '*/objects/*' -name 'tmp_*' -type f "
                "-mmin +60 -delete"
            ),
        )
