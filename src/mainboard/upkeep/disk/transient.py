"""Where a workspace's large transient trees go, and clearing those dead processes left."""

import os
import re
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager, suppress
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path

import psutil

from ...core import Disk, ProcessTree, Project
from ...manifest import header

# How long a stream still writing into a dead process's folder has to end before it is killed.
_GRACE = 3.0


class _Kind(StrEnum):
    """What a transient folder holds, the start of its name `<kind>-<pid>-<random>`.

    COLLECT: a collection pass's transfer, staged before the lake keeps it (`staging`).
    FIX: a copy of a project that fixes are tried and proven in (`copy_root`).
    """

    COLLECT = auto()
    FIX = auto()


# A folder's name: its kind, then the process that made it, then a random tail.
_NAMED = re.compile(rf"(?:{'|'.join(_Kind)})-(\d+)-")


class Transient:
    """Where a workspace's large transient trees go, and clearing those a dead process left.

    Each tree is a folder `<kind>-<pid>-<random>` named by the process that made it. A killed
    process cannot clean up after itself, and a stream it started (a collection's `ssh`, in a
    session of its own) keeps writing into its folder, so the disk fills with bytes nothing will
    read. `reap` ends both, at every new folder and at every settlement pass. The rule is the
    process alone: a folder is stale only once the process named in it is gone, so a place two
    machines share at once would need a marker naming the machine too.

    root: the workspace, the cwd's when None.
    """

    @cached_property
    def apart(self) -> Path:
        """Where a copy of a tree goes that tools then run in.

        That is `place` when it lies outside the workspace, else the system's temporary folder.
        Under the workspace, git, the manifest search and the ignore files would all reach the
        workspace from inside the copy (git lists no file of a copy it finds ignored), and the
        next copy of a tree outside git would copy this one too.
        """
        inside = self.place.is_relative_to(self.root.resolve())
        return Path(tempfile.gettempdir()).resolve() if inside else self.place

    @cached_property
    def place(self) -> Path:
        """Where the trees go: `[workspace] tmp`, else the generated `tmp`, links followed."""
        return header(self.root).tmp_home(self.root).resolve()

    @contextmanager
    def staging(self, *, doing: str) -> Generator[Path]:
        """A new folder under `place` for a transfer, removed on exit (see `_made`)."""
        folder = self._made(_Kind.COLLECT, need=0, doing=doing)
        try:
            yield folder
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    @staticmethod
    def _dead(folder: Path) -> bool:
        """Whether the process named in `folder`'s name is gone, so nothing live uses it."""
        named = _NAMED.match(folder.name)
        return named is not None and folder.is_dir() and not psutil.pid_exists(int(named[1]))

    @staticmethod
    def _holders(folder: Path) -> list[psutil.Process]:
        """The processes, this one aside, with a file open under `folder`.

        A process the user may not inspect cannot be one of them: the streams are this user's.
        """
        where = folder.resolve()
        found: list[psutil.Process] = []
        for process in psutil.process_iter():
            with suppress(psutil.Error):
                if process.pid != os.getpid() and any(
                    Path(opened.path).is_relative_to(where) for opened in process.open_files()
                ):
                    found.append(process)
        return found

    @staticmethod
    def _silenced(folder: Path) -> None:
        """End every process writing into `folder`, killing the ones that outlive `_GRACE`."""
        holders = Transient._holders(folder)
        for process in holders:
            ProcessTree.end(process.pid)
        _, alive = psutil.wait_procs(holders, timeout=_GRACE)
        for process in alive:
            ProcessTree.end(process.pid, force=True)

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Project().workspace()

    def _made(self, kind: _Kind, *, need: int, doing: str) -> Path:
        """A new folder for `kind`, named by this process.

        `reap` runs first, then the disk must keep its floor once `need` bytes land
        (`Disk.require`, `doing` naming the writer).
        """
        parent = self._parent(kind)
        parent.mkdir(parents=True, exist_ok=True)
        self.reap()
        Disk(path=parent).require(need, doing=doing)
        return Path(tempfile.mkdtemp(prefix=f"{kind}-{os.getpid()}-", dir=parent))

    def _parent(self, kind: _Kind) -> Path:
        """The folder `kind`'s trees go in: `apart` for a copy tools run in, else `place`."""
        return self.apart if kind is _Kind.FIX else self.place

    def copy_root(self, *, need: int, doing: str) -> Path:
        """A new folder under `apart` for a copy of `need` bytes, which the caller removes.

        See `_made` for what comes first.
        """
        return self._made(_Kind.FIX, need=need, doing=doing)

    def reap(self) -> list[Path]:
        """Clear every folder whose process is gone, ending the streams still writing into it.

        Answers the folders cleared.
        """
        stale = [
            folder
            for kind in _Kind
            for folder in self._parent(kind).glob(f"{kind}-*")
            if Transient._dead(folder)
        ]
        for folder in stale:
            Transient._silenced(folder)
            shutil.rmtree(folder, ignore_errors=True)
        return stale
