"""A mirror: a copy of a project's files that plans are written into, so fixes are tried, settled
as the project's Ruff would leave them, and proven (fixing.py) without touching the project until
the end."""

import contextlib
import os
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs ruff with fixed argv since=2026-10-28
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ...core.errors import MissionError
from ...upkeep.disk.transient import Transient
from ..providers.external import Ruff
from .facts import listed
from .settings import Settings

if TYPE_CHECKING:
    from .apply import Plan

# The Ruff codes of the imports a fix leaves unused or out of order, how Ruff is told to keep the
# project's exclusions and answer in JSON, and how long one call over a mirror's files may take.
_IMPORTS = "F401,I001"
_REPORTED = ("--force-exclude", "--output-format", "json")
_RUFF_SECONDS = 120


class Mirror:
    """A copy of a project's files (those git lists), so neither side sees the other's later
    edits: a proof runs on the files as they were when it began. The settings its checkers read
    from outside the project (`Settings`: the file Ruff finds or extends, the nearest ones of ty,
    pyrefly and mypy up to there) are copied to the same relative places, and the places those
    settings name (a search root `..`, the stubs in `../../stubs`) are linked to the real trees,
    so the tools run in the copy read what they read in place. Nothing else beside the project is
    linked, so a folder holding thousands of entries costs a mirror nothing."""

    @staticmethod
    def _pruned(root: Path, name: str) -> None:
        """Remove the directories above the gone file `name` that are left empty."""
        for parent in PurePosixPath(name).parents:
            with contextlib.suppress(OSError):
                (root / parent).rmdir()

    @staticmethod
    def _published(source: Path, *, target: Path) -> bool:
        """Make `target` what `source` is: the same link, the same bytes, or gone.

        Answers whether it is gone.
        """
        if source.is_symlink():
            target.unlink(missing_ok=True)
            target.symlink_to(os.readlink(source))
        elif source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        else:
            target.unlink(missing_ok=True)
            return True
        return False

    def __init__(self, source: Path) -> None:
        """The copy lands where the cwd's workspace keeps such copies (`Transient.apart`)."""
        source = source.resolve()
        settings = Settings(source)
        anchor = settings.root
        names = listed(source)
        self.top = Transient().copy_root(
            need=sum((source / name).lstat().st_size for name in names),
            doing=f"copying {source} to prove fixes in",
        )
        self.path = self.top / source.relative_to(anchor)
        for name in names:
            _copied(source / name, target=self.path / name)
        for file in settings.outside:
            _copied(file, target=self.top / file.relative_to(anchor))
        self._linked(settings.named, anchor=anchor, source=source)

    def _beside(self, place: Path, *, anchor: Path, source: Path) -> None:
        """Link everything in `place`, a search root above the project, down the path to it.

        Each folder on that path is linked but for the way to the project and `.git`.
        """
        folder = place
        for part in source.relative_to(place).parts:
            for entry in folder.iterdir():
                if entry.name not in {part, ".git"}:
                    self._link(entry, anchor=anchor)
            folder /= part

    def _expanded(self, other: Path, names: AbstractSet[str]) -> set[str]:
        """`names` with each folder, here or under `other`, as the files under it on both sides."""
        return {
            file
            for name in names
            for file in (
                _files(self.path, name) | _files(other, name)
                if (self.path / name).is_dir() or (other / name).is_dir()
                else {name}
            )
        }

    def _link(self, place: Path, *, anchor: Path) -> None:
        """Link `place` to the real tree where the mirror would hold it, unless it is held."""
        target = self.top / place.relative_to(anchor)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(place)

    def _linked(self, named: Sequence[Path], *, anchor: Path, source: Path) -> None:
        """Link the `named` places to the real trees; a search root above the project, by entry."""
        for place in named:
            if source.is_relative_to(place):
                self._beside(place, anchor=anchor, source=source)
            else:
                self._link(place, anchor=anchor)

    def adopt(self, source: Path, names: AbstractSet[str]) -> None:
        """Make the files `names` here what they are under `source`, present or not.

        A folder among `names` stands for the files under it, here and under `source`.
        """
        for name in self._expanded(source, names):
            (self.path / name).unlink(missing_ok=True)
            if (source / name).exists() or (source / name).is_symlink():
                _copied(source / name, target=self.path / name)
            else:
                Mirror._pruned(self.path, name)

    def apply(self, plan: Plan) -> set[str]:
        """Write `plan`; return the files it changed, by their names before and after it."""
        changed = set()
        for path, data in plan.written().items():
            target = self.path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            changed.add(path)
        for old, new in plan.renamed.items():
            moved = _files(self.path, old) or {old}
            changed |= {*moved, *(plan.moved(name) for name in moved)}
            (self.path / new).parent.mkdir(parents=True, exist_ok=True)
            (self.path / old).rename(self.path / new)
        for path in plan.removed:
            changed |= _files(self.path, path) or {path}
            target = self.path / path
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        return changed

    def close(self) -> None:
        shutil.rmtree(self.top)

    def publish(self, root: Path, names: AbstractSet[str]) -> None:
        """Write the files `names` into the project at `root` as they are here.

        A folder among `names` stands for the files under it, here and in the project. Those gone
        are removed, with the directories that leaves empty.
        """
        for name in sorted(self._expanded(root, names)):
            if Mirror._published(self.path / name, target=root / name):
                Mirror._pruned(root, name)

    def tidied(self, names: AbstractSet[str], *, found: Path) -> set[str]:
        """The Python files of `names` whose imports settling and formatting here changed.

        Both run as `mainboard lint` runs them, by the project's Ruff configuration. A file keeps
        the shape it was found in under `found`: an import check or the formatting it failed there
        is not imposed on it here, and a file only here counts as found tidy.
        """
        files = {
            name
            for name in names
            if name.endswith((".py", ".pyi")) and (self.path / name).is_file()
        }
        there = {name for name in files if (found / name).is_file()}
        before = {name: (self.path / name).read_bytes() for name in files}
        settling = files - _ruff(found, there, "check", "--select", _IMPORTS)
        _ruff(self.path, settling, "check", "--fix", "--select", _IMPORTS)
        _ruff(self.path, files - _ruff(found, there, "format", "--check"), "format")
        return {name for name in files if (self.path / name).read_bytes() != before[name]}


def is_under(path: str, roots: AbstractSet[str]) -> bool:
    """Whether `path` is one of `roots` or lies in one of them."""
    return path in roots or any(str(parent) in roots for parent in PurePosixPath(path).parents)


def _copied(source: Path, *, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)


def _files(root: Path, name: str) -> set[str]:
    """The files under the folder `name` of `root`, relative to it; none when it is no folder."""
    folder = root / name
    if not folder.is_dir():
        return set()
    return {file.relative_to(root).as_posix() for file in folder.rglob("*") if not file.is_dir()}


def _ruff(root: Path, names: AbstractSet[str], *arguments: str) -> set[str]:
    """The files of `names` a Ruff command reports on, run in `root` by its configuration.

    A file the configuration excludes is left alone, and no names run nothing. Ruff failing
    without a report (a configuration it cannot read) is refused with what it said; a file that
    does not parse is reported.
    """
    if not names:
        return set()
    ran = subprocess.run(
        ["ruff", *arguments, *_REPORTED, "--", *sorted(names)],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=_RUFF_SECONDS,
    )
    if ran.returncode > 1 and not ran.stdout.strip():
        said = " ".join(ran.stderr.decode(errors="replace").split())
        raise MissionError(f"ruff {arguments[0]} failed in {root.resolve()}: {said}")
    found = Ruff.report.validate_json(ran.stdout or b"[]")
    base = root.resolve()
    return {Path(item.filename).resolve().relative_to(base).as_posix() for item in found}
