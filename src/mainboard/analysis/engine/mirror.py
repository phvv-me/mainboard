"""A mirror: a copy of a project's files that plans are written into, so fixes are tried and
proven (fixing.py) without touching the project until the end."""

import contextlib
import os
import shutil
import tempfile
import tomllib
from collections.abc import Set as AbstractSet
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .facts import listed

if TYPE_CHECKING:
    from .apply import Plan


class Mirror:
    """A copy of a project's files (those git lists), so neither side sees the other's later
    edits: a proof runs on the files as they were when it began. The Ruff configurations the
    project extends (`extend = "../../../pyproject.toml"`) are copied to the same relative places,
    so the tools run in the copy read the configuration they read in place."""

    @staticmethod
    def _extended(project: Path) -> list[Path]:
        """The configuration files outside `project` that its Ruff configuration extends.

        Their own extensions count too; a `pyproject.toml` that does not extend one has none.
        """
        found: list[Path] = []
        pending = [project / name for name in ("pyproject.toml", "ruff.toml", ".ruff.toml")]
        while pending:
            parent = Mirror._extension(pending.pop())
            if parent is None:
                continue
            if parent not in found and not parent.is_relative_to(project):
                found.append(parent)
            pending.append(parent)
        return found

    @staticmethod
    def _extension(file: Path) -> Path | None:
        """The configuration `file` extends, None for one it does not or a file not there.

        An extension naming no file (a project copied out of its workspace) is none either.
        """
        try:
            table = tomllib.loads(file.read_text(encoding="utf-8"))
        except FileNotFoundError, tomllib.TOMLDecodeError:
            return None
        target = table.get("extend") if file.name != "pyproject.toml" else None
        target = target or table.get("tool", {}).get("ruff", {}).get("extend")
        extended = (file.parent / target).resolve() if target else None
        return extended if extended is not None and extended.is_file() else None

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
        source = source.resolve()
        extended = Mirror._extended(source)
        anchor = Path(os.path.commonpath([source, *(file.parent for file in extended)]))
        if anchor == source:
            anchor = source.parent
        self.top = Path(tempfile.mkdtemp(prefix="mb-fix-"))
        self.path = self.top / source.relative_to(anchor)
        for name in listed(source):
            _copied(source / name, target=self.path / name)
        for file in extended:
            _copied(file, target=self.top / file.relative_to(anchor))
        self._beside(anchor, source=source)

    def _beside(self, anchor: Path, *, source: Path) -> None:
        """Link what lies beside the project and its parents up to `anchor` to the real trees.

        Type-checker settings reach those trees as `..` or `../../stubs`, so they are symbolic
        links.
        """
        folder = anchor
        for part in source.relative_to(anchor).parts:
            for entry in folder.iterdir():
                target = self.top / entry.relative_to(anchor)
                if entry.name not in {part, ".git"} and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(entry)
            folder /= part

    def _files(self, path: str) -> set[str]:
        """The files at `path`: itself, or every file under it."""
        target = self.path / path
        if not target.is_dir():
            return {path}
        return {
            str(file.relative_to(self.path)) for file in target.rglob("*") if not file.is_dir()
        }

    def adopt(self, source: Path, names: AbstractSet[str]) -> None:
        """Make the files `names` here what they are under `source`, present or not."""
        for name in names:
            (self.path / name).unlink(missing_ok=True)
            if (source / name).exists() or (source / name).is_symlink():
                _copied(source / name, target=self.path / name)

    def apply(self, plan: Plan) -> set[str]:
        """Write `plan`; return the files it changed, by their names before and after it."""
        changed = set()
        for path, data in plan.written().items():
            target = self.path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            changed.add(path)
        for old, new in plan.renamed.items():
            changed |= {*self._files(old), *(plan.moved(name) for name in self._files(old))}
            (self.path / new).parent.mkdir(parents=True, exist_ok=True)
            (self.path / old).rename(self.path / new)
        for path in plan.removed:
            changed |= self._files(path)
            target = self.path / path
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        return changed

    def close(self) -> None:
        shutil.rmtree(self.top)

    def publish(self, root: Path, names: AbstractSet[str]) -> None:
        """Write the files `names` into the project at `root` as they are here.

        Those gone are removed, with the directories that leaves empty.
        """
        for name in sorted(names):
            if Mirror._published(self.path / name, target=root / name):
                Mirror._pruned(root, name)


def is_under(path: str, roots: AbstractSet[str]) -> bool:
    """Whether `path` is one of `roots` or lies in one of them."""
    return path in roots or any(str(parent) in roots for parent in PurePosixPath(path).parents)


def _copied(source: Path, *, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
