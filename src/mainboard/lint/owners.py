from functools import cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Owners:
    """Which project owns a file: the nearest enclosing directory the workspace calls an owner.

    A directory owns what lies beneath it when it holds one of the marker files or its
    workspace-relative path matches one of the owner globs, and the nearest such ancestor wins,
    so a package nested inside a research project is its own owner rather than its parent's.
    A file under no owner at all belongs to the workspace root.

    patterns: glob patterns of owner directories, relative to the root.
    markers: file names that make the directory holding them an owner.
    """

    def __init__(self, root: Path, patterns: Sequence[str], markers: Sequence[str]) -> None:
        self.root = root
        self.patterns = tuple(patterns)
        self.markers = tuple(markers)
        self.of = cache(self._of)

    def _of(self, directory: Path) -> Path:
        """The owner of everything in `directory`, an absolute directory beneath the root."""
        if directory == self.root or self._owns(directory):
            return directory
        return self.of(directory.parent)

    def _owns(self, directory: Path) -> bool:
        relative = PurePosixPath(directory.relative_to(self.root).as_posix())
        return any(relative.full_match(pattern) for pattern in self.patterns) or any(
            (directory / marker).exists() for marker in self.markers
        )
