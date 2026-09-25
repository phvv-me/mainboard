import tomllib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Sequence

# What makes a directory a Python project beside its own manifest.
_PYPROJECT = "pyproject.toml"


class Membership:
    """Which directories a workspace composes as members, from its `[workspace] members` globs.

    A directory is a member when a pattern matches its workspace-relative path, no `!` pattern
    leaves it out, and it holds a manifest or a `pyproject.toml`; a glob over a folder of mixed
    content therefore picks only the projects in it.

    patterns: workspace-relative globs, each `!pattern` excluding what it matches.
    manifest: the manifest's file name.
    """

    def __init__(self, root: Path, patterns: Sequence[str], manifest: str) -> None:
        self.root = root
        self.markers = (manifest, _PYPROJECT)
        self.included = [pattern for pattern in patterns if not pattern.startswith("!")]
        self.excluded = [pattern[1:] for pattern in patterns if pattern.startswith("!")]

    @classmethod
    def declared(cls, root: Path, manifest: str) -> Membership:
        """The membership the manifest file in `root` declares, read without validating it all."""
        path = root / manifest
        try:
            tree = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise MissionError(f"{path} is not valid TOML: {error}") from None
        return cls(root, tree.get("workspace", {}).get("members", []), manifest)

    def directories(self) -> list[str]:
        """Every member's workspace-relative path, sorted."""
        found = {
            PurePosixPath(match.relative_to(self.root).as_posix())
            for pattern in self.included
            for match in self.root.glob(pattern)
        }
        return sorted(str(path) for path in found if self._composes(path))

    def claims(self, directory: Path) -> bool:
        """Whether `directory`, anywhere on disk, is one of these members."""
        try:
            relative = PurePosixPath(directory.relative_to(self.root).as_posix())
        except ValueError:
            return False
        return any(relative.full_match(pattern) for pattern in self.included) and self._composes(
            relative
        )

    def _composes(self, relative: PurePosixPath) -> bool:
        """Whether a matched path is a project no exclusion leaves out, never the root itself."""
        directory = self.root / relative
        return (
            relative != PurePosixPath(".")
            and not any(relative.full_match(pattern) for pattern in self.excluded)
            and any((directory / marker).is_file() for marker in self.markers)
        )
