"""The repositories next to a project, where the consumers of its names live."""

import tomllib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec

from ...lint.files.git import topmost
from ..engine.facts import listed, vendored

_SCRIPTS = frozenset({"conftest", "setup", "__main__", "__init__"})
_ROOTS = ((), ("src",))


def module_key(path: str) -> str:
    """A Python file's path without its suffix and without `/__init__`: `src/patos/sql`."""
    key = PurePosixPath(path).with_suffix("").as_posix()
    return key.removesuffix("/__init__")


class Workspace:
    """The repositories next to a project: where its consumers live.

    root: the project; its workspace is the topmost working tree that holds it.
    """

    @staticmethod
    def _defines(owner: Path, packages: Sequence[str]) -> bool:
        return any(
            (owner / prefix / package / "__init__.py").is_file()
            for package in packages
            for prefix in ("", "src")
        )

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.top = topmost(self.root)

    def _related(self, owner: Path) -> bool:
        """Whether `owner` is the project or sits inside it.

        An owner holding the project (`research/cuda-tokenization` holds `cutoken`) stays a
        consumer through the files beside the project: the experiments that import it.
        """
        return owner == self.root or self.root in owner.parents

    def consumers(self, packages: list[str]) -> list[str]:
        """The other owned repositories, relative to the workspace.

        They are its manifest's owners minus the excluded and vendored ones, the project's own and
        any holding a package of the project.
        """
        if self.top is None:
            return []
        try:
            table = tomllib.loads((self.top / "mainboard.toml").read_text()).get("lint", {})
        except FileNotFoundError:
            return []
        excluded = GitIgnoreSpec.from_lines(table.get("exclude", ()))
        found = {
            path.relative_to(self.top).as_posix()
            for pattern in table.get("owners", ())
            for path in self.top.glob(pattern)
            if path.is_dir() and not path.is_symlink()
        }
        found -= vendored(self.top, found)
        return sorted(
            owner
            for owner in found
            if not excluded.match_file(f"{owner}/")
            and not self._related(self.top / owner)
            and not self._defines(self.top / owner, packages)
        )

    def packages(self) -> list[str]:
        """The top-level packages and modules the project defines.

        Packages are directories holding an `__init__.py`, at its root or in `src/`, and modules
        are the files lying there.
        """
        files = [PurePosixPath(name) for name in listed(self.root) if name.endswith(".py")]
        names = {
            file.parent.name if file.name == "__init__.py" else file.stem
            for file in files
            if file.parent.parts in _ROOTS
            and file.stem not in _SCRIPTS
            or file.name == "__init__.py"
            and file.parent.parent.parts in _ROOTS
        }
        return sorted(name for name in names if name.isidentifier())

    def submodules(self) -> frozenset[str]:
        """Every trailing part of the paths of the project's Python modules."""
        names = [name for name in listed(self.root) if name.endswith((".py", ".pyi"))]
        keys = [module_key(name).split("/") for name in names]
        return frozenset("/".join(parts[start:]) for parts in keys for start in range(len(parts)))
