# A member of the workspace: its installable Python project and its own manifest.

import tomllib
from typing import TYPE_CHECKING, ClassVar

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from patos import FrozenModel

from ...core.errors import MissionError
from ..schema.root import Manifest

if TYPE_CHECKING:
    from pathlib import Path


class Package(FrozenModel):
    """A member's installable Python project, as its `pyproject.toml` declares it.

    name: the distribution, normalized.
    python: its `requires-python`, empty when unstated.
    requires: every distribution any of its dependency lists or groups names, normalized, with
        every extra asked of it there.
    """

    PYPROJECT: ClassVar[str] = "pyproject.toml"

    name: str
    python: str = ""
    requires: dict[str, frozenset[str]] = {}

    @classmethod
    def read(cls, path: Path) -> Package | None:
        """The project `path` declares, None when the file or its `[project]` table is absent."""
        try:
            tree = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except tomllib.TOMLDecodeError as error:
            raise MissionError(f"{path} is not valid TOML: {error}") from None
        project = tree.get("project", {})
        if "name" not in project:
            return None
        lists = [
            project.get("dependencies", []),
            *project.get("optional-dependencies", {}).values(),
            *tree.get("dependency-groups", {}).values(),
        ]
        requires: dict[str, frozenset[str]] = {}
        for requirement in (
            Requirement(line) for listed in lists for line in listed if isinstance(line, str)
        ):
            name = canonicalize_name(requirement.name)
            requires[name] = requires.get(name, frozenset()) | requirement.extras
        return cls(
            name=canonicalize_name(project["name"]),
            python=project.get("requires-python", ""),
            requires=requires,
        )


class Member(FrozenModel):
    """One project a workspace composes.

    name: its directory's name, the namespace of its tasks, papers and lint tools.
    path: workspace-relative.
    package: its installable Python project, None without one.
    manifest: its own manifest with every local path rebased onto the workspace, None without.
    ignored: the root-only settings its manifest declares, read only when it stands alone.
    """

    name: str
    path: str
    package: Package | None = None
    manifest: Manifest | None = None
    ignored: tuple[str, ...] = ()
