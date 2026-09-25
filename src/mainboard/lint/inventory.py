from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from .git import git, printed

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# The one newline git writes back out for a file whose attributes say `eol=crlf`.
_CRLF = "\r\n"


class Attributes(FrozenModel):
    """What `.gitattributes` says about one file's bytes.

    binary: whether git stores the file untouched (`-text` or `binary`), so no step may edit it.
    newline: the line ending the file is kept in, LF unless its attributes name `eol=crlf`.
    """

    binary: bool = False
    newline: str = "\n"


class Inventory:
    """The files a lint run reads, asked of git so ignored and generated trees never appear.

    Submodules are entered rather than skipped, since each one is a project of the monorepo
    whose files the owner scan checks, and git reports a submodule as one opaque path.

    root: the workspace root, inside a git work tree.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def changed(self) -> list[Path]:
        """Every file that differs from HEAD or is new, deletions included.

        A repository with no commit yet has no HEAD to differ from, so every file in it is new.
        """
        return sorted(set(self._changed(self.root)))

    def under(self, paths: Sequence[Path]) -> list[Path]:
        """Every file at or beneath `paths` that git tracks or would track.

        paths: absolute files or directories inside the root, each of which must exist.
        """
        found: set[Path] = set()
        for path in paths:
            if not path.is_relative_to(self.root):
                raise MissionError(f"{path} is outside the workspace at {self.root}")
            if not path.exists():
                raise MissionError(f"nothing to lint at {path}")
            found.update(self._listed(path) if path.is_dir() else (path,))
        return sorted(found)

    def attributes(self, files: Sequence[Path]) -> dict[Path, Attributes]:
        """The `text` and `eol` attributes of `files`, read in one batched query.

        files: absolute paths beneath the root.
        """
        names = [path.relative_to(self.root).as_posix() for path in files]
        fields = self._names(
            self.root, "check-attr", "-z", "--stdin", "text", "eol", stdin="\0".join(names)
        )
        found: dict[str, dict[str, str]] = {}
        for name, attribute, value in zip(fields[::3], fields[1::3], fields[2::3], strict=True):
            found.setdefault(name, {})[attribute] = value
        return {
            self.root / name: Attributes(
                binary=values["text"] == "unset",
                newline=_CRLF if values["eol"] == "crlf" else "\n",
            )
            for name, values in found.items()
        }

    def tracked(self, path: Path) -> bool:
        """Whether HEAD already holds `path`, so a size limit on new files leaves it alone."""
        return not git(path.parent, "cat-file", "-e", f"HEAD:./{path.name}").returncode

    def _changed(self, repository: Path) -> Iterator[Path]:
        names = [
            *self._names(repository, "ls-files", "--others", "--exclude-standard", "-z"),
            *self._differing(repository),
        ]
        for name in names:
            path = repository / name
            if path.is_dir():
                yield from self._changed(path)
            else:
                yield path

    def _differing(self, repository: Path) -> list[str]:
        """What differs from HEAD in `repository`, everything git tracks while HEAD is unborn."""
        if git(repository, "rev-parse", "--verify", "--quiet", "HEAD").returncode:
            return self._names(repository, "ls-files", "--cached", "-z")
        return self._names(repository, "diff", "--name-only", "--relative", "-z", "HEAD")

    def _listed(self, directory: Path) -> Iterator[Path]:
        names = self._names(
            directory, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        )
        for name in names:
            path = directory / name
            if path.is_dir():
                yield from self._listed(path)
            elif path.exists():
                yield path

    @staticmethod
    def _names(repository: Path, *arguments: str, stdin: str = "") -> list[str]:
        """The NUL-separated fields a git query prints, refused with git's own words."""
        answer = git(repository, *arguments, stdin=stdin)
        if answer.returncode:
            raise MissionError(
                f"git {arguments[0]} failed in {repository}: {printed(answer.stderr).strip()}"
            )
        return [field for field in printed(answer.stdout).split("\0") if field]
