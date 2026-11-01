from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from patos import FrozenModel, Runtime

from ..process import Git

if TYPE_CHECKING:
    from collections.abc import Sequence

# The mode git records a symbolic link under in a tree.
_SYMLINK = "120000"

# Porcelain status codes: `??` is a file git does not track, `D` in either column a deletion.
_UNTRACKED = "??"
_DELETED = "D"


class Change(NamedTuple):
    """One path `git status` reports, with its two-letter porcelain code.

    code: the index and worktree columns, `??` for an untracked file.
    path: the path relative to the repository.
    """

    code: str
    path: str

    @property
    def deleted(self) -> bool:
        """Whether the change removes the path, in the index or in the worktree."""
        return _DELETED in self.code

    @property
    def staged(self) -> bool:
        """Whether the index already holds the whole change, as `git rm` leaves it."""
        # Git refuses such a path as a pathspec to `add`: neither index nor worktree holds it.
        return self.code == f"{_DELETED} "

    @property
    def untracked(self) -> bool:
        """Whether git does not track this path yet."""
        return self.code == _UNTRACKED


class Worktree(FrozenModel):
    """The working tree of one repository against its index and HEAD.

    git: git run in the repository's working tree.
    path: the working tree's root.
    """

    git: Runtime[Git]
    path: Path

    def changes(self, outside: Sequence[str] = (), *, within: Sequence[str] = ()) -> list[Change]:
        """Every changed, staged or untracked path, a submodule counted only when its commit moved.

        A submodule with edits of its own is that submodule's business, reported on its own row,
        so the parent sees it only once the pointer it records would change.

        outside: exclude pathspecs whose paths are left out, which git then never walks.
        within: repository-relative paths the listing is limited to, everything when empty.
        """
        inside = [f":(literal){path}" for path in within] or ["."]
        listing = self.git.out(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
            "--ignore-submodules=dirty",
            "--",
            *inside,
            *outside,
        )
        return [Change(entry[:2], entry[3:]) for entry in listing.split("\0") if entry]

    def faithful(self, path: str) -> bool:
        """Whether the file at `path` is only its link written out.

        It is when it holds the link's own target path as text, or the very bytes of the file the
        link points at.
        """
        target = self.git.out("cat-file", "blob", f":{path}")
        written = (self.path / path).read_bytes()
        if written == target.encode():
            return True
        try:
            return written == ((self.path / path).parent / target).read_bytes()
        except OSError:
            return False

    def identified(self) -> bool:
        """Whether git knows who authors a commit here (`user.name` and `user.email`).

        A freshly set-up machine does not.
        """
        return self.git.ok("var", "GIT_AUTHOR_IDENT")

    def lfs(self) -> bool:
        """Whether this repository keeps files in Git LFS, which a push has to upload first."""
        try:
            attributes = (self.path / ".gitattributes").read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        return "filter=lfs" in attributes

    def tracked(self) -> list[str]:
        """Every path the index tracks, sparse ones included, as git spells it."""
        return [path for path in self.git.out("ls-files", "-z").split("\0") if path]

    def unlinked(self) -> list[str]:
        """Every symbolic link the index records that the working tree holds as a changed file.

        A checkout or a tool that could not keep a link wrote it as a file, the target copied or
        the target's path as text (`.codex/config.toml`, 2026-10-01). Git sees a type change, and
        staging it would replace the link for every other checkout.
        """
        listing = self.git.out(
            "diff", "--raw", "-z", "--no-renames", "--ignore-submodules=all", "--diff-filter=TM"
        )
        # `--raw -z` answers in pairs: `:old_mode new_mode old new T`, then the path.
        fields = listing.split("\0")
        return [
            path
            for meta, path in zip(fields[::2], fields[1::2], strict=False)
            if meta.startswith(f":{_SYMLINK} ") and not (self.path / path).is_symlink()
        ]
