import posixpath
import re
from functools import cached_property
from typing import TYPE_CHECKING

from ..constants import REMOTE
from ..process import Git
from .commits import Commits
from .remote import Remote
from .tracking import Tracking
from .worktree import Worktree

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# The mode git records a submodule pointer (a gitlink) under in a tree.
_GITLINK = "160000"

# A remote URL's separators across https, scp-like ssh and a local path on either platform.
_URL_PARTS = re.compile(r"[/:\\]+")


class Repo:
    """One repository of the tree: the root or a submodule at any depth.

    Every question is asked of git in this working tree and answered fresh, since a verb that
    commits or pulls changes the answer halfway through. Only what cannot change during a verb is
    cached: the remote URL, the owner and the submodules `.gitmodules` and the index record.

    The repository itself knows its place in the tree. What git says about its commits, its
    branches, its remote and its working tree is asked of `commits`, `tracking`, `remote` and
    `worktree`.

    name: the workspace-relative path, `.` for the root.
    declared: the URL the parent's `.gitmodules` declares, empty for the root.
    branch: the branch the parent's `.gitmodules` says this submodule follows, empty when none.
    owns: whether a remote owner is this workspace's own.
    """

    @cached_property
    def _declared(self) -> list[dict[str, str]]:
        """Every submodule entry `.gitmodules` declares, its fields by name, in declared order."""
        entries: dict[str, dict[str, str]] = {}
        # No `.gitmodules`, or one declaring nothing, is exit 1 and no output: no submodules.
        listing = self.git.run(
            "config", "-z", "-f", ".gitmodules", "--get-regexp", r"^submodule\."
        ).stdout
        for record in filter(None, listing.split("\0")):
            key, _, value = record.partition("\n")
            section, _, field = key.removeprefix("submodule.").rpartition(".")
            entries.setdefault(section, {})[field] = value
        return list(entries.values())

    @cached_property
    def _gitlinks(self) -> frozenset[str]:
        """The declared paths the index records a submodule pointer at."""
        paths = [entry["path"] for entry in self._declared]
        if not paths:
            return frozenset()
        listing = self.git.out("ls-files", "--stage", "-z", "--", *paths)
        entries = (entry.partition("\t") for entry in listing.split("\0") if entry)
        return frozenset(path for meta, _, path in entries if meta.startswith(_GITLINK))

    @cached_property
    def children(self) -> list[Repo]:
        """Every submodule the index records a gitlink for, checked out or not, in declared order.

        `.gitmodules` only names each one's URL and branch: an entry there with no gitlink is
        stale, answered by `stale` and walked by no verb, since git itself refuses it as a
        pathspec (`packages/cunicode`, 2026-09-26).
        """
        linked = self._gitlinks
        return [
            Repo(
                self.path / entry["path"],
                name=posixpath.join(self.name, entry["path"]).removeprefix("./"),
                declared=Repo._resolved(entry.get("url", ""), base=self.url),
                branch=entry.get("branch", ""),
                owns=self.owns,
            )
            for entry in self._declared
            if entry["path"] in linked
        ]

    @cached_property
    def commits(self) -> Commits:
        """Where HEAD is and how two commits relate."""
        return Commits(git=self.git)

    @cached_property
    def owner(self) -> str:
        """The owner the remote URL names."""
        return Repo._owner_of(self.url)

    @cached_property
    def remote(self) -> Remote:
        """What `origin` holds and whether it hands a commit out."""
        return Remote(git=self.git, url=self.url)

    @cached_property
    def tracking(self) -> Tracking:
        """The branch HEAD is on, the trunk it follows and the upstream it is counted against."""
        return Tracking(git=self.git, commits=self.commits, declared_branch=self.declared_branch)

    @cached_property
    def url(self) -> str:
        """Where this repository pushes: its own `origin` once checked out, else the declared one.

        A checkout whose `origin` was pointed at a fork pushes to the fork, so ownership follows
        the remote a push would really reach rather than what the parent once declared.
        """
        if not self.initialized:
            return self.declared
        found = self.git.run("remote", "get-url", REMOTE)
        return self.declared if found.returncode else found.stdout.strip()

    @cached_property
    def worktree(self) -> Worktree:
        """The working tree against the index and HEAD."""
        return Worktree(git=self.git, path=self.path)

    @property
    def initialized(self) -> bool:
        """Whether the working tree is checked out, a `.git` file or directory present."""
        return (self.path / ".git").exists()

    @property
    def owned(self) -> bool:
        """Whether this is the workspace's own repository, one the writing verbs may touch."""
        return self.owns(self.owner)

    @property
    def stale(self) -> list[str]:
        """Every path `.gitmodules` declares a submodule at with no gitlink in the index."""
        linked = self._gitlinks
        return [path for path in self.submodule_paths if path not in linked]

    @property
    def submodule_paths(self) -> list[str]:
        """Every path `.gitmodules` declares a submodule at, recorded in the index or not."""
        return [entry["path"] for entry in self._declared]

    @staticmethod
    def _owner_of(url: str) -> str:
        """The owner a remote URL names: the path segment above the repository itself.

        `https://github.com/phvv-me/aizk.git`, `git@github.com:phvv-me/aizk.git` and a local
        `/srv/remotes/phvv-me/aizk.git` all name `phvv-me`; a URL with no such segment names
        nobody.
        """
        parts = [part for part in _URL_PARTS.split(url) if part]
        return parts[-2] if len(parts) > 1 else ""

    @staticmethod
    def _resolved(url: str, *, base: str) -> str:
        """A `.gitmodules` URL made absolute, a relative one against the parent remote `base`."""
        if not url.startswith(("./", "../")):
            return url
        return posixpath.normpath(posixpath.join(base, url))

    def __init__(
        self,
        path: Path,
        *,
        name: str = ".",
        declared: str = "",
        branch: str = "",
        owns: Callable[[str], bool],
    ) -> None:
        self.path = path
        self.name = name
        self.declared = declared
        self.declared_branch = branch
        self.owns = owns
        self.git = Git(path)

    def pointers(self) -> dict[str, str]:
        """The commit HEAD records for every submodule path, keyed by that path."""
        paths = [self.relative(child) for child in self.children]
        if not paths:
            return {}
        listing = self.git.out("ls-tree", "-z", "HEAD", "--", *paths)
        entries = (entry.partition("\t") for entry in listing.split("\0") if entry)
        return {path: meta.split()[2] for meta, _, path in entries if meta.startswith(_GITLINK)}

    def relative(self, child: Repo) -> str:
        """`child`'s path inside this repository, the way this repository's index names it."""
        return child.path.relative_to(self.path).as_posix()

    def unreadable(self) -> str:
        """Git's own words when a plain `git status` here aborts, empty when it does not.

        It aborts on a submodule checkout it cannot read.

        Every other question this tree asks passes `--ignore-submodules` and survives a checkout
        whose `.git` names a directory that is gone, so nothing else would ever name the path a
        bare `git status` dies on. Owned submodules are left out: each answers on its own row.
        """
        owned = [f":(exclude){self.relative(child)}" for child in self.children if child.owned]
        found = self.git.run("status", "--porcelain=v1", "--untracked-files=no", "--", ".", *owned)
        return "; ".join(filter(None, found.stderr.splitlines())) if found.returncode else ""
