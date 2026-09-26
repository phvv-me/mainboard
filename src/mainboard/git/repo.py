import posixpath
import re
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple

from .process import Git

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Every repository in the tree answers to `origin`: a submodule is cloned with that name and the
# workspace root is a clone too, so there is no second remote to choose between.
REMOTE = "origin"

# The branch a repository with no declared branch and no remote HEAD is assumed to follow.
_DEFAULT_TRUNK = "main"

# The mode git records a submodule pointer (a gitlink) under in a tree.
_GITLINK = "160000"

# Porcelain status codes: `??` is a file git does not track, `D` in either column a deletion.
_UNTRACKED = "??"
_DELETED = "D"

# A remote URL's separators across https, scp-like ssh and a local path on either platform.
_URL_PARTS = re.compile(r"[/:\\]+")


class Change(NamedTuple):
    """One path `git status` reports, with its two-letter porcelain code.

    code: the index and worktree columns, `??` for an untracked file.
    path: the path relative to the repository.
    """

    code: str
    path: str

    @property
    def untracked(self) -> bool:
        """Whether git does not track this path yet."""
        return self.code == _UNTRACKED

    @property
    def deleted(self) -> bool:
        """Whether the change removes the path, in the index or in the worktree."""
        return _DELETED in self.code

    @property
    def staged(self) -> bool:
        """Whether the index already holds the whole change, as `git rm` leaves it."""
        # Git refuses such a path as a pathspec to `add`: neither index nor worktree holds it.
        return self.code == f"{_DELETED} "


def owner_of(url: str) -> str:
    """The owner a remote URL names: the path segment above the repository itself.

    `https://github.com/phvv-me/aizk.git`, `git@github.com:phvv-me/aizk.git` and a local
    `/srv/remotes/phvv-me/aizk.git` all name `phvv-me`; a URL with no such segment names nobody.
    """
    parts = [part for part in _URL_PARTS.split(url) if part]
    return parts[-2] if len(parts) > 1 else ""


def resolved(url: str, base: str) -> str:
    """A `.gitmodules` URL made absolute, a `./` or `../` one against the parent remote `base`."""
    if not url.startswith(("./", "../")):
        return url
    return posixpath.normpath(posixpath.join(base, url))


class Repo:
    """One repository of the tree: the root or a submodule at any depth.

    Every question is asked of git in this working tree and answered fresh, since a verb that
    commits or pulls changes the answer halfway through. Only what cannot change during a verb is
    cached: the remote URL, the owner and the submodule entries `.gitmodules` declares.

    name: the workspace-relative path, `.` for the root.
    declared: the URL the parent's `.gitmodules` declares, empty for the root.
    branch: the branch the parent's `.gitmodules` says this submodule follows, empty when none.
    owns: whether a remote owner is this workspace's own.
    """

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

    @property
    def initialized(self) -> bool:
        """Whether the working tree is checked out, a `.git` file or directory present."""
        return (self.path / ".git").exists()

    @cached_property
    def url(self) -> str:
        """Where this repository pushes: its own `origin` once checked out, else the declared one.

        A checkout whose `origin` was pointed at a fork pushes to the fork, so ownership follows
        the remote a push would really reach rather than what the parent once declared.
        """
        if not self.initialized:
            return self.declared
        found = self.git.run("remote", "get-url", REMOTE)
        return found.stdout.strip() if found.succeeded else self.declared

    @cached_property
    def owner(self) -> str:
        """The owner the remote URL names."""
        return owner_of(self.url)

    @property
    def owned(self) -> bool:
        """Whether this is the workspace's own repository, one the writing verbs may touch."""
        return self.owns(self.owner)

    @cached_property
    def children(self) -> list[Repo]:
        """Every submodule `.gitmodules` declares here, checked out or not, in declared order."""
        entries: dict[str, dict[str, str]] = {}
        # No `.gitmodules`, or one declaring nothing, is exit 1 and no output: no submodules.
        listing = self.git.run(
            "config", "-z", "-f", ".gitmodules", "--get-regexp", r"^submodule\."
        ).stdout
        for record in filter(None, listing.split("\0")):
            key, _, value = record.partition("\n")
            section, _, field = key.removeprefix("submodule.").rpartition(".")
            entries.setdefault(section, {})[field] = value
        return [
            Repo(
                self.path / entry["path"],
                name=posixpath.join(self.name, entry["path"]).removeprefix("./"),
                declared=resolved(entry.get("url", ""), self.url),
                branch=entry.get("branch", ""),
                owns=self.owns,
            )
            for entry in entries.values()
        ]

    def relative(self, child: Repo) -> str:
        """`child`'s path inside this repository, the way this repository's index names it."""
        return child.path.relative_to(self.path).as_posix()

    def head(self) -> str:
        """The commit HEAD is on."""
        return self.git.line("rev-parse", "HEAD")

    def short(self, commit: str) -> str:
        """`commit` abbreviated the way git would print it here."""
        return self.git.line("rev-parse", "--short", commit)

    def branch(self) -> str:
        """The checked-out branch, empty when HEAD is detached."""
        found = self.git.run("symbolic-ref", "-q", "--short", "HEAD")
        return found.stdout.strip() if found.succeeded else ""

    def trunk(self) -> str:
        """The branch this repository follows: declared by the parent, else the remote's HEAD."""
        if self.declared_branch:
            return self.declared_branch
        found = self.git.run("symbolic-ref", "-q", "--short", f"refs/remotes/{REMOTE}/HEAD")
        if not found.succeeded:
            return _DEFAULT_TRUNK
        return found.stdout.strip().removeprefix(f"{REMOTE}/")

    def attach(self) -> str:
        """Put a detached HEAD back on its trunk where that moves no commit, returning the branch.

        The trunk is reset to HEAD only when that fast-forwards the local branch and HEAD sits on
        the remote trunk's own line, behind it or ahead of it, so no commit is orphaned and the
        next push is not a divergence. The working tree is untouched, since HEAD stays where it
        is. Empty when either condition fails, leaving HEAD detached.
        """
        trunk = self.trunk()
        head = self.head()
        local = f"refs/heads/{trunk}"
        remote = f"{REMOTE}/{trunk}"
        if self.exists(local) and not self.ancestor(local, head):
            return ""
        tracked = self.exists(remote)
        if tracked and not (self.ancestor(remote, head) or self.ancestor(head, remote)):
            return ""
        self.git.out("checkout", "-q", "-B", trunk)
        if tracked:
            self.git.out("branch", "-q", f"--set-upstream-to={remote}")
        return trunk

    def upstream(self) -> str:
        """The remote branch HEAD is counted against, empty when there is none to count.

        A branch answers with the upstream it is configured to track and nothing when it tracks
        none, since guessing one would count against a branch it never meant to follow. A
        detached HEAD answers with its trunk on the remote, which is the branch it is attached to
        the first time a commit or a pull puts it back on one.
        """
        if self.branch():
            found = self.git.run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
            return found.stdout.strip() if found.succeeded else ""
        remote = f"{REMOTE}/{self.trunk()}"
        return remote if self.exists(remote) else ""

    def counts(self, upstream: str) -> tuple[int, int]:
        """How many commits HEAD is ahead of and behind `upstream`, zero both without one."""
        if not upstream:
            return 0, 0
        counted = self.git.line("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
        ahead, behind = counted.split()
        return int(ahead), int(behind)

    def changes(self, outside: Sequence[str] = ()) -> list[Change]:
        """Every changed, staged or untracked path, a submodule counted only when its commit moved.

        A submodule with edits of its own is that submodule's business, reported on its own row,
        so the parent sees it only once the pointer it records would change.

        outside: exclude pathspecs whose paths are left out, which git then never walks.
        """
        listing = self.git.out(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
            "--ignore-submodules=dirty",
            *(("--", ".", *outside) if outside else ()),
        )
        return [Change(entry[:2], entry[3:]) for entry in listing.split("\0") if entry]

    def exists(self, ref: str) -> bool:
        """Whether `ref` names a commit this repository has."""
        return self.git.ok("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")

    def ancestor(self, older: str, newer: str) -> bool:
        """Whether `older` is `newer` or an ancestor of it, so one fast-forwards to the other."""
        return self.git.ok("merge-base", "--is-ancestor", older, newer)

    def homes(self, commit: str) -> list[str]:
        """The remote branches holding `commit`, empty when none does or it is unknown here.

        This is what decides whether a parent pointer at `commit` can be cloned: a commit no
        remote branch reaches is one the remote may refuse to serve.
        """
        found = self.git.run(
            "for-each-ref",
            "--format=%(if)%(symref)%(then)%(else)%(refname:short)%(end)",
            "--contains",
            commit,
            f"refs/remotes/{REMOTE}/",
        )
        return found.stdout.split() if found.succeeded else []

    def serves(self, commit: str) -> bool:
        """Whether `origin` hands out `commit`, which a clone of a parent pointing at it needs.

        A remote branch this clone knows holding it answers without the network. Otherwise the
        remote itself is asked for that one commit, since a clone's refs see only part of it: a
        shallow clone cannot trace a branch past its boundary, and a single-branch clone never
        tracks the other branches that may hold it.
        """
        return bool(self.homes(commit)) or self._offered(commit)

    def _offered(self, commit: str) -> bool:
        """Whether `origin` serves `commit` when asked for it alone, without its history or tree.

        The fetch lands in a scratch repository, so this clone's objects and shallow boundary
        stay as they were; a remote that ignores the tree filter sends one snapshot at most.
        """
        with TemporaryDirectory() as scratch:
            probe = Git(Path(scratch))
            probe.out("init", "--quiet", "--bare")
            asked = ("fetch", "--quiet", "--depth=1", "--filter=tree:0", self.url, commit)
            return probe.run(*asked, network=True).succeeded

    def pointers(self) -> dict[str, str]:
        """The commit HEAD records for every submodule path, keyed by that path."""
        paths = [self.relative(child) for child in self.children]
        if not paths:
            return {}
        listing = self.git.out("ls-tree", "-z", "HEAD", "--", *paths)
        entries = (entry.partition("\t") for entry in listing.split("\0") if entry)
        return {path: meta.split()[2] for meta, _, path in entries if meta.startswith(_GITLINK)}

    def fetch(self) -> str:
        """Fetch from `origin`, pruning branches it deleted, and return git's complaint if any."""
        found = self.git.run("fetch", "--prune", "--quiet", REMOTE, network=True)
        return "" if found.succeeded else found.stderr.strip()

    def lfs(self) -> bool:
        """Whether this repository keeps files in Git LFS, which a push has to upload first."""
        try:
            attributes = (self.path / ".gitattributes").read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        return "filter=lfs" in attributes
