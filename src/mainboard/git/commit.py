from stat import S_ISDIR
from typing import TYPE_CHECKING

from .process import said
from .report import Outcome, Step

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..manifest.schema.git import GitPolicy
    from .repo import Change, Repo
    from .tree import Tree

# How many withheld paths a step names before it only counts the rest.
_NAMED = 3

# `git check-attr -z` answers in path, attribute, value triples.
_TRIPLE = 3


class Commit:
    """Commit every dirty owned repository, submodules first so each parent records their commits.

    Each repository commits on a branch. A detached HEAD is put back on its trunk first when that
    moves no commit, and held otherwise, since a commit on a detached HEAD is one no branch will
    ever push. A repository behind its upstream is held too, because committing there makes the
    divergence the next push is refused for. A parent whose submodule did not commit is held as
    well, rather than recording a pointer that leaves that submodule's work out.

    What enters a commit is everything changed except what `[git]` withholds: anything under a
    `never-commit` pattern, which git is asked never to list, and any file over the size ceiling
    that Git LFS does not carry. Those stay in the working tree, unstaged. The step names the
    oversized files and any `never-commit` path somebody had staged by hand, since those are the
    ones a person expected to go in.
    """

    def __init__(self, tree: Tree, message: str) -> None:
        self.tree = tree
        self.message = message

    def run(self) -> list[Step]:
        """Commit bottom-up, holding every parent of a submodule that did not get there."""
        steps: list[Step] = []
        stuck: set[str] = set()
        for repo in reversed(self.tree.owned()):
            blocked = [child.name for child in repo.children if child.name in stuck]
            step = (
                Step(repo=repo.name, outcome=Outcome.HELD, detail=f"{blocked[0]} did not commit")
                if blocked
                else self._committed(repo)
            )
            if not step.outcome.settled:
                stuck.add(repo.name)
            steps.append(step)
        return steps

    def _committed(self, repo: Repo) -> Step:
        """Commit one repository's changes, or say why it was left alone."""
        changes = repo.changes(self.tree.policy.outside)
        if not changes:
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail="clean")
        if not repo.branch() and not repo.attach():
            detail = f"detached at {repo.short('HEAD')}, off the line of {repo.trunk()}"
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
        upstream = repo.upstream()
        _, behind = repo.counts(upstream)
        if behind:
            detail = f"{behind} behind {upstream}; pull first"
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
        withheld = Intake(repo, self.tree.policy).withheld(changes)
        pending = [change.path for change in changes if not change.staged]
        _stage(repo, [path for path in pending if path not in withheld])
        _stage(repo, sorted(withheld), "reset", "-q")
        note = _withheld(withheld)
        if repo.git.ok("diff", "--cached", "--quiet"):
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=note or "nothing staged")
        committed = repo.git.run("commit", "-q", "-m", self.message)
        if not committed.succeeded:
            return Step(repo=repo.name, outcome=Outcome.FAILED, detail=said(committed))
        detail = "; ".join(filter(None, [f"{repo.short('HEAD')} on {repo.branch()}", note]))
        return Step(repo=repo.name, outcome=Outcome.DONE, detail=detail)


class Intake:
    """Which of a repository's changed paths `[git]` keeps out of a commit.

    repo: the repository the paths belong to.
    policy: the workspace's `[git]` table.
    """

    def __init__(self, repo: Repo, policy: GitPolicy) -> None:
        self.repo = repo
        self.policy = policy

    def withheld(self, changes: Sequence[Change]) -> set[str]:
        """The paths that must stay out: `never-commit` content in the index, or an oversized file.

        `changes` was listed with every `never-commit` path already left out, so the one place
        such a path can still be is the index, staged by hand. A deletion there goes through,
        which is how something tracked by mistake leaves history.
        """
        return self._patterned() | self._heavy([c.path for c in changes if not c.deleted])

    def _patterned(self) -> set[str]:
        """The `never-commit` paths staged with content, a deletion aside."""
        if not self.policy.inside:
            return set()
        staged = self.repo.git.out(
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=d",
            "--",
            *self.policy.inside,
        )
        return set(filter(None, staged.split("\0")))

    def _heavy(self, paths: Sequence[str]) -> set[str]:
        """The paths over the size ceiling that Git LFS does not carry."""
        oversized = [path for path in paths if self._size(path) > self.policy.ceiling_bytes]
        if not oversized:
            return set()
        listing = self.repo.git.out("check-attr", "-z", "filter", "--", *oversized).split("\0")
        triples = zip(*[iter(listing)] * _TRIPLE, strict=False)
        lfs = {path for path, _, value in triples if value == "lfs"}
        return set(oversized) - lfs

    def _size(self, path: str) -> int:
        """The bytes `path` puts in history: its own, a symlink's rather than its target's.

        A moved submodule is a directory here and a commit id in history, so it weighs nothing.
        The directory's own size is filesystem bookkeeping (4096 on ext4, a few dozen bytes per
        entry on APFS) and once withheld every pointer on Linux while macOS committed it.
        """
        stat = (self.repo.path / path).lstat()
        return 0 if S_ISDIR(stat.st_mode) else stat.st_size


def _stage(repo: Repo, paths: Sequence[str], *command: str) -> None:
    """Hand `paths` to `git add -A`, or to `command`, through stdin rather than the command line.

    A tree this size easily passes the thirty-two thousand characters Windows allows a command
    line, and the paths are literal, so a file named `*.txt` never globs its neighbours in.
    """
    if not paths:
        return
    repo.git.out(
        "--literal-pathspecs",
        *(command or ("add", "-A")),
        "--pathspec-from-file=-",
        "--pathspec-file-nul",
        stdin="\0".join(paths),
    )


def _withheld(paths: set[str]) -> str:
    """The step's note naming what stayed out of the commit, empty when nothing did."""
    if not paths:
        return ""
    named = sorted(paths)[:_NAMED]
    rest = len(paths) - len(named)
    return f"withheld {', '.join(named)}" + (f" and {rest} more" if rest else "")
