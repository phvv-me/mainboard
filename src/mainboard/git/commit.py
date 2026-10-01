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

# What a repository with no commit identity is held for, and the fix.
NO_IDENTITY = "git knows no author here; set `git config --global user.name` and `user.email`"


class Commit:
    """Commit every dirty owned repository, submodules first so each parent records their commits.

    Each repository commits on a branch. A detached HEAD is put back on its trunk first when that
    moves no commit, and held otherwise, since a commit on a detached HEAD is one no branch will
    ever push. A repository behind its upstream commits and then merges the upstream in, so the
    next push is not refused; a merge that conflicts is aborted with its paths named and the
    commit kept. A parent whose submodule did not commit is held, rather than recording a
    pointer that leaves that submodule's work out.

    Symbolic links a Windows checkout wrote as plain files are put back first when the file is
    only the link written out. What enters a commit is everything changed except what has to
    stay out: anything under a `never-commit` pattern, which git is asked never to list, a file
    over the size ceiling that Git LFS does not carry, a link still a file because it was edited
    through, and a repository nested in the tree that `.gitmodules` does not declare, whose
    pointer no clone could follow. Those stay in the working tree, unstaged, and the step names
    them. A repository git knows no author for is held before anything is staged.
    """

    def __init__(self, tree: Tree, message: str) -> None:
        self.tree = tree
        self.message = message

    def run(self) -> list[Step]:
        """Commit bottom-up, holding every parent of a submodule that did not get there."""
        return self.tree.upward("commit", self._committed)

    def _committed(self, repo: Repo) -> Step:
        """Commit one repository's changes, or say why it was left alone."""
        relinked = [path for path in repo.unlinked() if repo.faithful(path)]
        _stage(repo, relinked, "checkout")
        notes = [_named("relinked", set(relinked))]
        changes = repo.changes(self.tree.policy.outside)
        if not changes:
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=notes[0] or "clean")
        if not repo.identified():
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=NO_IDENTITY)
        if not repo.branch() and not repo.attach():
            detail = f"detached at {repo.short('HEAD')}, off the line of {repo.trunk()}"
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
        withheld = Intake(repo, self.tree.policy).withheld(changes)
        _stage(repo, [c.path for c in changes if not c.staged and c.path not in withheld])
        _stage(repo, sorted(withheld), "reset", "-q")
        notes.append(_named("withheld", withheld))
        if repo.git.ok("diff", "--cached", "--quiet"):
            detail = "; ".join(filter(None, notes)) or "nothing staged"
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=detail)
        committed = repo.git.run("commit", "-q", "-m", self.message)
        if not committed.succeeded:
            return Step(repo=repo.name, outcome=Outcome.FAILED, detail=said(committed))
        notes = [f"{repo.short('HEAD')} on {repo.branch()}", *notes, _merged(repo)]
        return Step(repo=repo.name, outcome=Outcome.DONE, detail="; ".join(filter(None, notes)))


class Intake:
    """Which of a repository's changed paths `[git]` keeps out of a commit."""

    def __init__(self, repo: Repo, policy: GitPolicy) -> None:
        self.repo = repo
        self.policy = policy

    def withheld(self, changes: Sequence[Change]) -> set[str]:
        """The paths that must stay out: `never-commit` content in the index, an oversized file,
        a link checked out as a file, or a nested repository `.gitmodules` does not declare.

        `changes` was listed with every `never-commit` path already left out, so the one place
        such a path can still be is the index, staged by hand. A deletion there goes through,
        which is how something tracked by mistake leaves history.
        """
        return (
            self._patterned()
            | self._heavy([c.path for c in changes if not c.deleted])
            | set(self.repo.unlinked())
            | self._embedded(changes)
        )

    def _embedded(self, changes: Sequence[Change]) -> set[str]:
        """Every untracked nested repository `.gitmodules` does not declare, which `add` would
        record as a pointer with no URL. Git lists such a repository as one path ending in `/`."""
        declared = set(self.repo.submodule_paths)
        return {
            change.path
            for change in changes
            if change.untracked
            and change.path.endswith("/")
            and change.path.removesuffix("/") not in declared
        }

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
        # `git check-attr -z` answers in path, attribute, value triples.
        triples = zip(*[iter(listing)] * 3, strict=False)
        lfs = {path for path, _, value in triples if value == "lfs"}
        return set(oversized) - lfs

    def _size(self, path: str) -> int:
        """The bytes `path` puts in history: its own, a symlink's rather than its target's.

        A moved submodule is a directory here and a commit id in history, so it weighs nothing. A
        directory's size is filesystem bookkeeping (4096 on ext4, a few dozen bytes per entry on
        APFS), which once withheld every pointer on Linux while macOS committed it.
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


def _merged(repo: Repo) -> str:
    """Merge the upstream a freshly committed repository is behind, saying how that went."""
    upstream = repo.upstream()
    _, behind = repo.counts(upstream)
    if not behind:
        return ""
    if stopped := repo.merge(upstream):
        return f"merging {upstream} stopped on {stopped}; resolve with `git merge {upstream}`"
    return f"merged {behind} from {upstream}"


def _named(verb: str, paths: set[str]) -> str:
    """The step's note naming what `verb` happened to, empty when nothing."""
    if not paths:
        return ""
    named = sorted(paths)[:_NAMED]
    rest = len(paths) - len(named)
    return f"{verb} {', '.join(named)}" + (f" and {rest} more" if rest else "")
