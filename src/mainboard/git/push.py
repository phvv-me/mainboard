from typing import TYPE_CHECKING

from ..core.project import Project
from .process import said
from .repo import REMOTE
from .report import Outcome, Step

if TYPE_CHECKING:
    from .repo import Repo
    from .tree import Tree

# What GitHub says when a push lands on a protected branch: GH006 for classic branch protection,
# GH013 for a repository ruleset. Either means the commit belongs on a branch and a pull request.
_PROTECTED = ("gh006", "gh013", "protected branch")

# The prefix of the branch a push falls back to when the remote protects the one it tracks,
# named after this tool so a stranger reading the remote knows which process made it.
_FALLBACK = Project().name


class Push:
    """Push every owned repository, submodules before the parents whose pointers name them.

    A parent is pushed only once every pointer its HEAD records is held by a branch of that
    submodule's remote, because a pointer the remote cannot serve is a clone that fails for
    everybody else. A remote that protects the tracked branch gets the commit on
    `<tool>/<branch>` instead, which is a branch like any other as far as a pointer is concerned,
    and the step says a pull request is what moves it the rest of the way. Git LFS objects are
    uploaded before the push that references them, whether or not the checkout carries the LFS
    hook that would have done it.
    """

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def run(self) -> list[Step]:
        """Push bottom-up, holding every parent of a submodule that did not get there."""
        return self.tree.upward("push", _pushed)


def _pushed(repo: Repo) -> Step:
    """Push one repository's branch, or say why it stayed."""
    branch = repo.branch()
    head = repo.head()
    if not branch:
        if homes := repo.homes(head):
            detail = f"detached at {repo.short(head)}, already on {homes[0]}"
            return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=detail)
        detail = f"detached at {repo.short(head)}, which no remote branch holds; commit first"
        return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
    if missing := _unserved(repo):
        detail = f"records {missing}, which its remote does not hold"
        return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
    upstream = repo.upstream()
    ahead, behind = repo.counts(upstream)
    target = upstream.removeprefix(f"{REMOTE}/") if upstream else branch
    fallback = f"{_FALLBACK}/{target}"
    if upstream and not ahead:
        detail = f"{branch} level with {upstream}"
        return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=detail)
    if behind:
        detail = f"diverged from {upstream}: {ahead} ahead, {behind} behind; pull first"
        return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
    if f"{REMOTE}/{fallback}" in repo.homes(head):
        detail = f"{target} is protected; {repo.short(head)} waits on {fallback}"
        return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=detail)
    return _delivered(repo, branch, target, tracked=bool(upstream))


def _delivered(repo: Repo, branch: str, target: str, *, tracked: bool) -> Step:
    """Upload LFS objects, then push `branch` to `target`, falling back when it is protected."""
    if repo.lfs():
        uploaded = repo.git.run("lfs", "push", REMOTE, branch, network=True)
        if not uploaded.succeeded:
            detail = f"git-lfs could not upload: {said(uploaded)}"
            return Step(repo=repo.name, outcome=Outcome.FAILED, detail=detail)
    tracking = () if tracked else ("--set-upstream",)
    pushed = repo.git.run("push", "--quiet", *tracking, REMOTE, f"{branch}:{target}", network=True)
    short = repo.short("HEAD")
    if pushed.succeeded:
        return Step(repo=repo.name, outcome=Outcome.DONE, detail=f"{short} to {REMOTE}/{target}")
    if not any(marker in pushed.stderr.casefold() for marker in _PROTECTED):
        return Step(repo=repo.name, outcome=Outcome.FAILED, detail=said(pushed))
    fallback = f"{_FALLBACK}/{target}"
    diverted = repo.git.run("push", "--quiet", REMOTE, f"HEAD:refs/heads/{fallback}", network=True)
    if not diverted.succeeded:
        return Step(repo=repo.name, outcome=Outcome.FAILED, detail=said(diverted))
    detail = f"{target} is protected; pushed {short} to {fallback}, open a pull request"
    return Step(repo=repo.name, outcome=Outcome.DONE, detail=detail)


def _unserved(repo: Repo) -> str:
    """The first submodule pointer HEAD records that no branch of its remote holds, as `path@sha`.

    What this machine last heard from each remote is asked first, and a submodule is fetched
    only when that says no, so a pointer pushed from another machine is not held for stale refs.
    A submodule never checked out cannot be asked and is left to `check`.
    """
    recorded = repo.pointers()
    for child in repo.children:
        commit = recorded.get(repo.relative(child), "")
        if not (commit and child.initialized) or child.homes(commit):
            continue
        child.fetch()
        if not child.homes(commit):
            return f"{child.name}@{commit[:7]}"
    return ""
