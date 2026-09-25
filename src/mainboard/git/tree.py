from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from .check import Check
from .commit import Commit
from .pull import Pull
from .push import Push
from .repo import Repo
from .report import RepoState

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from ..manifest.schema.git import GitPolicy
    from .report import Finding, Step

# How many repositories talk to their remotes at once. A fetch is mostly waiting, so the tree's
# thirty remotes answer in the time a handful take, without opening thirty connections at once.
_FETCHERS = 8


class Tree:
    """The workspace repository and every submodule under it, operated as one repository.

    Ownership is read off each remote URL: a repository whose owner is the workspace root's own
    or one `[git] owners` names is this workspace's, and every writing verb touches those alone.
    Everything else in the tree (reference code pinned under `references/`, a vendored
    `third_party/`) is read to verify the pointers the owned repositories record, and never
    committed to, pushed or fast-forwarded. The walk only descends through owned repositories,
    so a foreign repository's own submodules are its own business.

    root: the workspace root, the top of a git working tree.
    policy: the workspace's `[git]` table.
    """

    def __init__(self, root: Path, policy: GitPolicy) -> None:
        self.policy = policy
        self.root = Repo(root, owns=self._owns)

    def owned(self) -> list[Repo]:
        """Every checked-out owned repository, each parent before its submodules."""
        return list(self._descend(self.root))

    def status(self) -> list[RepoState]:
        """One row per owned repository, read from what this machine already knows."""
        with ThreadPoolExecutor(max_workers=_FETCHERS) as pool:
            return list(pool.map(self._state, self.owned()))

    def pull(self) -> list[Step]:
        """Fast-forward every owned repository and bring submodule checkouts along."""
        return Pull(self).run()

    def commit(self, message: str) -> list[Step]:
        """Commit every dirty owned repository, submodules before the pointers to them.

        message: the commit message every repository's commit carries.
        """
        return Commit(self, message).run()

    def push(self) -> list[Step]:
        """Push every owned repository, submodules before the parents that point at them."""
        return Push(self).run()

    def check(self) -> list[Finding]:
        """Every inconsistency in the tree, an empty list for a tree safe to clone."""
        return Check(self).run()

    def fetch(self, repos: Sequence[Repo]) -> dict[str, str]:
        """Fetch every repository in `repos` at once, returning git's complaint for each failure.

        repos: the repositories to refresh, owned or not; a fetch writes only remote-tracking
            refs, so reading a foreign remote leaves its checkout exactly as it was.
        """
        with ThreadPoolExecutor(max_workers=_FETCHERS) as pool:
            complaints = list(pool.map(Repo.fetch, repos))
        return {
            repo.name: complaint
            for repo, complaint in zip(repos, complaints, strict=True)
            if complaint
        }

    def _owns(self, owner: str) -> bool:
        return self.policy.owns(owner, self.root.owner)

    def _descend(self, repo: Repo) -> Iterator[Repo]:
        if not (repo.initialized and repo.owned):
            return
        yield repo
        for child in repo.children:
            yield from self._descend(child)

    def _state(self, repo: Repo) -> RepoState:
        head = repo.head()
        upstream = repo.upstream()
        ahead, behind = repo.counts(upstream)
        changes = repo.changes(self.policy.outside)
        homes = repo.homes(head)
        untracked = sum(change.untracked for change in changes)
        return RepoState(
            repo=repo.name,
            owner=repo.owner,
            branch=repo.branch() or "detached",
            head=repo.short(head),
            upstream=upstream,
            ahead=ahead,
            behind=behind,
            changed=len(changes) - untracked,
            untracked=untracked,
            published=upstream if upstream in homes else next(iter(homes), ""),
        )
