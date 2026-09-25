from typing import TYPE_CHECKING

from .process import said
from .report import Outcome, Step

if TYPE_CHECKING:
    from .repo import Repo
    from .tree import Tree


class Pull:
    """Fast-forward every owned repository and bring submodule checkouts along, parents first.

    Nothing is merged or rebased. A branch that diverged from its upstream is held and named,
    and git's own refusal to overwrite local changes is what keeps a dirty checkout safe through
    a fast-forward. A submodule follows its parent's pointer only when it sat exactly on the old
    one, so a checkout somebody moved on purpose stays where it was put, and one never checked
    out is cloned at the pointer its parent records.
    """

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.steps: list[Step] = []
        self.unfetched: dict[str, str] = {}

    def run(self) -> list[Step]:
        """Fetch every owned repository at once, then walk the tree from the root down."""
        self.unfetched = self.tree.fetch(self.tree.owned())
        self._visit(self.tree.root, follow="")
        return self.steps

    def _visit(self, repo: Repo, *, follow: str) -> None:
        """Pull `repo`, then every submodule under it with the pointer the pull moved.

        follow: the commit the parent's pull moved this repository's pointer to, empty when it
            did not move or this checkout was not sitting on the old one.
        """
        before = repo.pointers()
        self.steps.append(self._pulled(repo, follow))
        after = repo.pointers()
        for child in repo.children:
            path = repo.relative(child)
            old, new = before.get(path, ""), after.get(path, "")
            moved = new != old
            if not child.initialized:
                self.steps.append(self._submodule(repo, child, "cloned at", "--init"))
            elif not child.owned and moved:
                self.steps.append(self._followed(repo, child, old))
            if child.initialized and child.owned:
                self._visit(child, follow=new if moved and child.head() == old else "")

    def _pulled(self, repo: Repo, follow: str) -> Step:
        """Move one owned repository forward: the parent's new pointer, its trunk, its upstream."""
        if complaint := self.unfetched.get(repo.name):
            return Step(repo=repo.name, outcome=Outcome.FAILED, detail=complaint)
        notes: list[str] = []
        if follow and follow != repo.head():
            moved = repo.git.run(*_forward(repo, follow))
            if not moved.succeeded:
                return Step(repo=repo.name, outcome=Outcome.HELD, detail=said(moved))
            notes.append(f"followed the parent to {repo.short(follow)}")
        if not repo.branch():
            if not (trunk := _attached(repo)):
                detail = f"detached at {repo.short('HEAD')}, off the line of {repo.trunk()}"
                return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
            notes.append(f"attached to {trunk}")
        return self._fast_forwarded(repo, notes)

    @staticmethod
    def _fast_forwarded(repo: Repo, notes: list[str]) -> Step:
        """Fast-forward the checked-out branch to its upstream, holding a divergence."""
        upstream = repo.upstream()
        ahead, behind = repo.counts(upstream)
        if ahead and behind:
            detail = f"diverged from {upstream}: {ahead} ahead, {behind} behind"
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
        if behind:
            merged = repo.git.run("merge", "--ff-only", "-q", upstream)
            if not merged.succeeded:
                return Step(repo=repo.name, outcome=Outcome.HELD, detail=said(merged))
            notes.append(f"fast-forwarded {behind} from {upstream}")
        if notes:
            return Step(repo=repo.name, outcome=Outcome.DONE, detail="; ".join(notes))
        where = f"level with {upstream}" if upstream else "tracking no upstream"
        return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=f"{repo.branch()} {where}")

    @staticmethod
    def _submodule(parent: Repo, child: Repo, done: str, *flags: str) -> Step:
        """Check `child` out at the pointer `parent` records, with git's own submodule update."""
        updated = parent.git.run(
            "submodule", "update", *flags, "--", parent.relative(child), network=True
        )
        if not updated.succeeded:
            return Step(repo=child.name, outcome=Outcome.FAILED, detail=said(updated))
        return Step(repo=child.name, outcome=Outcome.DONE, detail=f"{done} {child.short('HEAD')}")

    def _followed(self, parent: Repo, child: Repo, old: str) -> Step:
        """Move a foreign checkout to its parent's new pointer, if it sat on the old one."""
        if child.head() != old:
            detail = f"left at {child.short('HEAD')}, not the old pointer {old[:7]}"
            return Step(repo=child.name, outcome=Outcome.HELD, detail=detail)
        return self._submodule(parent, child, "moved to")


def _forward(repo: Repo, commit: str) -> tuple[str, ...]:
    """The git call that moves `repo` to `commit`: a fast-forward on a branch, else a checkout."""
    if repo.branch():
        return ("merge", "--ff-only", "-q", commit)
    return ("checkout", "-q", "--detach", commit)


def _attached(repo: Repo) -> str:
    """Put a detached HEAD back on its trunk, moving forward onto it when the trunk is ahead.

    Where the trunk already has commits HEAD lacks, and HEAD is one of the trunk's ancestors,
    switching to the trunk is itself the fast-forward a pull is for. Empty when neither fits.
    """
    if trunk := repo.attach():
        return trunk
    trunk = repo.trunk()
    local = f"refs/heads/{trunk}"
    if (
        repo.exists(local)
        and repo.ancestor("HEAD", local)
        and repo.git.ok("checkout", "-q", trunk)
    ):
        return trunk
    return ""
