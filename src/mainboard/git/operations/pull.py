from typing import TYPE_CHECKING

from ..process import said
from ..report import Outcome, Step

if TYPE_CHECKING:
    from ..repo import Repo
    from ..tree import Tree


class Pull:
    """Bring every owned repository level with its upstream and submodule checkouts along,
    parents first.

    A branch behind its upstream fast-forwards, and one that diverged from it merges the upstream
    in; a merge that conflicts is aborted and held with its paths named, nothing ever rebased.
    Git's own refusal to overwrite local changes keeps a dirty checkout safe through either. A
    submodule follows its parent's pointer only when it sat exactly on the old one, so a checkout
    somebody moved on purpose stays where it was put, and one never checked out is cloned at the
    pointer its parent records.
    """

    @staticmethod
    def _attached(repo: Repo) -> str:
        """Put a detached HEAD back on its trunk, moving forward onto it when the trunk is ahead.

        Where the trunk already has commits HEAD lacks, and HEAD is one of the trunk's ancestors,
        switching to the trunk is itself the fast-forward a pull is for. Empty when neither fits.
        """
        if trunk := repo.attach():
            return trunk
        trunk = repo.trunk()
        local = f"refs/heads/{trunk}"
        forward = repo.exists(local) and repo.ancestor("HEAD", newer=local)
        return trunk if forward and repo.git.ok("checkout", "-q", trunk) else ""

    @staticmethod
    def _caught_up(repo: Repo, notes: list[str]) -> Step:
        """Level the checked-out branch with its upstream: fast-forward, or merge a divergence."""
        upstream = repo.upstream()
        levelled = Pull._levelled(repo, upstream)
        if isinstance(levelled, Step):
            return levelled
        if notes := [*notes, levelled] if levelled else notes:
            return Step(repo=repo.name, outcome=Outcome.DONE, detail="; ".join(notes))
        where = f"level with {upstream}" if upstream else "tracking no upstream"
        return Step(repo=repo.name, outcome=Outcome.CURRENT, detail=f"{repo.branch()} {where}")

    @staticmethod
    def _following(repo: Repo, follow: str) -> Step | str:
        """Move onto the parent's new pointer, saying so; a held step when it cannot."""
        if not follow or follow == repo.head():
            return ""
        verb = ("merge", "--ff-only") if repo.branch() else ("checkout", "--detach")
        moved = repo.git.run(*verb, "-q", follow)
        if not moved.succeeded:
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=said(moved))
        return f"followed the parent to {repo.short(follow)}"

    @staticmethod
    def _levelled(repo: Repo, upstream: str) -> Step | str:
        """Fast-forward onto `upstream` or merge it in, saying so; a held step if it stops."""
        ahead, behind = repo.counts(upstream)
        if ahead and behind:
            if stopped := repo.merge(upstream):
                detail = (
                    f"diverged from {upstream}: {stopped}; resolve with `git merge {upstream}`"
                )
                return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)
            return f"merged {behind} from {upstream}"
        if not behind:
            return ""
        merged = repo.git.run("merge", "--ff-only", "-q", upstream)
        if not merged.succeeded:
            return Step(repo=repo.name, outcome=Outcome.HELD, detail=said(merged))
        return f"fast-forwarded {behind} from {upstream}"

    @staticmethod
    def _reattached(repo: Repo) -> Step | str:
        """Put a detached HEAD back on its trunk, saying so; a held step when off its line."""
        if repo.branch():
            return ""
        if trunk := Pull._attached(repo):
            return f"attached to {trunk}"
        detail = f"detached at {repo.short('HEAD')}, off the line of {repo.trunk()}"
        return Step(repo=repo.name, outcome=Outcome.HELD, detail=detail)

    @staticmethod
    def _submodule(parent: Repo, *, child: Repo, done: str, flags: tuple[str, ...] = ()) -> Step:
        """Check `child` out at the pointer `parent` records, with git's own submodule update."""
        updated = parent.git.run(
            "submodule", "update", *flags, "--", parent.relative(child), network=True
        )
        if not updated.succeeded:
            return Step(repo=child.name, outcome=Outcome.FAILED, detail=said(updated))
        return Step(repo=child.name, outcome=Outcome.DONE, detail=f"{done} {child.short('HEAD')}")

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.steps: list[Step] = []
        self.unfetched: dict[str, str] = {}

    def _followed(self, parent: Repo, *, child: Repo, old: str) -> Step:
        """Move a foreign checkout to its parent's new pointer, if it sat on the old one."""
        if child.head() != old:
            detail = f"left at {child.short('HEAD')}, not the old pointer {old[:7]}"
            return Step(repo=child.name, outcome=Outcome.HELD, detail=detail)
        return self._submodule(parent, child=child, done="moved to")

    def _pulled(self, repo: Repo, follow: str) -> Step:
        """Move one owned repository forward: the parent's new pointer, its trunk, its upstream."""
        if complaint := self.unfetched.get(repo.name):
            return Step(repo=repo.name, outcome=Outcome.FAILED, detail=complaint)
        followed = Pull._following(repo, follow)
        if isinstance(followed, Step):
            return followed
        attached = Pull._reattached(repo)
        if isinstance(attached, Step):
            return attached
        return self._caught_up(repo, [note for note in (followed, attached) if note])

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
                step = self._submodule(repo, child=child, done="cloned at", flags=("--init",))
                self.steps.append(step)
            elif not child.owned and moved:
                self.steps.append(self._followed(repo, child=child, old=old))
            if child.initialized and child.owned:
                self._visit(child, follow=new if moved and child.head() == old else "")

    def run(self) -> list[Step]:
        """Fetch every owned repository at once, then walk the tree from the root down."""
        self.unfetched = self.tree.fetch(self.tree.owned())
        self._visit(self.tree.root, follow="")
        return self.steps
