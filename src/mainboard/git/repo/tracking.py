from patos import FrozenModel, Runtime

from ..constants import REMOTE
from ..process import Git, said
from .commits import Commits

# The branch a repository with no declared branch and no remote HEAD is assumed to follow.
_DEFAULT_TRUNK = "main"


class Tracking(FrozenModel):
    """The branch HEAD is on, the trunk it follows and the upstream it is counted against.

    git: git run in the repository's working tree.
    commits: the repository's commits, which say how HEAD relates to the trunk.
    declared_branch: the branch the parent's `.gitmodules` says this submodule follows, empty
        when none.
    """

    git: Runtime[Git]
    commits: Commits
    declared_branch: str

    def attach(self) -> str:
        """Put a detached HEAD back on its trunk where that moves no commit, returning the branch.

        The trunk is reset to HEAD only when that fast-forwards the local branch and HEAD sits on
        the remote trunk's own line, behind it or ahead of it, so no commit is orphaned and the
        next push is not a divergence. The working tree is untouched, since HEAD stays where it
        is. Empty when either condition fails, leaving HEAD detached.
        """
        trunk = self.trunk()
        head = self.commits.head()
        local = f"refs/heads/{trunk}"
        remote = f"{REMOTE}/{trunk}"
        if self.commits.exists(local) and not self.commits.ancestor(local, newer=head):
            return ""
        tracked = self.commits.exists(remote)
        if tracked and not (
            self.commits.ancestor(remote, newer=head) or self.commits.ancestor(head, newer=remote)
        ):
            return ""
        self.git.out("checkout", "-q", "-B", trunk)
        if tracked:
            self.git.out("branch", "-q", f"--set-upstream-to={remote}")
        return trunk

    def branch(self) -> str:
        """The checked-out branch, empty when HEAD is detached."""
        found = self.git.run("symbolic-ref", "-q", "--short", "HEAD")
        return "" if found.returncode else found.stdout.strip()

    def merge(self, upstream: str) -> str:
        """Merge `upstream` into the checked-out branch, returning what stopped it, or nothing.

        A merge that conflicts is aborted rather than left half done, so the working tree is as
        it was and the conflicting paths are named for one `git merge` by hand.
        """
        merged = self.git.run("merge", "--no-edit", "-q", upstream)
        if not merged.returncode:
            return ""
        listing = self.git.run("diff", "--name-only", "-z", "--diff-filter=U").stdout
        conflicts = [path for path in listing.split("\0") if path]
        self.git.run("merge", "--abort")
        return f"conflicts in {', '.join(conflicts)}" if conflicts else said(merged)

    def trunk(self) -> str:
        """The branch this repository follows: declared by the parent, else the remote's HEAD."""
        if self.declared_branch:
            return self.declared_branch
        found = self.git.run("symbolic-ref", "-q", "--short", f"refs/remotes/{REMOTE}/HEAD")
        if found.returncode:
            return _DEFAULT_TRUNK
        return found.stdout.strip().removeprefix(f"{REMOTE}/")

    def upstream(self) -> str:
        """The remote branch HEAD is counted against, empty when there is none to count.

        A branch answers with the upstream it is configured to track and nothing when it tracks
        none, since guessing one would count against a branch it never meant to follow. A
        detached HEAD answers with its trunk on the remote, which is the branch it is attached to
        the first time a commit or a pull puts it back on one.
        """
        if self.branch():
            found = self.git.run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
            return "" if found.returncode else found.stdout.strip()
        remote = f"{REMOTE}/{self.trunk()}"
        return remote if self.commits.exists(remote) else ""
