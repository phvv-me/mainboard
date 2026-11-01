from patos import FrozenModel, Runtime

from ..process import Git


class Commits(FrozenModel):
    """The commits of one repository: where HEAD is and how two commits relate.

    git: git run in the repository's working tree.
    """

    git: Runtime[Git]

    def ancestor(self, older: str, *, newer: str) -> bool:
        """Whether `older` is `newer` or an ancestor of it, so one fast-forwards to the other."""
        return self.git.ok("merge-base", "--is-ancestor", older, newer)

    def counts(self, upstream: str) -> tuple[int, int]:
        """How many commits HEAD is ahead of and behind `upstream`, zero both without one."""
        if not upstream:
            return 0, 0
        counted = self.git.line("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
        ahead, behind = counted.split()
        return int(ahead), int(behind)

    def exists(self, ref: str) -> bool:
        """Whether `ref` names a commit this repository has."""
        return self.git.ok("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")

    def head(self) -> str:
        """The commit HEAD is on."""
        return self.git.line("rev-parse", "HEAD")

    def short(self, commit: str) -> str:
        """`commit` abbreviated the way git would print it here."""
        return self.git.line("rev-parse", "--short", commit)
