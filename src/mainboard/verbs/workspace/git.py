from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter

from ...core.section import Verdict
from ...render import progress

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...git import Step


# The values a compact record leaves out: they say nothing an absent field does not.

from ..base import Verbs
from ..options import Output

# The verbs' default output and batch declaration.
_COMPACT = Output()


def _stepped(steps: Sequence[Step], output: Output, *, title: str) -> int:
    """Print one row per repository a tree verb walked, exiting 1 when any did not settle."""
    output.print_rows(
        [step.model_dump() for step in steps], title=title, columns=_GIT_STEP_COLUMNS
    )
    return 0 if all(step.outcome.settled for step in steps) else 1


_GIT_STEP_COLUMNS = ("repo", "outcome", "detail")


_GIT_STATUS_COLUMNS = (
    "repo",
    "owner",
    "branch",
    "head",
    "upstream",
    "ahead",
    "behind",
    "changed",
    "untracked",
    "published",
    "broken",
)


_GIT_CHECK_COLUMNS = ("repo", "check", "verdict", "detail")


class GitVerbs(Verbs):
    """The `git` verbs."""

    def git_check(self, *, output: Output = _COMPACT) -> int:
        """Verify the tree is safe to clone and push, and exit 1 when anything fails.

        Fetches every owned repository and the foreign submodules they point at, then reports
        each pointer no branch of its remote holds, each diverged branch, each file in HEAD over
        the size ceiling and each LFS repository with no git-lfs here as `fail`, and a detached
        HEAD, unpushed or missing commits, and a checkout off its recorded pointer as `warn`.
        An empty table is a consistent tree.

        Args:
        """
        with progress("checking the repository tree"):
            findings = self.session.board("local").git().check()
        output.print_rows(
            [finding.model_dump() for finding in findings],
            title="git check",
            columns=_GIT_CHECK_COLUMNS,
        )
        return 1 if any(finding.verdict is Verdict.FAIL for finding in findings) else 0

    def git_commit(
        self,
        *paths: Path,
        message: Annotated[list[str], Parameter(name=["--message", "-m"])],
        everything: Annotated[bool, Parameter(name=["--all", "-a"], negative="")] = False,
        output: Output = _COMPACT,
    ) -> int:
        """Commit the named paths, or what each owned repository has staged, submodules first.

        One run can land a different commit in each repository: `-m REPO=TEXT` is the message
        of the owned repository at path `REPO`, and a bare `-m TEXT` is everyone else's. A
        parent that only records pointers needs none and says which submodules moved.

        A path goes to the deepest owned repository holding it and is committed alone, as `git
        commit <path>` would: whatever else sits in that index stays there for whoever staged
        it. A directory takes every owned repository under it whole. A parent records the
        pointer of each submodule that committed in this run, and no other. Without paths each
        repository commits its index, and with nothing staged anywhere the command refuses.
        A named path under a `[git] never-commit` pattern, over the size ceiling Git LFS does
        not carry, a link edited through as a file, or a nested repository `.gitmodules` does not
        declare holds its repository with the reason. Each commit lands on a branch: a detached
        HEAD is attached to its trunk when that moves no commit, and held otherwise.

        `--all` is the deliberate sweep: every change in every owned repository except what has
        to stay out, links written out as files put back first, a parent held when
        its submodule did not commit, and an upstream the repository is behind merged in after.
        Exits 1 when any repository was held or failed.

        Args:
            paths: files or directories to commit, relative to the working directory.
            message: `TEXT` for every repository, or `REPO=TEXT` for one, repeated as needed.
            everything: commit every change in every owned repository.
        """
        with progress("committing the repository tree"):
            steps = self.session.board("local").git().commit(message, paths, everything=everything)
        return _stepped(steps, output, title="git commit")

    def git_pull(self, *, output: Output = _COMPACT) -> int:
        """Bring every owned repository level with its upstream and submodule checkouts along.

        Every owned remote is fetched at once, then the tree is walked from the root down. A
        branch behind fast-forwards and a diverged one merges its upstream; a merge that
        conflicts is aborted and held with the paths named, and nothing is ever rebased. Git
        itself refuses to overwrite local changes. A detached HEAD is put back
        on its trunk where that moves no commit. A submodule follows its parent's new pointer
        only when it sat on the old one, and one never checked out is cloned at the recorded
        pointer. Exits 1 when any repository was held or failed.

        Args:
        """
        with progress("pulling the repository tree"):
            steps = self.session.board("local").git().pull()
        return _stepped(steps, output, title="git pull")

    def git_push(self, *, output: Output = _COMPACT) -> int:
        """Push every owned repository, submodules before the parents that point at them.

        A parent is pushed only once every submodule pointer its HEAD records is held by a
        branch of that submodule's remote. Git LFS objects are uploaded first. A remote that
        protects the tracked branch gets the commit on a branch named `<tool>/<branch>` after
        this tool instead, and the row asks for the pull request. HTTPS pushes to GitHub can
        use the `gh` login as a credential. Exits 1 when any repository was held or failed.

        Args:
        """
        with progress("pushing the repository tree"):
            steps = self.session.board("local").git().push()
        return _stepped(steps, output, title="git push")

    def git_status(self, *, output: Output = _COMPACT) -> None:
        """Show every owned repository in the tree on one table, without touching the network.

        Owned means the owner in the remote URL is the workspace root's own or one `[git]
        owners` names; reference code pinned from anybody else is left out. Each row says the
        branch (or `detached`), how far HEAD is ahead of and behind its upstream as last
        fetched, how many paths are changed and untracked, and which remote branch already
        holds HEAD, empty for a commit a parent pointer could not yet be cloned at. `broken`
        carries git's own words for a submodule checkout it cannot read, the one a bare `git
        status` aborts on, which this verb survives.

        Args:
        """
        with progress("reading the repository tree"):
            states = self.session.board("local").git().status()
        output.print_rows(
            [state.model_dump() for state in states],
            title="git status",
            columns=_GIT_STATUS_COLUMNS,
        )
