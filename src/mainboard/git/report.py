from enum import StrEnum, auto

from patos import FrozenModel

from ..doctor import Verdict


class Outcome(StrEnum):
    """What one verb did to one repository.

    `done` changed something, `current` had nothing to change, `held` was refused by a rule
    that protects work (a diverged branch, a pointer the remote cannot serve, a submodule that
    did not get there first), and `failed` is git refusing on its own terms.
    """

    DONE = auto()
    CURRENT = auto()
    HELD = auto()
    FAILED = auto()

    @property
    def settled(self) -> bool:
        """Whether the repository reached what the verb asked of it."""
        return self in {Outcome.DONE, Outcome.CURRENT}


class Step(FrozenModel):
    """One repository's line in a pull, commit or push.

    repo: the repository's workspace-relative path, `.` for the root.
    outcome: what happened to it.
    detail: the commit, the branch or git's own words behind the outcome.
    """

    repo: str
    outcome: Outcome
    detail: str = ""


class RepoState(FrozenModel):
    """One owned repository as `status` reads it, without touching the network.

    repo: the repository's workspace-relative path, `.` for the root.
    owner: the owner its remote URL names.
    branch: the checked-out branch, `detached` when HEAD is on no branch.
    head: the short commit HEAD is on.
    upstream: the remote branch counted against, a detached HEAD's trunk on the remote.
    ahead: commits HEAD has that the upstream does not.
    behind: commits the upstream has that HEAD does not.
    changed: tracked paths changed, staged or not, a moved submodule pointer included.
    untracked: files git does not track yet. Neither count includes a `never-commit` path, so
        both say what the next commit would take.
    published: a remote branch already holding HEAD, empty when none does, which is a commit a
        parent pointer cannot yet be fetched at.
    """

    repo: str
    owner: str
    branch: str
    head: str
    upstream: str
    ahead: int
    behind: int
    changed: int
    untracked: int
    published: str


class Finding(FrozenModel):
    """One inconsistency `check` found in the tree.

    repo: the repository's workspace-relative path, `.` for the root.
    check: the rule it breaks.
    verdict: `fail` for what would break a clone or a push, `warn` for what wants attention.
    detail: what exactly, with the path or commit involved.
    """

    repo: str
    check: str
    verdict: Verdict
    detail: str
