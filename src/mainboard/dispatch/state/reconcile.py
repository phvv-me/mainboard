"""The row a reconcile pass builds: a backend's `Scheduler.state` beside cached provenance."""

from patos import FrozenModel


class ReconcileRow(FrozenModel):
    """One recorded run paired with its live scheduler state.

    handle: the PBS job id, pueue task id or SLURM job id.
    name: a human label, shown instead of the internal script path when set.
    state: the scheduler's current state string, None if the job vanished.
    verdict: one of the `verdicts` vocabulary.
    """

    handle: str
    script: str
    submitted_at: str
    name: str = ""
    state: str | None = None
    exit_code: int | None = None
    verdict: str
