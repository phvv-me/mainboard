"""The `ReconcileRow` data structure a reconcile pass builds.

A reconcile asks each backend's `Scheduler.state` what actually happened to a recorded run and
pairs the answer with the cached provenance; a future CLI builds one `ReconcileRow` per run from
the returned `JobState` and this module's row shape.
"""

from patos import FrozenModel


class ReconcileRow(FrozenModel):
    """One recorded run paired with its live scheduler state.

    handle: the run handle (PBS job id, pueue task id, SLURM job id).
    script: the submitted script name.
    submitted_at: when the run was dispatched (from the cache).
    name: a human label for the run, shown instead of the internal script path when set.
    state: the scheduler's current state string, or None if the job vanished.
    exit_code: the job's exit status, when the scheduler reports one.
    verdict: one of the `verdicts` vocabulary.
    """

    handle: str
    script: str
    submitted_at: str
    name: str = ""
    state: str | None = None
    exit_code: int | None = None
    verdict: str
