# What a dispatched job is described with, shared by everything that dispatches one. A resource
# request, a post-mortem state, and the one-word verdict lifecycle those states report through.
#
# This module names no scheduler and no provider, which is the point: a provider backend asks for
# a `Resources` and answers with a `JobState` exactly as a queue backend does, and neither has to
# import the other's family to speak the common language.

from patos import FrozenModel, Lifecycle
from pydantic import Field

from .shared import HandleId

# Seconds between polls while a caller waits on a job it dispatched.
POLL_SECONDS = 5.0

QUEUED = "queued"
RUNNING = "running"
PREPARED = "prepared"
SUBMITTING = "submitting"
# The third live word, and the only one no backend reports for itself: the queue is done with the
# job and the durable sweep has not brought it home yet. Without it a listing printed whatever
# letter the backend spells that moment with, so a PBS job that finished clean showed as `F`
# beside `queued` and `running` and was read as failed (handle 3294174, 2026-09-04).
FINISHED = "finished"
OK = "ok"
FAILED = "failed"
VANISHED = "vanished"
UNKNOWN = "unknown"
TIMEOUT = "timeout"
# A job somebody stopped on purpose. It is its own word rather than `failed` or `vanished`
# because those two are things that happened to a run, and this is a decision about it: a
# provably doomed job killed at minute three is a success of judgment, and a table that files it
# beside a crash teaches a reader to distrust the column. It is reachable from both live states,
# since the whole point of cancelling is that it does not wait for the job to start.
CANCELLED = "cancelled"
# A dispatch a target refused for having too many jobs already, kept at this workstation until
# the quota has room. It is not a verdict about the work, which has not started: it is where the
# request is waiting, so it stays out of `TERMINAL` and a sweep asks again on its next pass.
HELD = "held"
# A job a plan declared and a run was told to leave out. Like `cancelled` it is a decision rather
# than something that happened, and it is terminal from the start: nothing was ever dispatched, so
# nothing about it can move and no watch may wait on it.
SKIPPED = "skipped"

# Declared edges: held -> queued once a quota lets it through, queued -> running/vanished/
# cancelled, running -> one terminal. Every terminal maps to the empty set, so a further move (a
# stale `running` after `ok`) raises rather than mutates.
VERDICTS: dict[str, set[str]] = {
    PREPARED: {SUBMITTING, FAILED, CANCELLED},
    SUBMITTING: {QUEUED, RUNNING},
    HELD: {QUEUED, RUNNING, FAILED, VANISHED, CANCELLED},
    QUEUED: {RUNNING, VANISHED, CANCELLED},
    RUNNING: {OK, FAILED, VANISHED, TIMEOUT, CANCELLED},
    OK: set(),
    FAILED: set(),
    VANISHED: set(),
    UNKNOWN: set(),
    TIMEOUT: set(),
    CANCELLED: set(),
    SKIPPED: set(),
}


# The verdicts no declared move can leave. A job that reached one is settled for good, so a
# durable sweep trusts it straight from the cache instead of asking a queue that may already
# have forgotten the job.
TERMINAL = frozenset(verdict for verdict, moves in VERDICTS.items() if not moves)


def tracker(initial: str = QUEUED) -> Lifecycle[str]:
    """A fresh `Lifecycle` over the verdict table, started at `initial`."""
    return Lifecycle(VERDICTS, initial)


class Resources(FrozenModel):
    """A backend-agnostic resource request for one job.

    Each backend maps these onto its own flags (`-l select=` for PBS, `--gpus`/`--mem` for
    SLURM, an instance type for a rental provider) and ignores what it can't express.

    gpus: number of GPUs to request (0 means none, so CPU-only scripts run on clusters without
        GPU GRES).
    gpu_name: the requested GPU type (`H100`, `A10G`), when a provider backend needs a name
        rather than a bare count; ignored by the ssh-family schedulers.
    nodes: number of nodes/chunks the resource request spans.
    walltime: requested walltime as `HH:MM:SS`, when capped.
    queue: scheduler queue/partition name.
    account: charging account / group list.
    container: a container name the job runs under, when the profile is containerized.
    mem_gb: system memory request in GB.
    max_usd: the explicit spend cap a provider backend must see before it will submit at all
        (0.0 means unset); ignored by the ssh-family schedulers, which run on owned hardware.
    """

    gpus: int = 0
    gpu_name: str = ""
    nodes: int = 1
    walltime: str | None = None
    queue: str | None = None
    account: str = ""
    container: str = ""
    mem_gb: int | None = None
    max_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class Request(FrozenModel):
    """One dispatch as it was asked for, enough to ask for it again unchanged.

    What a target refuses on a count or concurrency quota is not a rejection of the work, it is
    "not now", so the request is what a workstation keeps rather than the refusal. A held
    request is resubmitted by the durable sweep whenever the quota next has room, which is why
    every field here is the caller's own ask and none of them is a resolved value: the profile's
    defaults, the queue policy and the market price are all read again at the retry, so a
    request held overnight lands under whatever the manifest says in the morning. Provider
    creation intents also retain this model, with resolved values for recovery rather than
    automatic resubmission.

    target: the host alias the job is for.
    command: the command the job runs.
    name / node / fetch: the run's label, the ledger slug it serves, and the results path to
        pull back, exactly as the original dispatch gave them.
    needs: the data paths a job spelled by file was asked to reach on the host.
    env / container: the environment and container overrides the dispatch asked for.
    queue / walltime / mem_gb / gpus / gpu_name / max_usd / nodes: the resource request, unset
        fields falling back to the host profile's declared defaults at retry time.
    attempt: the 1-based try number the profile's expression defaults are evaluated against.
    """

    target: str
    command: str
    name: str = ""
    node: str = ""
    fetch: str | None = None
    needs: tuple[str, ...] = ()
    env: str = ""
    container: str = ""
    queue: str = ""
    walltime: str = ""
    mem_gb: int = 0
    gpus: int = 0
    gpu_name: str = ""
    max_usd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    nodes: int = 1
    attempt: int = 1


class JobState(FrozenModel):
    """A job's post-mortem state, the unit reconcile compares against the cache.

    The last three fields are what a listing shows about a job that has not ended yet. A verdict
    says `running` for everything still in flight, which is the right word for a lifecycle and
    the wrong one for a table: an operator watching a wave of thirty five needs to see that two
    are running and thirty three are waiting behind them, and that is the same fact on PBS, on
    SLURM and in a pueue queue. So the two live words, when each began, and where a backend
    estimates a start, are named here once rather than parsed out of each backend's own spelling
    by whoever reads it.

    handle: the backend's job handle (PBS job id, pueue task id, SLURM job id, a provider run
        id), always text even when its backend reports a bare number.
    label: the job's name/label, when the backend reports one.
    state: the backend's current state string, or None when the job vanished.
    exit_code: the process exit status, when the backend reports one.
    verdict: one word, `ok` / `failed` / `running` / `vanished` / `unknown` / `timeout`.
    stage: `queued` or `running` for a job still in flight, the backend's own state word mapped
        onto the two this vocabulary names; empty where a backend reports neither.
    since: ISO-8601 instant the backend says the job entered that stage, empty where it says
        nothing about when.
    estimated_start: ISO-8601 instant the backend expects a queued job to start, empty where it
        estimates none, which most backends do most of the time.
    note: why a queued job has not started, in the backend's words (on PBS the resource its
        queue is short of); empty once it runs and on every backend that says nothing.
    """

    handle: HandleId
    label: str | None = None
    state: str | None = None
    exit_code: int | None = None
    verdict: str
    stage: str = ""
    since: str = ""
    estimated_start: str = ""
    note: str = ""

    @property
    def phase(self) -> str:
        """The one word a listing shows for a job it has just asked its backend about.

        The backend's own `stage` while the job is in flight, since `queued` and `running` are
        the distinction a person reads the table for. Past that every backend has a moment the
        queue is done with the job and the sweep has not settled it yet, and each spells it with
        a different letter, `F` or `E` on PBS, `CD`, `TO` or `CA` on SLURM, `Done` in a pueue
        status. Printing the letter is how a job that finished clean was read as failed, so the
        moment is named once here rather than mapped three times: a terminal verdict nobody has
        brought home yet is `finished`, and the verdict itself lands on the row one sweep later.

        A live state a backend reports no stage for keeps that backend's own word, which is
        still more than nothing while a backend has yet to map its live states onto the two.
        """
        if self.stage:
            return self.stage
        if self.verdict in TERMINAL:
            return FINISHED
        return self.state or ""
