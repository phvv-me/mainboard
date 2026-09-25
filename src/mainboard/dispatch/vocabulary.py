# What a dispatched job is described with: a resource request, a post-mortem state, and the
# one-word verdict lifecycle. It names no scheduler and no provider on purpose, so a provider
# backend and a queue backend speak it without importing each other's family.

from patos import FrozenModel, Lifecycle
from pydantic import Field

from .shared import HandleId

POLL_SECONDS = 5.0
# How long a `wait` blocks before handing the shell back with the job still in flight. Finite
# because an unbounded waiter once sat two days on a job owed by a host that had left the
# network; 0 means forever.
WAIT_SECONDS = 3600.0

QUEUED = "queued"
RUNNING = "running"
PREPARED = "prepared"
SUBMITTING = "submitting"
# The third live word, which no backend reports itself: the queue is done with the job and the
# sweep has not brought it home yet (see `JobState.phase`).
FINISHED = "finished"
OK = "ok"
FAILED = "failed"
VANISHED = "vanished"
UNKNOWN = "unknown"
TIMEOUT = "timeout"
# A job somebody stopped on purpose: a decision, not something that happened to the run like
# `failed` or `vanished`, since a doomed job killed at minute three is a success of judgment and a
# table filing it beside a crash teaches distrust of the column. Reachable from both live states,
# since cancelling does not wait for the job to start.
CANCELLED = "cancelled"
# A dispatch a target refused for its job quota, kept here until the quota has room. Not a
# verdict about the unstarted work, so it stays out of `TERMINAL` and each sweep asks again.
HELD = "held"
# A job a plan declared and a run was told to leave out: a decision like `cancelled`, terminal
# from the start since nothing was dispatched and no watch may wait on it.
SKIPPED = "skipped"

# Every terminal maps to the empty set, so a further move (a stale `running` after `ok`) raises
# rather than mutates.
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

# Settled for good, so a durable sweep trusts the cache instead of a queue that may have forgotten.
TERMINAL = frozenset(verdict for verdict, moves in VERDICTS.items() if not moves)


def tracker(initial: str = QUEUED) -> Lifecycle[str]:
    return Lifecycle(VERDICTS, initial)


class Resources(FrozenModel):
    """A backend-agnostic resource request for one job.

    Each backend maps these onto its own flags (`-l select=` for PBS, `--gpus`/`--mem` for
    SLURM, an instance type for a rental provider) and ignores what it can't express.

    gpus: 0 means none, so CPU-only scripts run on clusters without GPU GRES.
    gpu_name: the GPU type (`H100`, `A10G`) a provider backend needs; ssh-family schedulers
        ignore it.
    nodes: nodes/chunks the request spans.
    walltime: `HH:MM:SS`, when capped.
    queue: scheduler queue/partition name.
    account: charging account / group list.
    container: the container the job runs under, when the profile is containerized.
    mem_gb: system (not GPU) memory.
    max_usd: the spend cap a provider backend requires before submitting (0.0 means unset);
        ssh-family schedulers, on owned hardware, ignore it.
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

    A quota refusal means "not now", so a workstation keeps the request and the durable sweep
    resubmits it when the quota has room. Every field is therefore the caller's own ask, never a
    resolved value: profile defaults, queue policy and market price are read again at the retry,
    so a request held overnight lands under the morning's manifest. Provider creation intents
    also retain this model, with resolved values, for recovery rather than resubmission.

    name / node / fetch: the run's label, the ledger slug it serves, and the results path to
        pull back.
    needs: the data paths a job spelled by file was asked to reach on the host.
    queue / walltime / mem_gb / gpus / gpu_name / max_usd / nodes: unset fields fall back to the
        host profile's defaults at retry time.
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

    A verdict says `running` for everything in flight, right for a lifecycle and wrong for a
    table: an operator watching a wave of thirty five needs two running and thirty three waiting,
    the same fact on PBS, SLURM and pueue. So `stage`, `since`, `estimated_start` and `note` name
    it once here rather than each reader parsing each backend's spelling.

    handle: PBS/SLURM job id, pueue task id or provider run id, always text.
    state: the backend's current state string, None when the job vanished.
    verdict: one word, `ok` / `failed` / `running` / `vanished` / `unknown` / `timeout`.
    stage: `queued` or `running` for a job in flight, mapped from the backend's own word; empty
        where a backend reports neither.
    since: ISO-8601 instant the job entered that stage, empty when unknown.
    estimated_start: ISO-8601 instant the backend expects a queued job to start, usually empty.
    note: why a queued job has not started, in the backend's words (on PBS the resource its
        queue is short of); empty once it runs and on backends that say nothing.
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

        `stage` while in flight. A terminal verdict not yet brought home is `finished`, rather
        than the backend's letter (`F`/`E` on PBS, `CD`/`TO`/`CA` on SLURM, `Done` in pueue):
        printing `F` beside `queued` and `running` got a clean PBS job read as failed (handle
        3294174, 2026-09-04). The verdict lands on the row one sweep later. A live state with no
        stage keeps the backend's own word.
        """
        if self.stage:
            return self.stage
        if self.verdict in TERMINAL:
            return FINISHED
        return self.state or ""
