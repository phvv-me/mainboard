# What a dispatched job is described with, shared by everything that dispatches one. A resource
# request, a post-mortem state, and the one-word verdict lifecycle those states report through.
#
# This module names no scheduler and no provider, which is the point: a provider backend asks for
# a `Resources` and answers with a `JobState` exactly as a queue backend does, and neither has to
# import the other's family to speak the common language.

from patos import FrozenModel, Lifecycle

from .shared import HandleId

# Seconds between polls while a caller waits on a job it dispatched.
POLL_SECONDS = 5.0

QUEUED = "queued"
RUNNING = "running"
OK = "ok"
FAILED = "failed"
VANISHED = "vanished"
UNKNOWN = "unknown"
TIMEOUT = "timeout"

# Declared edges: queued -> running/vanished, running -> one terminal. Every terminal maps to
# the empty set, so a further move (a stale `running` after `ok`) raises rather than mutates.
VERDICTS: dict[str, set[str]] = {
    QUEUED: {RUNNING, VANISHED},
    RUNNING: {OK, FAILED, VANISHED, TIMEOUT},
    OK: set(),
    FAILED: set(),
    VANISHED: set(),
    UNKNOWN: set(),
    TIMEOUT: set(),
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
    max_usd: float = 0.0


class JobState(FrozenModel):
    """A job's post-mortem state, the unit reconcile compares against the cache.

    handle: the backend's job handle (PBS job id, pueue task id, SLURM job id, a provider run
        id), always text even when its backend reports a bare number.
    label: the job's name/label, when the backend reports one.
    state: the backend's current state string, or None when the job vanished.
    exit_code: the process exit status, when the backend reports one.
    verdict: one word, `ok` / `failed` / `running` / `vanished` / `unknown` / `timeout`.
    """

    handle: HandleId
    label: str | None = None
    state: str | None = None
    exit_code: int | None = None
    verdict: str
