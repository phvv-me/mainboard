# The `Scheduler` contract every job backend implements, plus the polling/failure vocabulary
# every backend shares. New backends are new classes, never new `if kind == ...` branches.

import re
import shlex
from time import sleep
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from patos import FrozenModel, IllegalTransition, Lifecycle
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity import wait_fixed as tenacity_wait_fixed

from .. import verdicts as vocabulary
from ..shared import HandleId, logger, state_dir
from ..transport import HostUnreachable, is_transport_failure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..transport import Machine

# A handle is in flight while its verdict is `running`; every other verdict is terminal.
POLL_SECONDS = 5.0

# Consecutive unreachable probes absorbed before a wait gives up: rides out a blip while a
# genuinely down host still surfaces well under a minute.
_MAX_PROBE_RETRIES = 12


def login_run(remote: Machine, body: str) -> str:
    """Run `body` in a login shell on `remote` and return its stdout.

    The single chokepoint every scheduler probe shares. It captures the ssh exit status so a
    transport failure (exit 255 with a transport phrase in stderr) raises `HostUnreachable`
    instead of yielding the empty output a parser reads as a vanished job, which is exactly how a
    refused ssh session used to end a wait early. A command that genuinely ran and exited non-zero
    (`qstat` reporting an unknown id) returns its stdout unchanged.
    """
    retcode, out, err = remote["bash"][["-lc", body]].run(retcode=None)
    if is_transport_failure(retcode, err):
        raise HostUnreachable(err.strip()[-200:] or "ssh transport failure")
    return out


class Resources(FrozenModel):
    """A scheduler-agnostic resource request for one job.

    Each backend maps these onto its own flags (`-l select=` for PBS, `--gpus`/`--mem` for
    SLURM) and ignores what it can't express.

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

    handle: the scheduler's job handle (PBS job id, pueue task id, SLURM job id), always text
        even when its scheduler reports a bare number.
    label: the job's name/label, when the scheduler reports one.
    state: the scheduler's current state string, or None when the job vanished.
    exit_code: the process exit status, when the scheduler reports one.
    verdict: one word, `ok` / `failed` / `running` / `vanished` / `unknown` / `timeout`.
    """

    handle: HandleId
    label: str | None = None
    state: str | None = None
    exit_code: int | None = None
    verdict: str


@runtime_checkable
class Scheduler(Protocol):
    """A pluggable job backend dispatched to generically.

    `remote` is an open plumbum `SshMachine` (or `local`); `root` is the workspace path on the
    host. Implementations are stateless value objects, so one instance per kind is enough.
    """

    name: str

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        """Cancel `handle` on the host."""

    def jobs(self, remote: Machine, root: str) -> list[JobState]:
        """List the user's live/queued jobs on the host as structured states."""

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        """`handle`'s captured log so far (merged stdout+stderr)."""

    def queues(self, remote: Machine, root: str) -> list[str]:
        """The scheduler's own queue list (PBS queues, SLURM partitions).

        Each queue is a node class an onboarding probe can size with a minimal job; a backend
        without a queue concept (pueue, bare bash) returns `[]`.
        """

    def revive(self, remote: Machine, root: str) -> list[str]:
        """Restart the host's scheduler daemon, recovering a dead queue.

        The companion to the `unreachable: daemon down` verdict: when a backend owns a
        user-managed daemon and it died, this brings it back so jobs resolve again, returning the
        handles of any zombie tasks it had to clear on the way back. A backend whose scheduler is
        site-managed (PBS, SLURM) or has no daemon (bare bash) has nothing to revive and says so.
        """

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict, for reconcile."""

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """The state of `handles` (and any other live job) in one batched query, keyed by handle.

        One round-trip so a whole host's pending runs resolve at once instead of one probe per
        run. A handle the host no longer remembers is simply absent; the caller falls back to
        its cached verdict.
        """

    def stream(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """Print `handle`'s log as it grows until the job is terminal; return its final state."""

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
    ) -> str:
        """Launch `script` with `args` under `resources`; return the job handle."""

    def wait(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """Block until `handle` leaves the running/queued states, returning its final state.

        A backend with no queue (`Local`) already ran the job to completion, so it returns at
        once.
        """


def resilient(
    probe: Callable[[], JobState],
    *,
    interval: float = POLL_SECONDS,
    sleeper: Callable[[float], None] = sleep,
    retries: int = _MAX_PROBE_RETRIES,
) -> Callable[[], JobState]:
    """Wrap `probe` so a `HostUnreachable` is retried instead of surfacing as a verdict.

    A transport blip (a refused ssh session, a dropped link) is not an answer about the job, so
    tenacity retries the probe at a fixed `interval` up to `retries` times before re-raising the
    original error for a genuinely down host. Only `HostUnreachable` is retried; a real
    `JobState` (`running` included) returns at once. `sleeper` is injected so a test drives the
    backoff without real time passing.
    """

    def note(state: RetryCallState) -> None:
        logger.warning("host unreachable, retry %d/%d", state.attempt_number, retries)

    retrying = Retrying(
        retry=retry_if_exception_type(HostUnreachable),
        stop=stop_after_attempt(retries),
        wait=tenacity_wait_fixed(interval),
        sleep=sleeper,
        reraise=True,
        before_sleep=note,
    )
    return lambda: retrying(probe)


def _settle(verdicts: Lifecycle[str], observed: str) -> str:
    """Advance `verdicts` to `observed`, forgiving only a first poll that skips `running`.

    `queued` is a placeholder for "not yet observed", not a real report from the scheduler, so a
    fast job that already finished by the first poll (or a scheduler that never reports an
    intermediate `running`) legitimately lands straight on a terminal; that is the one
    illegal-looking edge honest polling produces, absorbed here by forcing the tracker onto the
    fresh verdict. A same-state re-read is a no-op. Any other illegal move, most importantly a
    settled terminal sliding back to `running`, is a real regression and is left to raise.
    """
    if observed == verdicts.current:
        return observed
    try:
        verdicts.to(observed)
    except IllegalTransition:
        if verdicts.current != vocabulary.QUEUED:
            raise
        verdicts.current = observed
    return observed


def poll_until_done(
    probe: Callable[[], JobState],
    *,
    interval: float = POLL_SECONDS,
    sleeper: Callable[[float], None] = sleep,
    retries: int = _MAX_PROBE_RETRIES,
) -> JobState:
    """Poll `probe` until the returned `JobState` is terminal, returning it.

    The shared body of every queued backend's `wait`: a job is terminal once its verdict leaves
    `running`. The probe is made `resilient` first, so a transient ssh blip is retried rather
    than ending the wait on a false `vanished`. Every observed verdict is routed through a
    `Lifecycle` tracker (see `verdicts`), so a scheduler that lies about a job's history (a
    finished job read back as running) raises instead of silently misleading the wait.
    """
    probe = resilient(probe, interval=interval, sleeper=sleeper, retries=retries)
    verdicts = vocabulary.tracker()
    while _settle(verdicts, (state := probe()).verdict) == vocabulary.RUNNING:
        sleeper(interval)
    return state


def stream_until_done(
    probe: Callable[[], JobState],
    drain: Callable[[int], int],
    *,
    interval: float = POLL_SECONDS,
    sleeper: Callable[[float], None] = sleep,
    retries: int = _MAX_PROBE_RETRIES,
) -> JobState:
    """Poll `probe`, printing new log content between polls, until the job is terminal.

    The shared body of the queued backends' `stream`: each tick checks the job's state, drains
    whatever the log grew since the last byte offset, and sleeps. After the terminal state one
    final drain catches output flushed between the last tick and the job's end.

    probe: returns the job's current `JobState`.
    drain: prints log content from the given byte offset, returning the bytes read.
    interval: seconds between polls.
    sleeper: injected so a test drives the loop without real time passing.
    """
    probe = resilient(probe, interval=interval, sleeper=sleeper, retries=retries)
    verdicts = vocabulary.tracker()
    offset = 0
    while _settle(verdicts, (state := probe()).verdict) == vocabulary.RUNNING:
        offset += drain(offset)
        sleeper(interval)
    drain(offset)
    return state


def log_path(root: str, *, handle: str) -> str:
    """The captured merged stdout+stderr path a rendered job script writes for `handle`.

    Both the PBS and bash job templates write to this same `{STATE_DIR}/logs/<stem>.log` path
    (the PBS stem drops the `.<server>` suffix `qstat` appends), so a backend can read a job's
    output straight off the host filesystem without an on-host executor in the loop.
    """
    stem = handle.split(".", maxsplit=1)[0]
    return f"{root}/{state_dir()}/logs/{stem}.log"


def read_log(remote: Machine, root: str, *, handle: str, offset: int = 0) -> str:
    """`handle`'s captured log from byte `offset` on, as a string."""
    path = shlex.quote(log_path(root, handle=handle))
    body = f"tail -c +{offset + 1} {path} 2>/dev/null"
    return str(remote["bash"][["-lc", body]](retcode=None))


def drain_log(remote: Machine, root: str, *, handle: str, offset: int) -> int:
    """Print `handle`'s captured log from byte `offset` on; return the bytes consumed."""
    chunk = read_log(remote, root, handle=handle, offset=offset)
    print(chunk, end="", flush=True)
    return len(chunk.encode())


# Failure markers in priority order: the walltime-kill verdict, then a raised Python exception,
# then a scheduler rejection, then a generic build/runtime error.
_FAILURE_MARKERS = (
    re.compile(r"^mainboard: killed at walltime.*", re.MULTILINE),
    re.compile(r"^\w[\w.]*(?:Error|Exception|Interrupt|Killed)\b.*", re.MULTILINE),
    re.compile(r"^(?:qsub|sbatch|srun|pueue):.*", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"^.*(?:fatal error|error:|failed to build|No such file|out of memory|cuda error).*",
        re.IGNORECASE | re.MULTILINE,
    ),
)

# Terminal control noise (ANSI escapes, box-drawing glyphs) a rich-UI log carries, stripped so
# a triage excerpt never quotes a panel border as the cause.
_ANSI_CODES = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX_DRAWING = re.compile(r"[─-▟]+")

# Exit codes that tell their own story with no traceback: signal N exits 128+N, so 137 is
# SIGKILL (OOM or walltime), 143 is SIGTERM, 124 is GNU `timeout`'s deadline code.
_SIGNAL_EXITS = {
    124: "timed out (walltime exceeded)",
    125: "timeout failed to start the job",
    137: "killed by SIGKILL (out of memory or walltime, exit 137)",
    139: "crashed with SIGSEGV (segfault, exit 139)",
    143: "terminated by SIGTERM (walltime or cancel, exit 143)",
}


def exit_reason(exit_code: int | None) -> str | None:
    """A human reason for an externally-imposed exit code, or None for a plain non-zero exit."""
    return _SIGNAL_EXITS.get(exit_code) if exit_code is not None else None


def failure_reason(log: str, exit_code: int | None = None) -> str:
    """One-line best-effort cause of a failed job, from its captured log and exit code."""
    for pattern in _FAILURE_MARKERS:
        matches: list[str] = pattern.findall(log)
        if matches:
            return matches[-1].strip()[:240]
    if reason := exit_reason(exit_code):
        return reason
    lines = meaningful_lines(log)
    return lines[-1][:240] if lines else "(no log output)"


def meaningful_lines(log: str) -> list[str]:
    """The log's content lines: ANSI codes and rich panel borders stripped, blanks dropped."""
    stripped = (
        _BOX_DRAWING.sub(" ", _ANSI_CODES.sub("", raw)).strip() for raw in log.splitlines()
    )
    return [line for line in stripped if line]


def log_excerpt(log: str, limit: int = 10) -> list[str]:
    """The last `limit` meaningful log lines, the tail a triage view prints under its verdict."""
    return meaningful_lines(log)[-limit:]


def short_reason(verdict: str, exit_code: int | None) -> str:
    """A short, network-free cause for a non-ok terminal verdict, from its cached state alone."""
    if verdict == vocabulary.VANISHED:
        return "vanished (the scheduler no longer remembers the job)"
    if (known := exit_reason(exit_code)) is not None:
        return known
    if exit_code is not None:
        return f"exited {exit_code}"
    return "failed"


def verdict_line(state: JobState, *, submitted_age: str = "") -> str:
    """The one structured verdict line a triage view leads with, before any log excerpt."""
    details: list[str] = []
    if state.exit_code is not None:
        details.append(f"exit {state.exit_code}")
        if (known := exit_reason(state.exit_code)) is not None:
            details.append(known)
    if submitted_age:
        details.append(f"submitted {submitted_age}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{state.handle} {state.verdict}{suffix}"
