# The `Scheduler` contract every job backend implements, plus the log-reading and failure-triage
# vocabulary they share; new backends are new classes, never `if kind == ...` branches. Requests,
# job states and verdicts live in `dispatch.vocabulary`, since provider backends speak them too.

import re
import shlex
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...core.project import Project
from .. import vocabulary
from ..shared import since, state_dir
from ..transport import HostUnreachable, is_transport_failure
from ..vocabulary import JobState, Resources

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine


def login_run(remote: Machine, body: str) -> str:
    """Run `body` in a login shell on `remote` and return its stdout; every probe goes here.

    A transport failure (exit 255, a transport phrase in stderr) raises `HostUnreachable` rather
    than yield the empty output a parser reads as a vanished job, which is how a refused ssh
    session used to end a wait early. A command that ran and exited non-zero (`qstat` on an
    unknown id) returns its stdout unchanged.
    """
    retcode, out, err = remote["bash"][["-lc", body]].run(retcode=None)
    if is_transport_failure(retcode, err):
        raise HostUnreachable(err.strip()[-200:] or "ssh transport failure")
    return out


def within(root: str, command: str) -> str:
    """`command` run from `root`, the one way a backend enters the tree a dispatch pinned it to.

    Every ssh-reached scheduler takes the job's working directory from where it was submitted:
    PBS exports it as `PBS_O_WORKDIR` (the generated script cds there), sbatch hands it to the
    job, a bare bash host stands there, and pueue takes it as a flag. Staged script paths are
    workspace-relative and the login home is not the root (a home that happened to be the root
    hid this; Miyabi's /work root did not). `root` is the dispatch's snapshot of the mirror, which
    keeps the job's code immutable while it runs.
    """
    return f"cd {shlex.quote(root)} && {command}"


@runtime_checkable
class Scheduler(Protocol):
    """A pluggable job backend, one stateless instance per kind.

    `remote` is an open plumbum `SshMachine` (or `local`); `root` is the host's workspace path.
    """

    name: str

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        """Cancel `handle` on the host."""

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        """The one command an interactive session runs inside the caller's ssh and staging.

        A queued backend asks for an interactive allocation (`qsub -I`, `srun --pty`); a host that
        runs the work itself hands the terminal to its own tool.

        command: run instead of handing over the terminal, empty for a session.
        resources: the allocation a queued backend asks for.
        """

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        """`handle`'s captured log so far (merged stdout+stderr)."""

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict, for reconcile."""

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """`handles` (and any other live job) in one round trip, keyed by handle.

        A handle the host no longer remembers may be absent; the caller falls back to `state`.
        """

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


def workspace_session(*, env: str, command: Sequence[str], resources: Resources) -> str:
    """The interactive line for a host that runs the work itself, with no queue in between.

    The host's own tool (`shell`, or `run` for a command) owns the activation, so an interactive
    session and a dispatched job never disagree about their interpreter. `resources` is ignored:
    such a host allocates nothing.
    """
    del resources
    tool = Project().name
    if command:
        return shlex.join([tool, "run", "--env", env, "--", *command])
    return shlex.join([tool, "shell", env])


def log_path(root: str, *, handle: str) -> str:
    """The merged stdout+stderr a PBS runner and SLURM's `--output` both write for `handle`."""
    return f"{root}/{state_dir()}/logs/{bare(handle)}.log"


def bare(handle: str) -> str:
    """A handle's bare job number: `2435326.opbs` and `2435326` both -> `2435326`.

    `qsub` prints a bare id on some wrappers and `<id>.<server>` elsewhere while `qstat -f` always
    reports the full id, so every lookup joins on the bare number.
    """
    return handle.split(".", maxsplit=1)[0]


def read_log(remote: Machine, root: str, *, handle: str, offset: int = 0) -> str:
    """`handle`'s captured log from byte `offset` on, as a string."""
    path = shlex.quote(log_path(root, handle=handle))
    body = f"tail -c +{offset + 1} {path} 2>/dev/null"
    return str(remote["bash"][["-lc", body]](retcode=None))


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


# A refusal for the count of jobs already queued, not for the job: PBS answers rc=39 with `would
# exceed group <g>'s limit on resource njobs-g` (Miyabi 2026-09-04 dropped four jobs of a thirteen
# job wave), SLURM names the association or QOS limit. That is "not now", so the dispatch is held
# and asked again. Every marker names a COUNT: a bad queue, a walltime over the ceiling or an
# account without permission is a real rejection, which re-asking would repeat forever.
_QUOTA_MARKERS = (
    "limit on resource njobs",
    "max_queued",
    "maximum number of jobs",
    "qosmaxjobsperuserlimit",
    "qosmaxsubmitjobperuserlimit",
    "assocmaxjobslimit",
    "assocmaxsubmitjoblimit",
)


def is_quota_refusal(reason: str) -> bool:
    """Whether a scheduler's refusal `reason` is about how many jobs are queued, not this one."""
    low = reason.lower()
    return any(marker in low for marker in _QUOTA_MARKERS)


def exit_reason(exit_code: int | None) -> str | None:
    """A human reason for an externally-imposed exit code, or None for a plain non-zero exit."""
    return _SIGNAL_EXITS.get(exit_code) if exit_code is not None else None


def failure_reason(log: str, exit_code: int | None = None) -> str:
    """One-line best-effort cause of a failed job, from its captured log and exit code."""
    for pattern in _FAILURE_MARKERS:
        if matches := pattern.findall(log):
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
    if verdict == vocabulary.CANCELLED:
        return "cancelled (stopped on purpose, not by the job or the queue)"
    if verdict == vocabulary.VANISHED:
        return "vanished (the scheduler no longer remembers the job)"
    if known := exit_reason(exit_code):
        return known
    if exit_code is not None:
        return f"exited {exit_code}"
    return "failed"


def standing(state: JobState, *, submitted_at: str = "", host: str = "") -> str:
    """Where a job that has printed nothing yet stands, in one line.

    An empty log means either not started or started and silent, so this gives the verdict, the
    scheduler's state word, the wait so far and whatever the backend says about when it will run
    (PBS's estimated start, else the resource its queue is short of).

    state: as the backend reports it now, or as the run registry last recorded it.
    submitted_at: the dispatch time, from the durable record rather than the host.
    """
    where = f" on {host}" if host else ""
    parts = [f"{state.handle} is {state.verdict}{where}"]
    if state.state:
        parts.append(f"scheduler state {state.state}")
    if submitted_at:
        waited = since(submitted_at)
        parts.append(f"submitted {submitted_at}" + (f" ({waited} ago)" if waited else ""))
    if state.estimated_start:
        parts.append(f"estimated start {state.estimated_start}")
    if state.note:
        parts.append(state.note)
    return "; ".join(parts)


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
