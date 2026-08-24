# The `Scheduler` contract every job backend implements, plus the log-reading and failure-triage
# vocabulary every backend shares. New backends are new classes, never new `if kind == ...`
# branches. The resource request, the job state and the verdict lifecycle live one level up in
# `dispatch.vocabulary`, since a provider backend speaks them without being a scheduler at all.

import re
import shlex
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...core.project import Project
from .. import vocabulary
from ..shared import state_dir
from ..transport import HostUnreachable, is_transport_failure
from ..vocabulary import JobState, Resources

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..transport import Machine


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


@runtime_checkable
class Scheduler(Protocol):
    """A pluggable job backend dispatched to generically.

    `remote` is an open plumbum `SshMachine` (or `local`); `root` is the workspace path on the
    host. Implementations are stateless value objects, so one instance per kind is enough.
    """

    name: str

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        """Cancel `handle` on the host."""

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        """What an interactive session runs once the caller's ssh has staged the workspace.

        A queued backend asks its scheduler for an interactive allocation (`qsub -I`, `srun
        --pty`), while a backend whose host is already the machine the work runs on hands the
        terminal to that host's own tool. Either way the caller owns the ssh and the staging, so
        this only ever describes the one command run inside them.

        env: the environment the session works in.
        command: a command to run instead of handing over the terminal, empty for a session.
        resources: the allocation an interactive job asks its scheduler for.
        """

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        """`handle`'s captured log so far (merged stdout+stderr)."""

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict, for reconcile."""

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """The state of `handles` (and any other live job) in one batched query, keyed by handle.

        One round-trip so a whole host's pending runs resolve at once instead of one probe per
        run. A handle the host no longer remembers is simply absent; the caller falls back to
        its cached verdict.
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

    The host's own tool owns the activation in both shapes, its interactive `shell` when the
    terminal is being handed over and `run` for a one-off command, so an interactive session and
    a dispatched job never disagree about which interpreter they got.

    env: the environment the session works in.
    command: a command to run instead of handing over the terminal, empty for a session.
    resources: ignored, since an ssh host allocates nothing and is the machine itself.
    """
    del resources
    tool = Project().name
    if command:
        return shlex.join([tool, "run", "--env", env, "--", *command])
    return shlex.join([tool, "shell", env])


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
