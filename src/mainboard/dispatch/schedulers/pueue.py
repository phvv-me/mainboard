"""The default ssh backend: jobs go to `pueue` (queue + exit codes + captured logs).

`submit` enqueues the rendered job script as one shell command with the host's workspace root as
the working directory; `state` resolves a handle against a single `pueue status` snapshot.
"""

import json
import shlex
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from patos import Model
from plumbum.commands.processes import ProcessExecutionError

from ..transport import DaemonDown, is_daemon_failure
from ..vocabulary import JobState, Resources
from .base import workspace_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plumbum.commands.base import BaseCommand

    from ..transport import Machine


class PueueState(StrEnum):
    """pueue task lifecycle, the externally-tagged key of a task's `status`."""

    LOCKED = "Locked"
    STASHED = "Stashed"
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    DONE = "Done"


# pueue states with a live child process, the ones a zombie must be killed out of before it can
# be removed (a Queued/Stashed/Done task is removed directly).
_KILLABLE = {PueueState.RUNNING, PueueState.PAUSED}
_REMOVABLE = {PueueState.LOCKED, PueueState.STASHED, PueueState.QUEUED}
# pueue states that still mean "in flight".
_LIVE = {PueueState.RUNNING, PueueState.QUEUED, PueueState.PAUSED}


class PueueTask(Model):
    """A task from `pueue status --json`.

    id: pueue's task id (the dispatch handle for ssh targets). label: the submit label. state:
    lifecycle state. result: the task result once `Done` (`Success`, `Failed`, `Killed`, ...).
    exit_code: process code when `Failed`.
    """

    id: int
    label: str | None = None
    state: PueueState | str
    result: str | None = None
    exit_code: int | None = None

    @property
    def succeeded(self) -> bool:
        """True once the task finished with a `Success` result."""
        return self.state == PueueState.DONE and self.result == "Success"


def binary(machine: Machine) -> BaseCommand:
    """The `pueue` command on `machine`, resolved off PATH.

    pueue is expected as a host-level tool (installed once per host, like the dispatch tool
    itself), not baked into a per-workspace env, so no env-specific path is searched here.
    """
    return machine["pueue"]


def _raise_as_daemon_down_when_dead(error: ProcessExecutionError) -> None:
    """Re-raise `error` as `DaemonDown` when its stderr names a dead daemon, else verbatim."""
    if is_daemon_failure(error.stderr or ""):
        raise DaemonDown("daemon down") from error
    raise error


def add(command: str, *, machine: Machine, root: str, label: str) -> str:
    """Enqueue `command` on `machine` and return its task id.

    pueue runs the trailing string in a subshell, so the whole command is passed as one string
    to keep its quoting intact.
    """
    args = ["add", "--print-task-id", "--label", label, "--working-directory", root]
    return str(binary(machine)[[*args, "--", command]]().strip())


def status(machine: Machine) -> list[PueueTask]:
    """The queue's tasks, parsed from `pueue status --json`.

    A task's `status` is externally tagged, `{"Running": {...}}` or `{"Done": {"start", "end",
    "result", ...}}`, and `result` is a string (`"Success"`/`"Killed"`/...) or `{"Failed":
    <exit-code>}`.

    A dead `pueued` refuses its control socket, so the client exits non-zero; that case is
    re-raised as `DaemonDown` rather than crashing every caller, so a host whose queue died reads
    as unreachable and a sweep reports it once instead of failing the whole pass.
    """
    try:
        output = binary(machine)[["status", "--json"]]()
    except ProcessExecutionError as error:
        _raise_as_daemon_down_when_dead(error)
    tasks: list[PueueTask] = []
    for task in json.loads(output).get("tasks", {}).values():
        state, fields = next(iter(task["status"].items()))
        result = fields.get("result")
        tasks.append(
            PueueTask(
                id=task["id"],
                label=task.get("label"),
                state=PueueState(state),
                result=next(iter(result)) if isinstance(result, dict) else result,
                exit_code=result.get("Failed")
                if isinstance(result, dict)
                else (0 if result == "Success" else None),
            ),
        )
    return tasks


def log(task_id: int | str, *, machine: Machine) -> str:
    """The full captured log of `task_id`."""
    return str(binary(machine)[["log", "--full", str(task_id)]]())


def kill(task_ids: Sequence[int | str], *, machine: Machine) -> str:
    """Kill one or many tasks."""
    return str(binary(machine)[["kill", *(str(task_id) for task_id in task_ids)]]())


def remove(task_ids: Sequence[int | str], *, machine: Machine) -> str:
    """Drop one or many tasks from the list entirely, so each reads as `vanished` afterwards."""
    return str(binary(machine)[["remove", *(str(task_id) for task_id in task_ids)]]())


def pueue_verdict(task: PueueTask | None) -> str:
    """A one-word verdict for a pueue task (None means it's gone from the queue)."""
    if task is None:
        return "vanished"
    if task.state in _LIVE:
        return "running"
    return "ok" if task.succeeded else "failed"


class Pueue:
    """Dispatch jobs to a host's `pueue` daemon (the ssh default)."""

    name = "ssh"

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        del root
        task = next((task for task in status(remote) if str(task.id) == handle), None)
        if task is None or task.state == PueueState.DONE:
            return
        if task.state in _KILLABLE:
            kill([handle], machine=remote)
        elif task.state in _REMOVABLE:
            remove([handle], machine=remote)

    def interactive(self, *, env: str, command: Sequence[str], resources: Resources) -> str:
        return workspace_session(env=env, command=command, resources=resources)

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        del root
        return log(handle, machine=remote)

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        return self.states(remote, root, [handle])[handle]

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """Every task the daemon still remembers, plus a vanished answer for the rest.

        One `pueue status` covers the whole daemon, so a caller holding a thousand handles on
        this host pays one round trip for all of them rather than one each. A requested handle
        the daemon has forgotten is answered here rather than left absent, since the listing is
        already everything pueue knows and asking again would only fetch the same nothing.
        """
        del root
        live = {str(task.id): self._job_state(task) for task in status(remote)}
        return live | {
            handle: JobState(handle=handle, verdict=pueue_verdict(None))
            for handle in handles
            if handle not in live
        }

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        args: Sequence[str],
        resources: Resources,
    ) -> str:
        del resources  # pueue has no queue-side resource request; the script is self-contained.
        arg_str = " ".join(shlex.quote(arg) for arg in args)
        command = f"bash {shlex.quote(script)} {arg_str}".rstrip()
        return add(command, machine=remote, root=root, label=Path(script).stem)

    @staticmethod
    def _job_state(task: PueueTask) -> JobState:
        return JobState(
            handle=str(task.id),
            label=task.label,
            state=str(task.state),
            exit_code=task.exit_code,
            verdict=pueue_verdict(task),
        )
