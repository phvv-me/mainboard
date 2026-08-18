"""The default ssh backend: jobs go to `pueue` (queue + exit codes + captured logs).

`submit` enqueues the rendered job script as one shell command with the host's workspace root as
the working directory; `state` resolves a handle against a single `pueue status` snapshot.
"""

import json
import shlex
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from patos import Model
from plumbum.commands.processes import ProcessExecutionError

from ..transport import DaemonDown, is_daemon_failure
from .base import JobState, Resources, poll_until_done

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
    exit_code: process code when `Failed`. start: ISO start time.
    """

    id: int
    label: str | None = None
    state: PueueState | str
    result: str | None = None
    exit_code: int | None = None
    start: str | None = None

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


def start(machine: Machine) -> str:
    """Start the pueue daemon detached (`pueued -d`), reviving a host whose queue died."""
    return str(machine["sh"][["-c", "pueued -d >/dev/null 2>&1"]]())


def shutdown(machine: Machine) -> str:
    """Stop the pueue daemon, or do nothing when it is already down."""
    try:
        return str(binary(machine)[["shutdown"]]())
    except ProcessExecutionError as error:
        if is_daemon_failure(error.stderr or ""):
            return ""
        raise


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
    as unreachable and can be revived.
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
                start=fields.get("start"),
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


def resume(machine: Machine, *, group: str = "default") -> str:
    """Set `group` back to running so its tasks dispatch again."""
    return str(binary(machine)[["start", "--group", group]]())


def pueue_verdict(task: PueueTask | None) -> str:
    """A one-word verdict for a pueue task (None means it's gone from the queue)."""
    if task is None:
        return "vanished"
    if task.state in _LIVE:
        return "running"
    return "ok" if task.succeeded else "failed"


def is_pueue_inherited(task: PueueTask, boundary: datetime) -> bool:
    """Whether a freshly (re)started daemon inherited `task` rather than launching it itself.

    True for an in-flight task (Running / Queued / Paused) that the current daemon did not
    start, meaning its run began before `boundary` or never began at all. When `pueued` restarts
    after a crash it resets every interrupted Running task to Queued (clearing its start) and
    pauses the group, so these inherited tasks are the zombies whose real process died with the
    old daemon. A task the revived daemon genuinely relaunched carries a fresh start at or after
    `boundary` and is spared.

    boundary: the instant the daemon was revived at.
    """
    if task.state not in _LIVE:
        return False
    if task.start is None:
        return True
    started = datetime.fromisoformat(task.start)
    aware = started if started.tzinfo is not None else started.replace(tzinfo=UTC)
    return aware < boundary


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

    def jobs(self, remote: Machine, root: str) -> list[JobState]:
        del root
        return [
            self._job_state(task) for task in status(remote) if task.state is not PueueState.DONE
        ]

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        del root
        return log(handle, machine=remote)

    def queues(self, remote: Machine, root: str) -> list[str]:
        del remote, root
        return []

    def revive(self, remote: Machine, root: str) -> list[str]:
        """Restart the daemon, retire the zombie tasks it inherits, then resume the queue.

        The one backend with a user-managed daemon: a dead `pueued` is restarted (idempotent, so
        a host whose daemon is already up is left as is). pueue's own crash recovery then resets
        every task that was running when the daemon died to Queued and pauses the group, so
        those tasks read as still in flight while their real process is gone. Any in-flight task
        that predates this revive is therefore a zombie and never a job the just-restarted
        daemon launched, so it is killed if needed, removed, and the group is resumed.
        """
        del root
        before = datetime.now(UTC)
        shutdown(remote)
        start(remote)
        handles = self.__clear_zombies(remote, before)
        resume(remote)
        return handles

    @staticmethod
    def __clear_zombies(remote: Machine, before: datetime) -> list[str]:
        """Kill (if live) and remove every task the just-restarted daemon inherited."""
        zombies = [task for task in status(remote) if is_pueue_inherited(task, before)]
        live = [str(task.id) for task in zombies if task.state in _KILLABLE]
        if live:
            kill(live, machine=remote)
        handles = [str(task.id) for task in zombies]
        if handles:
            remove(handles, machine=remote)
        return handles

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        del root
        task = next((task for task in status(remote) if str(task.id) == handle), None)
        return JobState(
            handle=handle,
            label=task.label if task else None,
            state=str(task.state) if task else None,
            exit_code=task.exit_code if task else None,
            verdict=pueue_verdict(task),
        )

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        del root, handles
        return {str(task.id): self._job_state(task) for task in status(remote)}

    def stream(self, remote: Machine, root: str, *, handle: str) -> JobState:
        try:
            binary(remote)[["follow", handle]]()
        except ProcessExecutionError:
            return self._settle_broken_follow(remote, root, handle=handle)
        return self.wait(remote, root, handle=handle)

    def _settle_broken_follow(self, remote: Machine, root: str, *, handle: str) -> JobState:
        """The task's current state when `follow` dies mid-stream, re-raising a live task's loss.

        A `follow` that dies because the task already vanished is not itself a failure worth
        propagating, so its resolved state is returned instead of the `follow` error.
        """
        final = self.state(remote, root, handle=handle)
        if final.verdict == "vanished":
            return final
        raise

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

    def wait(self, remote: Machine, root: str, *, handle: str) -> JobState:
        return poll_until_done(lambda: self.state(remote, root, handle=handle))

    @staticmethod
    def _job_state(task: PueueTask) -> JobState:
        return JobState(
            handle=str(task.id),
            label=task.label,
            state=str(task.state),
            exit_code=task.exit_code,
            verdict=pueue_verdict(task),
        )
