"""The default ssh backend: jobs go to `pueue` (queue + exit codes + captured logs).

`submit` enqueues the job script as one shell command whose working directory is the tree the
dispatch pinned; `state` resolves a handle against a single `pueue status` snapshot.
"""

import json
import shlex
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from patos import Model
from plumbum.commands.processes import ProcessExecutionError

from ...core.errors import MissionError
from ..jobs import vocabulary
from ..jobs.vocabulary import JobState, Resources
from ..transport.core import DaemonDown, is_daemon_failure
from .base import Scheduler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plumbum.commands.base import BaseCommand

    from ..transport.core import Machine


class PueueState(StrEnum):
    """pueue task lifecycle, the externally-tagged key of a task's `status`."""

    LOCKED = "Locked"
    STASHED = "Stashed"
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    DONE = "Done"


# Every state but Done is in flight. A started task has a child process, which a cancel kills; a
# waiting one has none, which a cancel removes. Locked is a task being edited, Stashed one held
# back by hand: neither has started, and neither is over.
_STARTED = {PueueState.RUNNING, PueueState.PAUSED}
_WAITING = {PueueState.LOCKED, PueueState.STASHED, PueueState.QUEUED}
# The stage a listing shows; a paused or finished task keeps pueue's own word.
_STAGES: dict[str, str] = {
    **dict.fromkeys(_WAITING, vocabulary.Word.QUEUED),
    PueueState.RUNNING: vocabulary.Word.RUNNING,
}


class PueueTask(Model):
    """A task from `pueue status --json`.

    id: pueue's task id, the dispatch handle for ssh targets.
    result: once `Done`, `Success`, `Failed`, `Killed`, ...
    exit_code: 0 on `Success`, the process code when `Failed`.
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


def _binary(machine: Machine) -> BaseCommand:
    """The `pueue` command on `machine`, off PATH: a host-level tool, never in a workspace env."""
    return machine["pueue"]


def _status(machine: Machine) -> list[PueueTask]:
    """The queue's tasks, parsed from `pueue status --json`.

    A task's `status` is externally tagged, `{"Running": {...}}` or `{"Done": {"result", ...}}`,
    and `result` is a string (`"Success"`/`"Killed"`/...) or `{"Failed": <exit-code>}`; a variant
    without fields is the bare string. A state this table does not name is kept as its word. A
    dead `pueued` refuses its control socket, raised as `DaemonDown` so the host reads as
    unreachable and a sweep reports it once instead of failing the whole pass.
    """
    try:
        output = _binary(machine)[["status", "--json"]]()
    except ProcessExecutionError as error:
        if is_daemon_failure(error.stderr or ""):
            raise DaemonDown("daemon down") from error
        raise
    tasks: list[PueueTask] = []
    for task in json.loads(output).get("tasks", {}).values():
        tagged = task["status"]
        state, fields = (tagged, {}) if isinstance(tagged, str) else next(iter(tagged.items()))
        result = fields.get("result")
        tasks.append(
            PueueTask(
                id=task["id"],
                label=task.get("label"),
                state=_known(state),
                result=next(iter(result)) if isinstance(result, dict) else result,
                exit_code=result.get("Failed")
                if isinstance(result, dict)
                else (0 if result == "Success" else None),
            ),
        )
    return tasks


def _known(word: str) -> PueueState | str:
    """The state `word` names, else `word` itself for a variant a newer pueue added."""
    try:
        return PueueState(word)
    except ValueError:
        return word


def _pueue_verdict(task: PueueTask | None) -> str:
    """A one-word verdict for a pueue task (None means it's gone from the queue).

    Only a Done task has an outcome; waiting, started and unrecognized states are `running`.
    """
    if task is None:
        return "vanished"
    if task.state != PueueState.DONE:
        return "running"
    return "ok" if task.succeeded else "failed"


class Pueue(Scheduler):
    """Dispatch jobs to a host's `pueue` daemon (the ssh default)."""

    name = "ssh"

    @staticmethod
    def _add(command: str, *, machine: Machine, root: str, label: str) -> str:
        """Enqueue `command` as one string, which pueue runs in a subshell; return its task id."""
        args = ["add", "--print-task-id", "--label", label, "--working-directory", root]
        return _binary(machine)[[*args, "--", command]]().strip()

    @staticmethod
    def _job_state(task: PueueTask) -> JobState:
        return JobState(
            handle=str(task.id),
            label=task.label,
            state=str(task.state),
            exit_code=task.exit_code,
            verdict=_pueue_verdict(task),
            stage=_STAGES.get(task.state, ""),
        )

    def cancel(self, remote: Machine, root: str, *, handle: str) -> None:
        """Kill a started task, remove a waiting one, and leave one that is gone or done.

        A state this tool cannot place raises: stopping it by a guess could report a cancel that
        did not happen.
        """
        state = next((task.state for task in _status(remote) if str(task.id) == handle), None)
        if state in _STARTED:
            _binary(remote)[["kill", str(handle)]]()
        elif state in _WAITING:
            _binary(remote)[["remove", str(handle)]]()
        elif state not in {None, PueueState.DONE}:
            raise MissionError(f"pueue task {handle} is {state!r}, a state not known to stop")

    def logs(self, remote: Machine, root: str, *, handle: str) -> str:
        return _binary(remote)[["log", "--full", str(handle)]]()

    def state(self, remote: Machine, root: str, *, handle: str) -> JobState:
        return self.states(remote, root, [handle])[handle]

    def states(self, remote: Machine, root: str, handles: Sequence[str]) -> dict[str, JobState]:
        """Every task the daemon remembers, plus `vanished` for the rest.

        A forgotten handle is answered rather than left absent, since one `pueue status` is
        already everything pueue knows and a per-handle re-probe would fetch the same nothing.
        """
        live = {str(task.id): self._job_state(task) for task in _status(remote)}
        return live | {
            handle: JobState(handle=handle, verdict=_pueue_verdict(None))
            for handle in handles
            if handle not in live
        }

    def submit(
        self,
        remote: Machine,
        root: str,
        *,
        script: str,
        resources: Resources,
    ) -> str:
        del resources  # pueue has no queue-side resource request; the script is self-contained.
        command = shlex.join(["sh", script])
        return Pueue._add(command, machine=remote, root=root, label=Path(script).stem)
