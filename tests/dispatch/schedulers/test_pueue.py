import json
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING

import pytest
from plumbum.commands.processes import ProcessExecutionError

from mainboard.dispatch import DaemonDown
from mainboard.dispatch.schedulers import Pueue, Resources
from mainboard.dispatch.schedulers.pueue import (
    PueueState,
    PueueTask,
    add,
    binary,
    is_pueue_inherited,
    kill,
    log,
    pueue_verdict,
    remove,
    resume,
    shutdown,
    start,
    status,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class FakeCommand:
    """A `machine["pueue"][[...]]` double that records argv and replays (or raises)."""

    def __init__(self, output: str = "", *, error: ProcessExecutionError | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[list[str]] = []
        self.bound: list[str] = []

    def __call__(self, *_, **__) -> str:
        self.calls.append(self.bound)
        if self.error is not None:
            raise self.error
        return self.output

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> FakeCommand:
        extra = list(args) if isinstance(args, list | tuple) else [args]
        child = type(self)(self.output, error=self.error)
        child.calls = self.calls
        child.bound = [*self.bound, *(str(a) for a in extra)]
        return child


class FakeMachine:
    def __init__(self, output: str = "", *, error: ProcessExecutionError | None = None) -> None:
        self.command = FakeCommand(output, error=error)

    def __getitem__(self, name: str) -> FakeCommand:
        return self.command


type Json = str | int | float | bool | None | dict[str, "Json"] | list["Json"]


def status_json(*tasks: Mapping[str, Json]) -> str:
    return json.dumps({"tasks": {str(i): task for i, task in enumerate(tasks)}})


# --- binary / start / shutdown ---


def test_binary_resolves_off_path() -> None:
    machine = FakeMachine()
    binary(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls == []  # binary() only indexes, never calls


def test_start_runs_pueued_detached() -> None:
    machine = FakeMachine("")
    start(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls == [["-c", "pueued -d >/dev/null 2>&1"]]


def test_shutdown_returns_output_on_success() -> None:
    machine = FakeMachine("stopped\n")
    assert shutdown(machine) == "stopped\n"  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


def test_shutdown_swallows_a_dead_daemon() -> None:
    error = ProcessExecutionError(["pueue"], 1, "", "Error connecting to the daemon")
    machine = FakeMachine(error=error)
    assert shutdown(machine) == ""  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


def test_shutdown_reraises_a_genuine_failure() -> None:
    error = ProcessExecutionError(["pueue"], 1, "", "permission denied")
    machine = FakeMachine(error=error)
    with pytest.raises(ProcessExecutionError):
        shutdown(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


# --- add / log / kill / remove / resume ---


def test_add_prints_the_task_id() -> None:
    machine = FakeMachine(" 7 \n")
    assert add("run.sh", machine=machine, root="/repo", label="job") == "7"  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


def test_log_requests_the_full_log() -> None:
    machine = FakeMachine("output\n")
    assert log(3, machine=machine) == "output\n"  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1] == ["log", "--full", "3"]


def test_kill_and_remove_join_task_ids() -> None:
    machine = FakeMachine("")
    kill([1, "2"], machine=machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1] == ["kill", "1", "2"]
    remove([1, "2"], machine=machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1] == ["remove", "1", "2"]


def test_resume_targets_the_given_group() -> None:
    machine = FakeMachine("")
    resume(machine, group="default")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1] == ["start", "--group", "default"]


# --- status() ---


def test_status_parses_a_running_task() -> None:
    payload = status_json({"id": 0, "label": "job", "status": {"Running": {"start": "t0"}}})
    machine = FakeMachine(payload)
    [task] = status(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert task == PueueTask(
        id=0, label="job", state=PueueState.RUNNING, result=None, exit_code=None, start="t0"
    )


def test_status_parses_a_successful_done_task() -> None:
    payload = status_json({"id": 1, "status": {"Done": {"start": "t0", "result": "Success"}}})
    machine = FakeMachine(payload)
    [task] = status(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert task.succeeded
    assert task.exit_code == 0


def test_status_parses_a_failed_done_task_with_exit_code() -> None:
    payload = status_json({"id": 2, "status": {"Done": {"start": "t0", "result": {"Failed": 7}}}})
    machine = FakeMachine(payload)
    [task] = status(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert task.result == "Failed"
    assert task.exit_code == 7
    assert not task.succeeded


def test_status_parses_a_killed_done_task_with_no_exit_code() -> None:
    payload = status_json({"id": 3, "status": {"Done": {"start": "t0", "result": "Killed"}}})
    machine = FakeMachine(payload)
    [task] = status(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert task.result == "Killed"
    assert task.exit_code is None


def test_status_raises_daemon_down_on_a_dead_daemon() -> None:
    error = ProcessExecutionError(["pueue"], 1, "", "Error connecting to the daemon")
    machine = FakeMachine(error=error)
    with pytest.raises(DaemonDown):
        status(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


def test_status_reraises_a_genuine_failure() -> None:
    error = ProcessExecutionError(["pueue"], 1, "", "unexpected token")
    machine = FakeMachine(error=error)
    with pytest.raises(ProcessExecutionError):
        status(machine)  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


# --- pueue_verdict ---


def test_pueue_verdict_none_task_is_vanished() -> None:
    assert pueue_verdict(None) == "vanished"


@pytest.mark.parametrize("state", [PueueState.RUNNING, PueueState.QUEUED, PueueState.PAUSED])
def test_pueue_verdict_live_states_are_running(state: PueueState) -> None:
    task = PueueTask(id=1, state=state)
    assert pueue_verdict(task) == "running"


def test_pueue_verdict_done_success_is_ok() -> None:
    task = PueueTask(id=1, state=PueueState.DONE, result="Success")
    assert pueue_verdict(task) == "ok"


def test_pueue_verdict_done_failure_is_failed() -> None:
    task = PueueTask(id=1, state=PueueState.DONE, result="Failed", exit_code=1)
    assert pueue_verdict(task) == "failed"


# --- is_pueue_inherited ---

_BOUNDARY = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)


def test_is_pueue_inherited_false_for_a_terminal_task() -> None:
    task = PueueTask(id=1, state=PueueState.DONE)
    assert is_pueue_inherited(task, _BOUNDARY) is False


def test_is_pueue_inherited_true_when_start_is_missing() -> None:
    task = PueueTask(id=1, state=PueueState.RUNNING, start=None)
    assert is_pueue_inherited(task, _BOUNDARY) is True


def test_is_pueue_inherited_true_for_a_run_that_predates_the_boundary() -> None:
    task = PueueTask(id=1, state=PueueState.RUNNING, start="2026-06-28T11:00:00+00:00")
    assert is_pueue_inherited(task, _BOUNDARY) is True


def test_is_pueue_inherited_false_for_a_run_started_after_the_boundary() -> None:
    task = PueueTask(id=1, state=PueueState.RUNNING, start="2026-06-28T13:00:00+00:00")
    assert is_pueue_inherited(task, _BOUNDARY) is False


def test_is_pueue_inherited_treats_a_naive_start_as_utc() -> None:
    task = PueueTask(id=1, state=PueueState.RUNNING, start="2026-06-28T11:00:00")
    assert is_pueue_inherited(task, _BOUNDARY) is True


# --- Pueue backend ---


def test_submit_enqueues_the_activated_script() -> None:
    machine = FakeMachine(" 5 \n")
    handle = Pueue().submit(  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
        machine, "/repo", script="job.sh", args=("--x", "1"), resources=Resources()
    )
    assert handle == "5"
    assert machine.command.calls[-1][:2] == ["add", "--print-task-id"]


def test_jobs_excludes_done_tasks() -> None:
    payload = status_json(
        {"id": 0, "status": {"Running": {"start": "t0"}}},
        {"id": 1, "status": {"Done": {"start": "t0", "result": "Success"}}},
    )
    machine = FakeMachine(payload)
    [state] = Pueue().jobs(machine, "/repo")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert state.handle == "0"


def test_logs_returns_the_captured_output() -> None:
    machine = FakeMachine("captured\n")
    assert Pueue().logs(machine, "/repo", handle="3") == "captured\n"  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


def test_state_finds_the_matching_task() -> None:
    payload = status_json({"id": 9, "status": {"Running": {"start": "t0"}}})
    machine = FakeMachine(payload)
    state = Pueue().state(machine, "/repo", handle="9")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert state.verdict == "running"


def test_state_of_an_unknown_handle_is_vanished() -> None:
    machine = FakeMachine(status_json())
    state = Pueue().state(machine, "/repo", handle="999")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert state.verdict == "vanished"


def test_states_keys_every_task_by_id() -> None:
    payload = status_json({"id": 0, "status": {"Running": {"start": "t0"}}})
    machine = FakeMachine(payload)
    states = Pueue().states(machine, "/repo", ["0"])  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert "0" in states


def test_wait_polls_state_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = status_json({"id": 0, "status": {"Done": {"start": "t0", "result": "Success"}}})
    machine = FakeMachine(payload)
    monkeypatch.setattr(
        "mainboard.dispatch.schedulers.pueue.poll_until_done", lambda probe: probe()
    )
    assert Pueue().wait(machine, "/repo", handle="0").verdict == "ok"  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


def test_stream_follows_then_reports_the_final_wait_state(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = status_json({"id": 0, "status": {"Done": {"start": "t0", "result": "Success"}}})
    machine = FakeMachine(payload)
    result = Pueue().stream(machine, "/repo", handle="0")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert result.verdict == "ok"


def test_stream_when_follow_fails_but_the_task_vanished_returns_that_state() -> None:
    error = ProcessExecutionError(["pueue"], 1, "", "no such task")

    class FollowThenStatus(FakeCommand):
        def __call__(self, *args, **kwargs) -> str:
            if self.bound and self.bound[0] == "follow":
                raise error
            return super().__call__(*args, **kwargs)

    machine = FakeMachine(status_json())
    machine.command = FollowThenStatus(status_json())
    result = Pueue().stream(machine, "/repo", handle="0")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert result.verdict == "vanished"


def test_stream_reraises_a_follow_failure_when_the_task_is_not_vanished() -> None:
    error = ProcessExecutionError(["pueue"], 1, "", "boom")

    class FollowThenRunning(FakeCommand):
        def __call__(self, *args, **kwargs) -> str:
            if self.bound and self.bound[0] == "follow":
                raise error
            return super().__call__(*args, **kwargs)

    machine = FakeMachine()
    machine.command = FollowThenRunning(
        status_json({"id": 0, "status": {"Running": {"start": "t0"}}})
    )
    with pytest.raises(ProcessExecutionError):
        Pueue().stream(machine, "/repo", handle="0")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16


def test_cancel_kills_a_running_task() -> None:
    payload = status_json({"id": 0, "status": {"Running": {"start": "t0"}}})
    machine = FakeMachine(payload)
    Pueue().cancel(machine, "/repo", handle="0")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1] == ["kill", "0"]


def test_cancel_removes_a_queued_task() -> None:
    payload = status_json({"id": 0, "status": {"Queued": {}}})
    machine = FakeMachine(payload)
    Pueue().cancel(machine, "/repo", handle="0")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1] == ["remove", "0"]


def test_cancel_is_a_no_op_for_a_done_task() -> None:
    payload = status_json({"id": 0, "status": {"Done": {"start": "t0", "result": "Success"}}})
    machine = FakeMachine(payload)
    Pueue().cancel(machine, "/repo", handle="0")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1][0] == "status"


def test_cancel_is_a_no_op_for_an_unknown_handle() -> None:
    machine = FakeMachine(status_json())
    Pueue().cancel(machine, "/repo", handle="999")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls[-1][0] == "status"


def test_cancel_is_a_no_op_for_a_state_that_is_neither_killable_nor_removable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future pueue state this codebase does not yet recognize is left untouched."""
    task = PueueTask(id=0, state="Mystery")
    monkeypatch.setattr("mainboard.dispatch.schedulers.pueue.status", lambda remote: [task])
    machine = FakeMachine(status_json())
    Pueue().cancel(machine, "/repo", handle="0")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert machine.command.calls == []


class FrozenDatetime(datetime):
    """A `datetime` subclass whose `now()` is pinned, keeping every other method real."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        del tz
        return _BOUNDARY


def test_revive_restarts_the_daemon_and_clears_zombies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mainboard.dispatch.schedulers.pueue.datetime", FrozenDatetime)
    payload = status_json({"id": 0, "status": {"Running": {"start": "2020-01-01T00:00:00+00:00"}}})
    machine = FakeMachine(payload)
    handles = Pueue().revive(machine, "/repo")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert handles == ["0"]
    kinds = [call[0] for call in machine.command.calls]
    assert kinds == ["shutdown", "-c", "status", "kill", "remove", "start"]


def test_revive_resumes_the_queue_with_no_zombies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mainboard.dispatch.schedulers.pueue.datetime", FrozenDatetime)
    machine = FakeMachine(status_json())
    handles = Pueue().revive(machine, "/repo")  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
    assert handles == []
    kinds = [call[0] for call in machine.command.calls]
    assert kinds == ["shutdown", "-c", "status", "start"]


def test_queues_is_always_empty() -> None:
    assert Pueue().queues(FakeMachine(), "/repo") == []  # type: ignore[arg-type]  reason=test double stands in for the Machine union since=2026-08-16
