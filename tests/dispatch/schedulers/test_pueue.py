import json
from typing import TYPE_CHECKING

import pytest
from plumbum.commands.processes import ProcessExecutionError

from mainboard.dispatch import DaemonDown
from mainboard.dispatch.schedulers import Pueue
from mainboard.dispatch.schedulers.pueue import (
    PueueState,
    PueueTask,
    add,
    binary,
    kill,
    log,
    pueue_verdict,
    remove,
    status,
)
from mainboard.dispatch.vocabulary import Resources

from ..conftest import machine_with

if TYPE_CHECKING:
    from collections.abc import Mapping

type Json = str | int | float | bool | None | dict[str, "Json"] | list["Json"]

_DEAD_DAEMON = ProcessExecutionError(["pueue"], 1, "", "Error connecting to the daemon")
_REAL_FAILURE = ProcessExecutionError(["pueue"], 1, "", "permission denied")


def status_json(*tasks: Mapping[str, Json]) -> str:
    """The `pueue status --json` payload wrapping each task under its own numeric key."""
    return json.dumps({"tasks": {str(index): task for index, task in enumerate(tasks)}})


def test_each_client_helper_builds_its_own_pueue_argv() -> None:
    machine = machine_with(" 7 \n")
    binary(machine)
    assert machine.calls == []
    assert add("run.sh", machine=machine, root="/repo", label="job") == "7"
    assert machine.calls[-1] == [
        "pueue",
        "add",
        "--print-task-id",
        "--label",
        "job",
        "--working-directory",
        "/repo",
        "--",
        "run.sh",
    ]
    log(3, machine=machine)
    assert machine.calls[-1] == ["pueue", "log", "--full", "3"]
    kill([1, "2"], machine=machine)
    assert machine.calls[-1] == ["pueue", "kill", "1", "2"]
    remove([1, "2"], machine=machine)
    assert machine.calls[-1] == ["pueue", "remove", "1", "2"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"id": 0, "label": "job", "status": {"Running": {"start": "t0"}}},
            PueueTask(id=0, label="job", state=PueueState.RUNNING),
        ),
        (
            {"id": 1, "status": {"Done": {"start": "t0", "result": "Success"}}},
            PueueTask(id=1, state=PueueState.DONE, result="Success", exit_code=0),
        ),
        (
            {"id": 2, "status": {"Done": {"start": "t0", "result": {"Failed": 7}}}},
            PueueTask(id=2, state=PueueState.DONE, result="Failed", exit_code=7),
        ),
        (
            {"id": 3, "status": {"Done": {"start": "t0", "result": "Killed"}}},
            PueueTask(id=3, state=PueueState.DONE, result="Killed"),
        ),
    ],
)
def test_status_reads_each_externally_tagged_task_result(
    payload: Mapping[str, Json], expected: PueueTask
) -> None:
    """`result` is a bare string or a `{"Failed": <code>}` object, and only one of them codes."""
    [task] = status(machine_with(status_json(payload)))
    assert task == expected
    assert task.succeeded == (expected.result == "Success")


def test_status_raises_daemon_down_only_for_a_refused_control_socket() -> None:
    with pytest.raises(DaemonDown, match="daemon down"):
        status(machine_with(faults=[("status", _DEAD_DAEMON)]))
    with pytest.raises(ProcessExecutionError, match="permission denied"):
        status(machine_with(faults=[("status", _REAL_FAILURE)]))


@pytest.mark.parametrize(
    ("task", "verdict"),
    [
        (None, "vanished"),
        (PueueTask(id=1, state=PueueState.RUNNING), "running"),
        (PueueTask(id=1, state=PueueState.QUEUED), "running"),
        (PueueTask(id=1, state=PueueState.PAUSED), "running"),
        (PueueTask(id=1, state=PueueState.DONE, result="Success"), "ok"),
        (PueueTask(id=1, state=PueueState.DONE, result="Failed", exit_code=1), "failed"),
    ],
)
def test_pueue_verdict_reads_one_word_out_of_a_task(task: PueueTask | None, verdict: str) -> None:
    assert pueue_verdict(task) == verdict


def test_the_backend_submits_lists_and_reads_a_task_back() -> None:
    backend = Pueue()
    submitting = machine_with(" 5 \n")
    handle = backend.submit(
        submitting, "/repo", script="job.sh", args=("--x", "1"), resources=Resources()
    )
    assert handle == "5"
    assert submitting.calls[-1][:4] == ["pueue", "add", "--print-task-id", "--label"]
    assert submitting.calls[-1][-1] == "bash job.sh --x 1"
    queue = machine_with(
        status_json(
            {"id": 0, "label": "job", "status": {"Running": {"start": "t0"}}},
            {"id": 1, "status": {"Done": {"start": "t0", "result": "Success"}}},
        )
    )
    # One `pueue status` answers for the whole daemon and for every handle asked about, the
    # forgotten ones included, so a host holding a thousand handles costs one round trip.
    batched = backend.states(queue, "/repo", ["0", "999"])
    assert sorted(batched) == ["0", "1", "999"]
    assert batched["999"].verdict == "vanished"
    assert backend.state(queue, "/repo", handle="0").verdict == "running"
    assert backend.state(queue, "/repo", handle="999").verdict == "vanished"
    assert backend.logs(machine_with("captured\n"), "/repo", handle="3") == "captured\n"


def test_an_ssh_host_hands_its_terminal_to_its_own_tool() -> None:
    """A pueue host is already the machine the work runs on, so nothing is allocated for it."""
    assert Pueue().interactive(env="serving", command=(), resources=Resources()) == (
        "mainboard shell serving"
    )
    assert Pueue().interactive(env="default", command=("pwd",), resources=Resources()) == (
        "mainboard run --env default -- pwd"
    )


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ({"id": 0, "status": {"Running": {"start": "t0"}}}, ["pueue", "kill", "0"]),
        ({"id": 0, "status": {"Paused": {"start": "t0"}}}, ["pueue", "kill", "0"]),
        ({"id": 0, "status": {"Queued": {}}}, ["pueue", "remove", "0"]),
        ({"id": 0, "status": {"Stashed": {}}}, ["pueue", "remove", "0"]),
        ({"id": 0, "status": {"Done": {"result": "Success"}}}, ["pueue", "status", "--json"]),
    ],
)
def test_cancel_kills_a_live_task_removes_a_waiting_one_and_leaves_a_finished_one(
    task: Mapping[str, Json], expected: list[str]
) -> None:
    queue = machine_with(status_json(task))
    Pueue().cancel(queue, "/repo", handle="0")
    assert queue.calls[-1] == expected


def test_cancel_leaves_an_unknown_handle_and_a_state_this_codebase_never_heard_of_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future pueue state is neither killable nor removable, so it is touched by nothing."""
    empty = machine_with(status_json())
    Pueue().cancel(empty, "/repo", handle="999")
    assert empty.calls[-1] == ["pueue", "status", "--json"]
    monkeypatch.setattr(
        "mainboard.dispatch.schedulers.pueue.status",
        lambda remote: [PueueTask(id=0, state="Mystery")],
    )
    untouched = machine_with(status_json())
    Pueue().cancel(untouched, "/repo", handle="0")
    assert untouched.calls == []
