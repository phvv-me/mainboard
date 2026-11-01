"""The center's scheduler over the held line: admission, states, cancel and fence.

The center's shell is executed by a local bash (plumbum's `local` is a `Machine`) and PBS itself
is absent.
"""

import subprocess
import time
from pathlib import Path

import pytest
from plumbum import local

from mainboard import MissionError
from mainboard.dispatch import JobState, Resources, Spool
from mainboard.dispatch.jobs import vocabulary
from mainboard.dispatch.mirroring.spool import Claim, Remote, spool_path
from mainboard.dispatch.schedulers import Held, Pbs, kind_of, pick
from mainboard.dispatch.schedulers.pbs import held as held_module
from mainboard.manifest import HostProfile, QueuePolicy

from .builders import entry, halt, script, serve, until


def test_a_submission_is_queued_then_running_then_settled_from_its_exit(
    held: Held, spool: Spool, root: str, node: dict[str, str], tmp_path: Path
) -> None:
    resources = Resources(queue="held", walltime="00:05:00")
    login = local
    name = held.enqueue(
        login, root, script=script(tmp_path, "sleep 1\n"), resources=resources, label="x"
    )
    assert held.states(login, root, [name])[name].stage == vocabulary.Word.QUEUED
    thread = serve(spool, node)
    until(lambda: held.states(login, root, [name])[name].stage == vocabulary.Word.RUNNING)
    until(lambda: held.states(login, root, [name])[name].verdict == vocabulary.Word.OK)
    assert held.states(login, root, [name])[name].exit_code == 0
    halt(spool, thread)


def test_a_claim_without_an_exit_is_lost_only_when_pbs_says_its_allocation_ended(
    held: Held, spool: Spool, root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission = entry(spool, script(tmp_path, "true"), name="h01")
    spool.claim(submission, Claim(alloc="123.opbs", node="n1", at=time.time()))
    login = local
    assert held.states(login, root, ["h01"])["h01"].verdict == vocabulary.Word.RUNNING
    ended = vocabulary.Word.OK
    monkeypatch.setattr(
        Pbs,
        "states",
        lambda self, remote, path, handles: {
            "123.opbs": JobState(handle="123.opbs", verdict=ended)
        },
    )
    assert held.states(login, root, ["h01"])["h01"].verdict == vocabulary.Word.VANISHED
    spool.publish(spool.logs, "h01.exit", text="exit=0\n")
    assert held.states(login, root, ["h01"])["h01"].verdict == vocabulary.Word.OK


def test_a_submission_needs_a_line_that_outlasts_it(
    held: Held, spool: Spool, root: str, tmp_path: Path
) -> None:
    login = local
    too_long = Resources(queue="held", walltime="02:00:00")
    with pytest.raises(MissionError, match="before a 02:00:00 job could finish"):
        held.enqueue(login, root, script="s", resources=too_long, label="x")
    (spool.path / "stop").touch()
    with pytest.raises(MissionError, match="no line is held"):
        held.enqueue(login, root, script="s", resources=Resources(walltime="00:05:00"), label="x")


def test_a_submission_must_fit_one_allocation_of_the_line(
    held: Held, spool: Spool, root: str
) -> None:
    line = spool.line().model_copy(update={"walltime": "00:30:00", "deadline": time.time() + 9000})
    spool.replace(spool.path / "line.json", line.model_dump_json())

    with pytest.raises(MissionError, match="each allocation of the held line lasts 00:30:00"):
        held.enqueue(local, root, script="s", resources=Resources(walltime="00:29:00"), label="x")
    with pytest.raises(MissionError, match="has 100 GB, less than the 200 GB"):
        held.enqueue(
            local,
            root,
            script="s",
            resources=Resources(walltime="00:05:00", mem_gb=200),
            label="x",
        )


def test_cancelling_an_unclaimed_job_fences_it_so_it_can_never_run(
    held: Held, spool: Spool, root: str, node: dict[str, str], tmp_path: Path
) -> None:
    login = local
    resources = Resources(queue="held", walltime="00:05:00")
    name = held.enqueue(
        login, root, script=script(tmp_path, "true"), resources=resources, label="x"
    )
    held.cancel(login, root, handle=name)
    assert held.states(login, root, [name])[name].verdict == vocabulary.Word.CANCELLED
    thread = serve(spool, node)
    time.sleep(0.5)
    halt(spool, thread)
    assert spool.exit_of(name) is None, "a fenced job never reaches a node"


def test_cancelling_a_running_job_returns_once_it_has_ended(
    held: Held, spool: Spool, root: str, node: dict[str, str], tmp_path: Path
) -> None:
    login = local
    resources = Resources(queue="held", walltime="00:05:00")
    name = held.enqueue(
        login, root, script=script(tmp_path, "sleep 60\n"), resources=resources, label="x"
    )
    thread = serve(spool, node)
    until(lambda: held.states(login, root, [name])[name].stage == vocabulary.Word.RUNNING)
    time.sleep(0.3)
    held.cancel(login, root, handle=name)
    assert held.states(login, root, [name])[name].verdict == vocabulary.Word.FAILED
    halt(spool, thread)


def test_stopping_a_line_cancels_what_nobody_claimed_and_keeps_the_claimed(
    held: Held, spool: Spool, root: str, tmp_path: Path
) -> None:
    login = local
    waiting = entry(spool, script(tmp_path, "true"), name="h01")
    taken = entry(spool, script(tmp_path, "true"), name="h02")
    spool.claim(taken, Claim(alloc="9.opbs"))
    subprocess.run(["bash", "-c", Remote(spool_path(root)).stop()], check=True)
    states = held.states(login, root, [waiting.handle, taken.handle])
    assert states["h01"].verdict == vocabulary.Word.CANCELLED
    assert states["h02"].verdict == vocabulary.Word.RUNNING


def test_a_missing_spool_is_a_refusal_not_a_verdict(held: Held, root: str) -> None:
    login = local
    with pytest.raises(MissionError, match="cannot be read"):
        held.states(login, root, ["h01"])


def test_a_lost_reply_is_reconciled_by_asking_whether_it_landed(
    held: Held, spool: Spool, root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = held_module.login_ask
    seen: list[str] = []

    def flaky(remote, body, **kwargs):
        if "mktemp" in body and not seen:
            real(remote, body, **kwargs)
            seen.append("sent")
            raise held_module.HostUnreachable("reply lost")
        return real(remote, body, **kwargs)

    monkeypatch.setattr(held_module, "login_ask", flaky)
    resources = Resources(queue="held", walltime="00:05:00")
    name = held.enqueue(local, root, script="s", resources=resources, label="x")
    assert [s.handle for s in spool.queued()] == [name], "written once, not twice"


def test_a_queue_declaring_a_scheduler_routes_its_jobs_there() -> None:
    profile = HostProfile(kind="pbs", queues={"held": QueuePolicy(scheduler="held")})
    assert kind_of(profile, "held") == "held"
    assert kind_of(profile, "debug-g") == "pbs" == kind_of(profile)
    assert isinstance(pick(profile, "held"), Held)
    assert isinstance(Held(), Pbs), "logs and autopsy are PBS's own"
