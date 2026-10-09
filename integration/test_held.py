"""The held line: its spool protocol, the node server, the center scheduler and the keeper loop.

Everything runs on this machine against a temporary directory: the center's shell is executed by
a local bash (plumbum's `local` is a `Machine`), the node's allocation is faked by the two
environment variables a PBS job has, and `qsub` is a stub on PATH. No ssh, no cluster.
"""

import os
import socket
import stat
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from plumbum import local

from mainboard.core import MissionError
from mainboard.dispatch import vocabulary
from mainboard.dispatch.dispatcher import Dispatcher
from mainboard.dispatch.schedulers import Held, Pbs, kind_of, pick
from mainboard.dispatch.schedulers import held as held_module
from mainboard.dispatch.shared import now
from mainboard.dispatch.spool import (
    Beat,
    Claim,
    LineSpec,
    Remote,
    Spool,
    Submission,
    spool_path,
)
from mainboard.dispatch.state.cache import Cache, RunRecord
from mainboard.dispatch.vocabulary import Resources
from mainboard.line import Line, keeper_script
from mainboard.manifest import HostProfile, QueuePolicy
from mainboard.serve import Server

if TYPE_CHECKING:
    from mainboard.board import Board


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A host workspace root; the spool and the runner's logs live beneath it."""
    return str(tmp_path)


@pytest.fixture
def spool(root: str) -> Spool:
    """The node's view of the spool at `root`, with a line open for an hour."""
    opened = Spool(Path(spool_path(root)))
    line = LineSpec(
        queue="interact-g",
        walltime="02:00:00",
        mem_gb=100,
        account="g",
        deadline=time.time() + 3600,
        created=time.time(),
        tool="mainboard",
    )
    opened.replace(opened.path / "line.json", line.model_dump_json())
    return opened


def entry(spool: Spool, script: str, *, walltime: str = "00:05:00", name: str = "") -> Submission:
    """Publish a queued job running `script` the way the center would."""
    submission = Submission(
        handle=name or f"h{time.time_ns():x}",
        label="mainboard-test",
        script=script,
        cwd=str(spool.path),
        walltime=walltime,
        submitted="now",
    )
    assert spool.publish(
        spool.inbox, f"{submission.handle}.json", text=submission.model_dump_json()
    )
    return submission


def script(tmp_path: Path, body: str) -> str:
    """A job script file running `body`."""
    path = tmp_path / f"job-{time.time_ns():x}.sh"
    path.write_text(body, encoding="utf-8")
    return str(path)


@pytest.fixture
def node(tmp_path: Path) -> dict[str, str]:
    """The two variables that make a process look like it runs inside a PBS allocation."""
    nodefile = tmp_path / "nodefile"
    nodefile.write_text(socket.gethostname() + "\n", encoding="utf-8")
    return {**os.environ, "PBS_JOBID": "123.opbs", "PBS_NODEFILE": str(nodefile)}


def serve(spool: Spool, node: dict[str, str], *, walltime: str = "01:00:00") -> threading.Thread:
    """Run a server for `spool` in a thread, quickly polling, and answer the thread."""
    server = Server(
        spool, gen=1, walltime=walltime, environ=node, sleep=lambda _: time.sleep(0.02)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def until(condition, seconds: float = 20.0) -> None:
    """Wait for `condition()` to hold, failing the test after `seconds`."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def test_one_claimant_wins_a_contended_entry(spool: Spool, tmp_path: Path) -> None:
    submission = entry(spool, script(tmp_path, "true"))
    with ThreadPoolExecutor(16) as pool:
        won = list(pool.map(lambda n: spool.claim(submission, Claim(gen=n)), range(64)))
    assert won.count(True) == 1
    assert spool.queued() == []
    assert not list(spool.claims.glob(".tmp.*")), "temporary names are removed"


def test_oldest_fitting_entry_is_next_and_claimed_or_cancelled_ones_are_skipped(
    spool: Spool, tmp_path: Path
) -> None:
    first = entry(spool, script(tmp_path, "true"), name="h01")
    second = entry(spool, script(tmp_path, "true"), name="h02")
    third = entry(spool, script(tmp_path, "true"), name="h03")
    assert [s.handle for s in spool.queued()] == ["h01", "h02", "h03"]
    spool.claim(first, Claim())
    spool.publish(spool.cancel, second.handle, text="")
    assert [s.handle for s in spool.queued()] == [third.handle]


def test_the_node_runs_jobs_in_order_and_leaves_an_exit_artifact_each(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    passing = entry(spool, script(tmp_path, "echo first\n"), name="h01")
    bad = entry(spool, script(tmp_path, "echo second; exit 3\n"), name="h02")
    thread = serve(spool, node)
    until(lambda: spool.exit_of(bad.handle) is not None)
    (spool.path / "stop").touch()
    thread.join(20)
    assert (spool.exit_of(passing.handle), spool.exit_of(bad.handle)) == (0, 3)
    assert "first" in (spool.logs / f"{passing.handle}.log").read_text()
    claim = Claim.model_validate_json((spool.claims / f"{passing.handle}.json").read_text())
    assert (claim.alloc, claim.gen, claim.node) == (
        "123.opbs",
        1,
        socket.gethostname().split(".")[0],
    )
    assert (spool.path / "gen" / "1" / "ended").exists()


def test_a_tombstone_ends_a_running_job_with_a_terminated_exit(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    sleeper = entry(spool, script(tmp_path, "sleep 60\n"), name="h01")
    thread = serve(spool, node)
    until(lambda: (spool.claims / "h01.json").exists())
    time.sleep(0.3)
    spool.publish(spool.cancel, sleeper.handle, text="")
    until(lambda: spool.exit_of(sleeper.handle) is not None)
    assert spool.exit_of(sleeper.handle) in {143, -15}
    (spool.path / "stop").touch()
    thread.join(20)


def test_a_job_that_cannot_fit_the_allocation_waits_and_the_server_ends_early(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    big = entry(spool, script(tmp_path, "true"), walltime="00:00:30", name="h01")
    thread = serve(spool, node, walltime="00:02:02")
    thread.join(20)
    assert not thread.is_alive(), "no job fits, so the allocation is given back"
    assert [s.handle for s in spool.queued()] == [big.handle]


def test_a_server_refuses_to_run_outside_an_allocation(spool: Spool, tmp_path: Path) -> None:
    server = Server(spool, gen=1, walltime="01:00:00", environ={"PATH": os.environ["PATH"]})
    with pytest.raises(MissionError, match="PBS allocation"):
        server.run()


def test_the_beat_names_the_job_in_flight(
    spool: Spool, node: dict[str, str], tmp_path: Path
) -> None:
    entry(spool, script(tmp_path, "sleep 1\n"), name="h01")
    thread = serve(spool, node)
    until(lambda: (spool.path / "gen" / "1" / "beat").exists())
    until(
        lambda: Beat.model_validate_json((spool.path / "gen/1/beat").read_text()).running == "h01"
    )
    (spool.path / "stop").touch()
    thread.join(20)


@pytest.fixture
def held(monkeypatch: pytest.MonkeyPatch) -> Held:
    """The scheduler with PBS itself absent: no allocation is ever reported ended."""
    monkeypatch.setattr(Pbs, "states", lambda self, remote, root, handles: {})
    return Held()


def test_a_submission_is_queued_then_running_then_settled_from_its_exit(
    held: Held, spool: Spool, root: str, node: dict[str, str], tmp_path: Path
) -> None:
    resources = Resources(queue="held", walltime="00:05:00")
    login = local
    name = held.enqueue(
        login, root, script=script(tmp_path, "sleep 1\n"), resources=resources, label="x"
    )
    assert held.states(login, root, [name])[name].stage == vocabulary.QUEUED
    thread = serve(spool, node)
    until(lambda: held.states(login, root, [name])[name].stage == vocabulary.RUNNING)
    until(lambda: held.states(login, root, [name])[name].verdict == vocabulary.OK)
    assert held.states(login, root, [name])[name].exit_code == 0
    (spool.path / "stop").touch()
    thread.join(20)


def test_a_claim_without_an_exit_is_lost_only_when_pbs_says_its_allocation_ended(
    held: Held, spool: Spool, root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission = entry(spool, script(tmp_path, "true"), name="h01")
    spool.claim(submission, Claim(alloc="123.opbs", node="n1", at=time.time()))
    login = local
    assert held.states(login, root, ["h01"])["h01"].verdict == vocabulary.RUNNING
    ended = vocabulary.OK
    monkeypatch.setattr(
        Pbs,
        "states",
        lambda self, remote, path, handles: {"123.opbs": SimpleNamespace(verdict=ended)},
    )
    assert held.states(login, root, ["h01"])["h01"].verdict == vocabulary.VANISHED
    spool.publish(spool.logs, "h01.exit", text="exit=0\n")
    assert held.states(login, root, ["h01"])["h01"].verdict == vocabulary.OK


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


def test_cancelling_an_unclaimed_job_fences_it_so_it_can_never_run(
    held: Held, spool: Spool, root: str, node: dict[str, str], tmp_path: Path
) -> None:
    login = local
    resources = Resources(queue="held", walltime="00:05:00")
    name = held.enqueue(
        login, root, script=script(tmp_path, "true"), resources=resources, label="x"
    )
    held.cancel(login, root, handle=name)
    assert held.states(login, root, [name])[name].verdict == vocabulary.CANCELLED
    thread = serve(spool, node)
    time.sleep(0.5)
    (spool.path / "stop").touch()
    thread.join(20)
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
    until(lambda: held.states(login, root, [name])[name].stage == vocabulary.RUNNING)
    time.sleep(0.3)
    held.cancel(login, root, handle=name)
    assert held.states(login, root, [name])[name].verdict == vocabulary.FAILED
    (spool.path / "stop").touch()
    thread.join(20)


def test_stopping_a_line_cancels_what_nobody_claimed_and_keeps_the_claimed(
    held: Held, spool: Spool, root: str, tmp_path: Path
) -> None:
    login = local
    waiting = entry(spool, script(tmp_path, "true"), name="h01")
    taken = entry(spool, script(tmp_path, "true"), name="h02")
    spool.claim(taken, Claim(alloc="9.opbs"))
    subprocess.run(["bash", "-c", Remote(spool_path(root)).stop()], check=True)
    states = held.states(login, root, [waiting.handle, taken.handle])
    assert states["h01"].verdict == vocabulary.CANCELLED
    assert states["h02"].verdict == vocabulary.RUNNING


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


def status_for(spool: Spool, **files: str) -> Line:
    """A line whose spool holds `files`, read through a local shell."""
    for name, text in files.items():
        spool.replace(spool.path / name.replace("__", "/"), text)
    return Line(cast("Board", SimpleNamespace(host="miyabi-g")))


def test_a_line_reads_as_what_its_spool_adds_up_to(spool: Spool) -> None:
    line = status_for(spool)
    remote = Remote(str(spool.path))
    now = time.time()
    assert line._status(local, remote).state == "unkept"
    spool.replace(spool.path / "keeper.beat", str(int(now)))
    assert line._status(local, remote).state == "opening"
    spool.replace(spool.path / "gen/1/alloc", "7.opbs")
    spool.replace(spool.path / "gen/1/beat", Beat(at=now, state="ready").model_dump_json())
    ready = line._status(local, remote)
    assert (ready.state, ready.alloc, ready.generation) == ("ready", "7.opbs", 1)
    spool.replace(
        spool.path / "gen/1/beat", Beat(at=now, state="running", running="h1").model_dump_json()
    )
    assert line._status(local, remote).state == "busy"
    spool.replace(spool.path / "gen/1/ended", "1")
    assert line._status(local, remote).state == "renewing"
    (spool.path / "stop").touch()
    assert line._status(local, remote).state == "ended"


def fake_qsub(directory: Path, body: str) -> dict[str, str]:
    """A `qsub` on PATH running `body`, which sees the spool's generation as `$GEN`."""
    stub = directory / "qsub"
    stub.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return {**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"}


def test_the_keeper_opens_a_generation_per_allocation_until_released(
    spool: Spool, tmp_path: Path
) -> None:
    text = keeper_script(
        spool.line(),
        spool=str(spool.path),
        token="abc",
        flags=["-q", "interact-g", "-l", "select=1:mem=100gb"],
        template="serve --gen @GEN@ --walltime @WALL@",
    )
    (spool.path / "keeper.token").write_text("abc")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "calls"
    env = fake_qsub(
        bin_,
        f'echo "$@" >> {calls}; n=$(ls {spool.path}/gen | sort -n | tail -1); '
        f'echo "j$n" > {spool.path}/gen/$n/alloc; '
        f'[ "$n" -ge 2 ] && touch {spool.path}/stop; exit 0',
    )
    done = subprocess.run(["bash", "-c", text], env=env, timeout=60, check=False)
    assert done.returncode == 0
    first, second = calls.read_text().splitlines()
    assert first.startswith("-I -q interact-g -l select=1:mem=100gb -l walltime=0")
    assert "-N mbhold -- /bin/bash -lc serve --gen 1 --walltime " in first
    assert "--gen 2 --walltime " in second
    assert (spool.path / "keeper.beat").exists() and (spool.path / "keeper.json").exists()
    assert "keeper finished" in (spool.path / "keeper.log").read_text()


def test_an_older_keeper_exits_once_a_newer_one_holds_the_token(
    spool: Spool, tmp_path: Path
) -> None:
    text = keeper_script(
        spool.line(),
        spool=str(spool.path),
        token="old",
        flags=[],
        template="serve",
    )
    (spool.path / "keeper.token").write_text("new")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    env = fake_qsub(bin_, "exit 99")
    done = subprocess.run(["bash", "-c", text], env=env, timeout=30, check=False)
    assert done.returncode == 0
    assert not (spool.path / "gen" / "1").exists(), "a fenced keeper opens nothing"


def test_the_keeper_script_is_valid_bash(spool: Spool) -> None:
    text = keeper_script(
        spool.line(), spool="/a b/c", token="t", flags=["-q", "x y"], template="serve 'q'"
    )
    assert subprocess.run(["bash", "-n"], input=text, text=True, check=False).returncode == 0
    assert LineSpec.grace == 120


def test_a_held_submission_is_recorded_before_it_is_queued_and_closed_when_refused(
    spool: Spool, root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Pbs, "states", lambda self, remote, path, handles: {})
    dispatcher = Dispatcher(cache=Cache.private(), root=tmp_path)
    record = RunRecord(
        handle="",
        target="miyabi-g",
        kind="held",
        script="echo hi",
        args="",
        submitted_at=now(),
    )
    resources = Resources(queue="held", walltime="00:05:00")
    handle = dispatcher._handed(
        Held(), local, root, record, script="job.sh", args=(), resources=resources
    )
    queued = dispatcher.cache.run(handle, "miyabi-g")
    assert (queued.verdict, queued.kind) == (vocabulary.QUEUED, "held")
    assert queued.creation.startswith("mainboard-"), "the intent's label is kept"
    (spool.path / "stop").touch()
    with pytest.raises(MissionError, match="no line is held"):
        dispatcher._handed(
            Held(), local, root, record, script="job.sh", args=(), resources=resources
        )
    refused = [run for run in dispatcher.cache.recent() if run.handle != handle]
    assert [run.verdict for run in refused] == [vocabulary.FAILED]


def test_only_a_declared_pbs_host_has_a_line() -> None:
    hosts = {"p": HostProfile(kind="pbs"), "g": HostProfile(kind="ssh")}
    board = SimpleNamespace(
        manifest=SimpleNamespace(profiles=lambda: hosts),
        on=lambda host: SimpleNamespace(host=host),
    )
    kept = Line.of(cast("Board", board), "p")
    assert kept is not None and kept.host == "p"
    assert Line.of(cast("Board", board), "g") is None
    assert Line.of(cast("Board", board), "rented-4090") is None
