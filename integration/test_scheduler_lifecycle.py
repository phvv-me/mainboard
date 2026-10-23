"""Cancellation, release and waiting states say only what a scheduler confirmed.

Each cancel is judged by what it requested, what the scheduler acknowledged and what is observed
after: a refused `qdel`, a spool write that failed, a lost connection or a state the table does
not know never reads as a cancelled job. A line's release lets a running job finish, and a pueue
task that has not started is waiting, never failed. Everything runs on this machine: the login
node is a local bash, `qdel` is a stub, `pueue` a fake, and the allocation is faked as in
`test_held`.
"""

import json
import os
import shlex
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from plumbum import local

from mainboard import line as line_module
from mainboard.board import Board, Job
from mainboard.core import MissionError
from mainboard.dispatch import HostUnreachable, now, vocabulary
from mainboard.dispatch.schedulers import Held, Pbs, Pueue, pueue
from mainboard.dispatch.schedulers import held as held_module
from mainboard.dispatch.spool import Claim, Spool
from mainboard.dispatch.state import RunRecord
from mainboard.line import Line
from mainboard.monitor import Monitor

from . import test_held
from .test_held import entry, script, serve, until

# The fixtures of the held-line tests: a host root, its open spool, a faked allocation, and the
# scheduler with PBS itself absent.
root, spool, node, held = test_held.root, test_held.spool, test_held.node, test_held.held

_HOST = "deadbox"


@contextmanager
def qdel_says(directory: Path, *, code: int, text: str) -> Generator[Path]:
    """A `qdel` first on PATH that logs what it was asked, then answers `code` with `text`."""
    directory.mkdir(exist_ok=True)
    calls, stub = directory / "qdel-calls", directory / "qdel"
    stub.write_text(
        f'#!/bin/sh\necho "$@" >> {shlex.quote(str(calls))}\n'
        f"echo {shlex.quote(text)} >&2\nexit {code}\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    with local.env(PATH=f"{directory}{os.pathsep}{local.env['PATH']}"):
        yield calls


def asked(calls: Path) -> list[str]:
    """The handles a stub `qdel` was asked to delete, none when it was never run."""
    return calls.read_text(encoding="utf-8").split() if calls.exists() else []


@contextmanager
def unwritable(directory: Path) -> Generator[None]:
    """`directory` made read-only, so every write into it fails."""
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o500)
    try:
        yield
    finally:
        directory.chmod(0o700)


def is_fenced(spool: Spool, name: str) -> bool:
    """Whether the center fenced `name`, rather than an allocation claiming it."""
    return Claim.model_validate_json((spool.claims / f"{name}.json").read_text()).cancelled


@pytest.fixture
def line(root: str, monkeypatch: pytest.MonkeyPatch) -> Line:
    """The line of a PBS host, reached through this machine's own bash."""
    monkeypatch.setattr(line_module, "connection", lambda host: nullcontext(local))
    return Line(SimpleNamespace(host="miyabi-g", remote_root=lambda: root))


@pytest.fixture
def serving(spool: Spool, node: dict[str, str], tmp_path: Path) -> Generator[threading.Thread]:
    """A server running h01 (ready, a second's work, `finished`) with h02 (`late`) behind it."""
    entry(
        spool,
        script(tmp_path, f"echo ready\nsleep 1\necho done > {tmp_path / 'finished'}\n"),
        name="h01",
    )
    entry(spool, script(tmp_path, f"echo late > {tmp_path / 'late'}\n"), name="h02")
    thread = serve(spool, node)
    until(lambda: (log := spool.logs / "h01.log").exists() and "ready" in log.read_text())
    yield thread
    thread.join(30)


def test_releasing_a_line_lets_its_running_job_finish_and_fences_only_the_pending(
    line: Line, spool: Spool, serving: threading.Thread, tmp_path: Path
) -> None:
    released = line.release()
    serving.join(30)
    assert released.running == "h01" and not serving.is_alive(), "released while h01 ran"
    assert (spool.exit_of("h01"), (tmp_path / "finished").exists()) == (0, True), "h01 ended"
    assert not (spool.tombstoned("h01") or is_fenced(spool, "h01")), "a live claim is spared"
    assert spool.tombstoned("h02") and is_fenced(spool, "h02")
    assert (spool.exit_of("h02"), (tmp_path / "late").exists()) == (None, False), "h02 never ran"


@pytest.mark.parametrize("blocked", ["claims", "spool"])
def test_a_release_that_could_not_fence_an_entry_fails_instead_of_reporting_it(
    line: Line, spool: Spool, tmp_path: Path, blocked: str
) -> None:
    entry(spool, script(tmp_path, "true"), name="h01")
    stuck = spool.path if blocked == "spool" else spool.claims
    with unwritable(stuck), pytest.raises(MissionError, match="could not stop"):
        line.release()
    assert not (spool.claims / "h01.json").exists()
    assert not spool.tombstoned("h01"), "nothing is tombstoned for a fence that did not win"


_UNCONFIRMED = (MissionError, "not cancelled")


@pytest.mark.parametrize(
    ("login", "error", "said"),
    [
        pytest.param(Mock(return_value=(4, "", "read-only")), *_UNCONFIRMED, id="write"),
        pytest.param(Mock(return_value=(0, "", "")), *_UNCONFIRMED, id="silence"),
        pytest.param(Mock(return_value=(0, "gone\n", "")), *_UNCONFIRMED, id="word"),
        pytest.param(Mock(side_effect=HostUnreachable("x")), HostUnreachable, "x", id="lost"),
    ],
)
def test_a_held_cancel_the_spool_did_not_confirm_never_returns(
    held: Held, monkeypatch: pytest.MonkeyPatch, login: Mock, error: type[Exception], said: str
) -> None:
    monkeypatch.setattr(held_module, "login_ask", login)
    with pytest.raises(error, match=said):
        held.cancel(local, "/fake-root", handle="h01")


@pytest.mark.parametrize("blocked", ["cancel", "claims", "spool"])
def test_a_held_cancel_whose_fence_could_not_be_written_is_not_reported(
    held: Held, spool: Spool, tmp_path: Path, blocked: str
) -> None:
    entry(spool, script(tmp_path, "true"), name="h01")
    stuck = spool.path if blocked == "spool" else spool.path / blocked
    with unwritable(stuck), pytest.raises(MissionError, match="not cancelled"):
        held.cancel(local, str(tmp_path), handle="h01")
    assert not (spool.claims / "h01.json").exists(), "no cancelled claim stands in for the fence"
    assert held.states(local, str(tmp_path), ["h01"])["h01"].verdict == vocabulary.RUNNING


def test_a_held_cancel_racing_a_claim_waits_for_the_allocation_to_end_the_job(
    held: Held, spool: Spool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission = entry(spool, script(tmp_path, "true"), name="h01")
    assert spool.claim(submission, Claim(alloc="9.opbs", node="n1", at=time.time()))
    monkeypatch.setattr(held_module, "_CANCEL_SECONDS", 0.3)
    monkeypatch.setattr(held_module, "_POLL_SECONDS", 0.05)
    with pytest.raises(MissionError, match="still running in its allocation"):
        held.cancel(local, str(tmp_path), handle="h01")
    assert spool.tombstoned("h01"), "the allocation ends it on this durable tombstone"


def test_a_held_cancel_of_a_finished_job_keeps_its_exit_artifact(
    held: Held, spool: Spool, tmp_path: Path
) -> None:
    submission = entry(spool, script(tmp_path, "true"), name="h01")
    assert spool.claim(submission, Claim(alloc="9.opbs"))
    assert spool.publish(spool.logs, "h01.exit", text="exit=3\n")
    held.cancel(local, str(tmp_path), handle="h01")
    assert spool.exit_of("h01") == 3
    assert not spool.tombstoned("h01"), "a finished job is left as it ended"
    state = held.states(local, str(tmp_path), ["h01"])["h01"]
    assert (state.verdict, state.exit_code) == (vocabulary.FAILED, 3)


@pytest.mark.parametrize(
    ("code", "text", "refused"),
    [
        pytest.param(0, "", None, id="accepted"),
        pytest.param(35, "qdel: Job has finished 12.opbs", None, id="finished"),
        pytest.param(153, "qdel: Unknown Job Id 12.opbs", None, id="unknown"),
        pytest.param(168, "qdel: invalid for state of job 12.opbs", MissionError, id="refused"),
        pytest.param(255, "ssh: Connection closed by remote host", HostUnreachable, id="lost"),
    ],
)
def test_pbs_cancels_only_what_qdel_accepted_or_found_already_gone(
    tmp_path: Path, code: int, text: str, refused: type[Exception] | None
) -> None:
    with qdel_says(tmp_path, code=code, text=text) as calls:
        if refused is None:
            Pbs().cancel(local, "/fake-root", handle="12.opbs")
        else:
            with pytest.raises(refused, match=r"qdel|Connection closed"):
                Pbs().cancel(local, "/fake-root", handle="12.opbs")
    assert asked(calls) == ["12.opbs"], "it was requested; only an accepted one is acknowledged"


def test_a_line_frees_its_allocation_through_the_same_checked_qdel(
    line: Line, spool: Spool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spool.replace(spool.path / "keeper.beat", str(int(time.time())))
    spool.replace(spool.path / "gen/1/alloc", "7.opbs")
    spool.replace(spool.path / "gen/1/beat", f'{{"at": {time.time()}, "state": "ready"}}')
    monkeypatch.setattr(line_module, "_ENDS", 0.0)
    refusal = "qdel: Request invalid for state of job 7.opbs"
    with (
        qdel_says(tmp_path / "refusing", code=168, text=refusal),
        pytest.raises(MissionError, match=r"qdel 7\.opbs failed \(168\)"),
    ):
        line.release()
    with qdel_says(tmp_path / "late", code=35, text="qdel: Job has finished 7.opbs") as calls:
        line.release()
    assert asked(calls) == ["7.opbs"]


class FakePueue:
    """A host whose pueue daemon lists `tasks` and records every verb it is asked but `status`."""

    def __init__(self, *tasks: Mapping[str, object]) -> None:
        self.listing = json.dumps({"tasks": {str(task["id"]): task for task in tasks}})
        self.verbs: list[list[str]] = []

    def __getitem__(self, key: str | list[str]) -> FakePueue | Callable[[], str]:
        if key == "pueue":
            return self
        words = list(key)
        return lambda: self.ask(words)

    def ask(self, words: list[str]) -> str:
        if words[0] != "status":
            self.verbs.append(words)
            return ""
        return self.listing


_DONE = {"enqueued_at": "t", "start": "t", "end": "t"}
_HOLD, _GO = {"enqueued_at": "t", "start": "t"}, ("running", "running", False)
_WAIT, _STAY = ("running", "queued", False), ("running", "", False)
_OVER, _FAIL = ("ok", "", True), ("failed", "", True)

# A status pueue prints, its (verdict, stage, settled), and how a cancel reaches it (None leaves
# it alone, "raises" refuses a state this tool cannot place).
_VARIANTS = [
    pytest.param({"Locked": {"previous_status": "Queued"}}, _WAIT, "remove", id="Locked"),
    pytest.param("Locked", _WAIT, "remove", id="Locked-bare"),
    pytest.param({"Stashed": {"enqueue_at": None}}, _WAIT, "remove", id="Stashed"),
    pytest.param({"Queued": {"enqueued_at": "t"}}, _WAIT, "remove", id="Queued"),
    pytest.param({"Running": _HOLD}, _GO, "kill", id="Running"),
    pytest.param({"Paused": _HOLD}, _STAY, "kill", id="Paused"),
    pytest.param({"Done": {**_DONE, "result": "Success"}}, _OVER, None, id="Done-success"),
    pytest.param({"Done": {**_DONE, "result": {"Failed": 3}}}, _FAIL, None, id="Done-3"),
    pytest.param({"Done": {**_DONE, "result": "Killed"}}, _FAIL, None, id="Done-killed"),
    pytest.param({"Frozen": {}}, _STAY, "raises", id="unknown"),
    pytest.param("Frozen", _STAY, "raises", id="unknown-bare"),
    pytest.param(None, ("vanished", "", True), None, id="forgotten"),
]


def test_the_pueue_table_names_every_state_pueue_has() -> None:
    tags = {
        status if isinstance(status, str) else next(iter(status))
        for status, *_ in (case.values for case in _VARIANTS)
        if status is not None
    }
    assert {state.value for state in pueue.PueueState} <= tags


@pytest.mark.parametrize(("status", "meaning", "stop"), _VARIANTS)
def test_every_pueue_state_has_one_meaning_for_listing_settling_and_cancelling(
    status: str | Mapping[str, object] | None, meaning: tuple[str, str, bool], stop: str | None
) -> None:
    task = [] if status is None else [{"id": 7, "label": "job", "status": status}]
    host = FakePueue(*task)
    found = Pueue().states(host, "/fake-root", ["7"])["7"]
    assert (found.verdict, found.stage, found.verdict in vocabulary.TERMINAL) == meaning
    if stop == "raises":
        with pytest.raises(MissionError, match="pueue task 7"):
            Pueue().cancel(host, "/fake-root", handle="7")
        assert host.verbs == [], "a state this tool cannot place is not stopped by a guess"
    else:
        Pueue().cancel(host, "/fake-root", handle="7")
        assert host.verbs == ([[stop, "7"]] if stop else [])


@pytest.fixture
def owing(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Board:
    """A workspace owing one run on a declared ssh host, which answers every evidence request."""
    manifest = f'[workspace]\nname = "it"\n[hosts.{_HOST}]\nkind = "ssh"\nroot = "/srv/jobs"\n'
    (workspace / "mb.toml").write_text(manifest, encoding="utf-8", newline="\n")
    board = Board(workspace)
    board.dispatcher.cache.record(
        RunRecord(
            handle="7",
            target=_HOST,
            kind="ssh",
            script="python train.py",
            args="",
            submitted_at=now(),
            name="run-7",
        )
    )
    monkeypatch.setattr(Monitor, "capture", lambda self, record, run, required=True: ())
    monkeypatch.setattr(Monitor, "pull", lambda self, run: None)
    monkeypatch.setattr(Monitor, "verify", lambda self, *args, **kwargs: None)
    return board


def test_a_stop_the_scheduler_refused_leaves_the_run_in_flight(
    owing: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Job, "kill", Mock(side_effect=MissionError("qdel 7 failed (168): state")))
    with pytest.raises(MissionError, match="qdel 7 failed"):
        owing.verdicts().cancel("7")
    assert owing.dispatcher.cache.run("7", _HOST).verdict not in vocabulary.TERMINAL


def test_a_stop_the_scheduler_accepted_settles_the_run_cancelled(
    owing: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    kill = Mock()
    monkeypatch.setattr(Job, "kill", kill)
    assert owing.verdicts().cancel("7").code == 1
    kept = owing.dispatcher.cache.run("7", _HOST)
    assert (kept.verdict, kept.reported) == (vocabulary.CANCELLED, vocabulary.CANCELLED)
    kill.assert_called_once_with()
