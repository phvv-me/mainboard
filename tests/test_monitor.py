import json
import logging
from collections.abc import Callable, Sequence
from itertools import islice
from typing import TYPE_CHECKING, ClassVar

import pytest
from plumbum.commands.processes import ProcessExecutionError

from mainboard import Board, Job
from mainboard.batch.runner import directory
from mainboard.dispatch import SshTransport
from mainboard.dispatch.backends import HpcAiBackend, VastBackend
from mainboard.dispatch.schedulers import HostUnreachable
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import JobState
from mainboard.experiments import StudyLedger
from mainboard.experiments.identity import study_label

from .dispatch.backends.support import FakeTransport, refused

if TYPE_CHECKING:
    from urllib.request import Request

    from mainboard.dispatch import Handle

    from .dispatch.backends.support import Reply

_HOST = "miyabi-g"
_STUDY = "ec15c1b1e073"
_VAST_INSTANCE = "https://console.vast.ai/api/v0/instances/13/"


class Rented(VastBackend):
    """Vast's own backend under a test-only kind, so a sweep drives its real cancel path.

    The sweep builds its backend out of the registry rather than out of the test, so the queued
    replies and the calls that answered them live on the class, the one channel the two share.
    """

    name = "vast-rental"
    replies: ClassVar[list[Reply]] = []
    calls: ClassVar[list[Request]] = []

    def __init__(self) -> None:
        transport = FakeTransport(*Rented.replies)
        Rented.calls = transport.calls
        super().__init__(transport=transport, sleeper=lambda _: None)


class Instance(HpcAiBackend):
    """HPC-AI's own backend under a test-only kind, sharing its queue the same way."""

    name = "hpc-ai-rental"
    replies: ClassVar[list[Reply]] = []
    calls: ClassVar[list[Request]] = []

    def __init__(self) -> None:
        transport = FakeTransport(*Instance.replies)
        Instance.calls = transport.calls
        super().__init__(transport=transport)


def rented(status: str = "exited", *, exit_code: int = 0) -> list[Reply]:
    """The replies one Vast post-mortem takes, the instance row, its log url, and the log.

    Any container that has been up costs a log fetch, since the wrapper's marker is the only
    thing that knows the command ended and the container's own status does not. A run that
    settles on that marker then reads the log a second time, to capture the output before the
    release destroys the instance, so the upload pair is queued twice.
    """
    log = f"training done\nmainboard-exit:{exit_code}\n"
    upload: list[Reply] = [{"result_url": "https://s3.example/logs/7.log"}, log]
    pending = status in {"created", "loading"}
    return [
        {"instances": {"id": 7, "actual_status": status}},
        *([] if pending else [*upload, *upload]),
    ]


def listed(handle: str, status: str) -> Reply:
    """The one `/instance/list` page HPC-AI answers for `handle`, at `status`."""
    return {
        "instances": [
            {
                "instanceMetadata": {"instanceId": handle},
                "instanceRuntimeInfo": {"status": status},
            }
        ],
        "pager": {"currentPage": 1, "pageSize": 50, "totalEntries": 1},
    }


def seed(
    handle: str,
    *,
    target: str = _HOST,
    kind: str = "pbs",
    name: str = "",
    fetch_path: str | None = None,
    verdict: str | None = None,
    reported: str | None = None,
) -> RunRecord:
    """Record one dispatched run in the shared cache and hand it back."""
    run = RunRecord(
        handle=handle,
        target=target,
        kind=kind,
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at=f"2026-08-17T00:00:{handle.zfill(2)}",
        name=name,
        fetch_path=fetch_path,
        verdict=verdict,
        reported=reported,
    )
    Cache().record(run)
    return run


def probing(
    board: Board, monkeypatch: pytest.MonkeyPatch, answer: Callable[[Handle], JobState]
) -> list[list[str]]:
    """Pin the board's batched scheduler probe to `answer`, one entry per round trip it made.

    The seam is the batched probe rather than the single one, since a sweep asks each host once
    about every handle it still owes an answer on. What the returned list therefore says is both
    which handles were probed and how many times the host was actually reached for them.
    """
    trips: list[list[str]] = []

    def states(handles: Sequence[Handle]) -> dict[str, JobState]:
        trips.append([handle.id for handle in handles])
        return {handle.id: answer(handle) for handle in handles}

    monkeypatch.setattr(board.dispatcher, "states", states)
    return trips


def finishing(verdict: str = "ok", exit_code: int | None = 0) -> Callable[[Handle], JobState]:
    """A probe answer settling every handle on the same terminal verdict."""
    return lambda handle: JobState(
        handle=handle.id, state="F", exit_code=exit_code, verdict=verdict
    )


def test_a_still_running_job_is_counted_and_its_state_memoized(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("1")
    probing(
        board, monkeypatch, lambda handle: JobState(handle=handle.id, state="R", verdict="running")
    )
    report = board.monitor().once()
    assert report.running == 1
    assert not report.changed
    assert board.dispatcher.cache.run("1").state == "R"


def test_a_finished_job_is_pulled_reported_and_announced_once(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("2", fetch_path="results/run")
    trips = probing(board, monkeypatch, finishing())
    pulled: list[str] = []
    monkeypatch.setattr(board.dispatcher, "fetch", lambda handle, **kw: pulled.append(handle.id))
    report = board.monitor().once()
    assert [(item.handle, item.target, item.pulled_path) for item in report.finished] == [
        ("2", _HOST, "results/run")
    ]
    assert report.changed and pulled == ["2"]
    assert board.dispatcher.cache.run("2").reported == "ok"
    again = board.monitor().once()
    assert not again.changed and again.running == 0
    assert trips == [["2"]]  # the settled run is never probed a second time


@pytest.mark.parametrize(
    "fetch_path",
    [None, "results/run"],
    ids=[
        "a run that recorded no results path pulls nothing",
        "a pull that fails leaves the job finished without a path",
    ],
)
def test_a_finished_job_reports_only_the_results_it_could_actually_bring_back(
    board: Board, monkeypatch: pytest.MonkeyPatch, fetch_path: str | None
) -> None:
    """The verdict lands whatever the transfer did.

    One missing artifact is a warning in the log, never a sweep that dies holding every
    other job's outcome.
    """
    seed("3", fetch_path=fetch_path)
    probing(board, monkeypatch, finishing())

    def explode(handle: Handle, **kw: SshTransport | None) -> None:
        raise ProcessExecutionError(["rsync"], 23, "", "no such file")

    monkeypatch.setattr(board.dispatcher, "fetch", explode)
    [item] = board.monitor().once().finished
    assert item.pulled_path is None
    assert board.dispatcher.cache.run("3").reported == "ok"


def test_a_failed_job_carries_a_network_free_reason(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("5")
    probing(board, monkeypatch, finishing(verdict="failed", exit_code=137))
    report = board.monitor().once()
    assert [(item.handle, item.target) for item in report.failed] == [("5", _HOST)]
    assert "memory" in report.failed[0].reason


@pytest.mark.parametrize(
    ("reported", "changed"),
    [
        (None, True),
        ("ok", False),
    ],
    ids=[
        "a cached terminal verdict is harvested without probing",
        "a settled run the sweep already reported is not tracked again",
    ],
)
def test_a_verdict_the_cache_already_holds_costs_no_probe(
    board: Board, monkeypatch: pytest.MonkeyPatch, reported: str | None, changed: bool
) -> None:
    """A terminal verdict can never change.

    That is also what keeps a finished job the queue has already forgotten from reading back
    as vanished.
    """
    seed("6", verdict="ok", reported=reported)
    trips = probing(board, monkeypatch, finishing())
    report = board.monitor().once()
    assert report.changed is changed
    assert [item.handle for item in report.finished] == (["6"] if changed else [])
    assert trips == []


def test_a_finished_trial_settles_the_study_that_owns_it(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("8", name=study_label(_STUDY))
    probing(board, monkeypatch, finishing())
    board.monitor().once()
    assert StudyLedger(board.root, _STUDY).statuses() == {"8": "ok"}


def test_a_down_host_is_reported_once_and_its_jobs_left_for_the_next_pass(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("9")
    seed("10")

    def unreachable(handle: Handle) -> JobState:
        raise HostUnreachable("ssh connect to 'miyabi-g' failed: connection timed out")

    trips = probing(board, monkeypatch, unreachable)
    report = board.monitor().once()
    assert [(host.host, "timed out" in host.reason) for host in report.unreachable_hosts] == [
        (_HOST, True)
    ]
    assert report.running == 0 and not report.changed
    assert trips == [["10", "9"]]  # one round trip carried both jobs on the host, newest first
    assert len(board.dispatcher.cache.tracked()) == 2


def test_a_host_the_manifest_can_no_longer_resolve_is_reported_not_raised(board: Board) -> None:
    seed("11", target="gold")  # the fixture's gold profile declares no root
    [host] = board.monitor().once().unreachable_hosts
    assert host.host == "gold"
    assert "root" in host.reason


def test_a_target_that_will_not_answer_is_asked_once_whatever_kinds_its_runs_carry(
    board: Board,
) -> None:
    """A host redeclared under another scheduler leaves older runs carrying the older kind.

    Which scheduler answers for a run is the kind it was dispatched under, so those runs are two
    groups on one target. The target is still one machine, so the second group is not reached for
    once the first has said the machine is not there.
    """
    seed("19", target="gold", kind="pbs")
    seed("20", target="gold", kind="ssh")
    report = board.monitor().once()
    assert [(host.host, "root" in host.reason) for host in report.unreachable_hosts] == [
        ("gold", True)
    ]
    assert len(board.dispatcher.cache.tracked()) == 2


@pytest.mark.parametrize(
    ("backend", "replies", "ended"),
    [
        (Rented, [*rented(), {"success": True}], [_VAST_INSTANCE]),
        (
            Instance,
            [listed("13", "Stopped"), {}, {}],  # hpc-ai keeps no server-side log to capture
            [
                "https://www.hpc-ai.com/api/instance/stop",
                "https://www.hpc-ai.com/api/instance/terminate",
            ],
        ),
    ],
    ids=[
        "vast restarts the exited container until someone cancels",
        "an hpc-ai instance runs until it is terminated, whatever its command did",
    ],
)
def test_a_finished_rental_is_settled_and_then_ended(
    board: Board,
    monkeypatch: pytest.MonkeyPatch,
    backend: type[Rented] | type[Instance],
    replies: Sequence[Reply],
    ended: list[str],
) -> None:
    """A provider run is cancelled the moment its verdict is terminal.

    A finished command does not end a provider run, so the cancel follows here where the
    scheduler path deliberately never makes one.
    """
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    monkeypatch.setenv("HPCAI_API_KEY", "key-123")
    seed("13", target="rented", kind=backend.name)
    backend.replies = replies
    report = board.monitor().once()
    assert [item.handle for item in report.finished] == ["13"]
    assert [call.full_url for call in backend.calls[-len(ended) :]] == ended
    assert board.dispatcher.cache.run("13").reported == "ok"


def test_a_rental_vast_restarted_settles_on_its_marker_and_stops_billing(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The money leak end to end: a finished command whose container Vast brought straight back.

    An instance is held at its intended status, so the container the command exited from restarts
    and reads `running` again. Every sweep believed it, so the run never settled, the rental was
    never cancelled, and eight instances were still billing after their campaign had finished
    ($2.35 against $0.65 expected, 2026-08-26). The marker settles it and the release destroys it.
    """
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("23", target="rented", kind=Rented.name)
    Rented.replies = [*rented(status="running"), {"success": True}]
    assert [item.handle for item in board.monitor().once().finished] == ["23"]
    assert (Rented.calls[-1].full_url, Rented.calls[-1].get_method()) == (
        "https://console.vast.ai/api/v0/instances/23/",
        "DELETE",
    )
    assert board.dispatcher.cache.run("23").reported == "ok"


def test_the_cancel_verb_destroys_the_rental_rather_than_leaving_it_stopped(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every path that lets a rental go reaches one call, and that call has to be the destroy.

    A stopped Vast instance is not a released one: it still holds its machine and still bills for
    the disk it sits on. The verb kills through the backend and then releases, which is the same
    idempotent destroy twice by design, since a cancel killed between the two must repeat rather
    than leave the meter running.
    """
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("24", target="rented", kind=Rented.name)
    Rented.replies = [{"success": True}, {"success": True}]
    settled = board.verdicts().cancel("24")
    assert settled.trials[0].verdict == "cancelled"
    assert [(call.full_url, call.get_method()) for call in Rented.calls] == [
        ("https://console.vast.ai/api/v0/instances/24/", "DELETE")
    ] * 2


def test_a_finished_scheduler_job_is_never_cancelled(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queue stops charging when the job ends, so a finished pueue job needs no kill."""
    seed("15")
    probing(board, monkeypatch, finishing())
    killed: list[str] = []
    monkeypatch.setattr(Job, "kill", lambda self: killed.append(self.handle.id))
    assert [item.handle for item in board.monitor().once().finished] == ["15"]
    assert killed == []


def test_a_settled_rentals_output_and_receipts_come_home_before_the_instance_is_destroyed(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the exit code used to survive a run, and a rented disk dies with the rental.

    The capture must happen before the release, since releasing destroys the instance and its
    log goes with it, and the framed receipts are lifted out so `verdict` can settle on trials
    a printed line would have been truncated out of.
    """
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("21", target="rented", kind=Rented.name, name="trial-a")
    receipt = json.dumps({"trial_receipt": {"run_id": "r1", "outcome": "passed"}})
    log = f"epoch 1\n{receipt}\nmainboard-exit:0\n"
    upload: list[Reply] = [{"result_url": "https://s3.example/logs/7.log"}, log]
    Rented.replies = [
        {"instances": {"id": 7, "actual_status": "exited"}},
        *upload,
        *upload,
        {"success": True},
    ]
    board.monitor().once()
    under = directory(board, "trial-a")
    assert "epoch 1" in (under / "21.log").read_text(encoding="utf-8")
    assert receipt in (under / "receipts.ndjson").read_text(encoding="utf-8")
    # The read happened while the instance still existed, so the destroy is the last call made.
    assert Rented.calls[-1].get_method() == "DELETE"


def test_a_run_whose_backend_keeps_no_output_captures_nothing_and_still_settles(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hpc-ai keeps no server-side log, so an empty transcript is a quiet skip not a refusal."""
    monkeypatch.setenv("HPCAI_API_KEY", "key-123")
    seed("22", target="rented", kind=Instance.name, name="trial-b")
    Instance.replies = [listed("22", "Stopped"), {}, {}]
    assert [item.handle for item in board.monitor().once().finished] == ["22"]
    assert not (directory(board, "trial-b") / "22.log").exists()


def test_a_provider_that_refuses_the_cancel_is_a_warning_not_a_failed_sweep(
    board: Board, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One rental this pass could not end must not cost every other job its outcome."""
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("16", target="rented", kind=Rented.name)
    Rented.replies = [*rented(exit_code=1), refused(500)]
    caplog.set_level(logging.WARNING)
    report = board.monitor().once()
    assert [item.handle for item in report.failed] == ["16"]
    assert "could not release 16" in caplog.text
    assert board.dispatcher.cache.run("16").reported == "failed"


def test_a_provider_that_cannot_deliver_still_settles_and_ends_the_rental(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rented disk dies with the instance, so the missing artifact is a note, not a stop."""
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("17", target="rented", kind=Rented.name, fetch_path="results/run")
    Rented.replies = [*rented(), {"success": True}]
    [item] = board.monitor().once().finished
    assert item.pulled_path is None
    assert Rented.calls[-1].get_method() == "DELETE"


def test_a_provider_api_that_refuses_the_probe_is_reported_as_a_down_target(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("18", target="rented", kind=Rented.name)
    Rented.replies = [refused(503)]
    [target] = board.monitor().once().unreachable_hosts
    assert target.host == "rented" and "503" in target.reason


def test_watch_repeats_the_pass_at_the_given_interval(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("12")
    probing(board, monkeypatch, finishing())
    passes = list(islice(board.monitor().watch(0.0), 2))
    assert [report.changed for report in passes] == [True, False]
