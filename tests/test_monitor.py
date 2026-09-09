import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from getpass import getuser
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest
from filelock import FileLock
from plumbum.commands.processes import ProcessExecutionError

from mainboard import Board, Job, MissionError
from mainboard.batch.runner import directory
from mainboard.cli import build
from mainboard.costs.catalog import Offer
from mainboard.dispatch import SshTransport
from mainboard.dispatch import dispatcher as dispatch_module
from mainboard.dispatch.backends import HpcAiBackend, VastBackend
from mainboard.dispatch.lease import Lease
from mainboard.dispatch.rentals import Identity
from mainboard.dispatch.schedulers import HostUnreachable
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.durable import (
    Every,
    Settler,
    Settling,
    SystemdUser,
    Unsupported,
    locally,
    settler,
)
from mainboard.experiments import StudyLedger
from mainboard.experiments.identity import study_label
from mainboard.manifest import HostProfile

from .dispatch.backends.support import FakeTransport, refused
from .dispatch.support import RecordingScheduler, machine_with, plan
from .support import Lab

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


@pytest.mark.parametrize("verdict", ["running", "ok"])
def test_due_rentals_release_without_waiting_for_a_host_or_transfer(
    board: Board, monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    record = seed("13", target="vast", kind=Rented.name, verdict=verdict)
    record = record.model_copy(
        update={
            "lease": Lease(
                offer=Offer(provider="vast", gpu="RTX 5080", rate_usd_hr=1),
                release_by=datetime.now(UTC) - timedelta(seconds=1),
            ),
            "evidence": "pending",
        }
    )
    board.dispatcher.cache.record(record)
    Rented.replies = [{"success": True}]
    monkeypatch.setattr(board, "job", lambda *args, **kwargs: pytest.fail("host probed"))
    failed = board.monitor().expired()
    assert len(failed) == 1 and "deadline" in failed[0].reason
    current = board.dispatcher.cache.run("13")
    assert current.verdict == ("ok" if verdict == "ok" else "timeout")
    assert current.evidence == "unverified" and current.reported == current.verdict
    assert [call.get_method() for call in Rented.calls] == ["DELETE"]


def test_failed_deadline_deletion_stays_tracked_for_retry(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    record = seed("13", target="vast", kind=Rented.name, verdict="running")
    record = record.model_copy(
        update={
            "lease": Lease(
                offer=Offer(provider="vast", gpu="RTX 5080", rate_usd_hr=1),
                release_by=datetime.now(UTC) - timedelta(seconds=1),
            ),
        }
    )
    board.dispatcher.cache.record(record)
    Rented.replies = [{"success": False}]
    failed = board.monitor().expired()
    assert "release failed" in failed[0].reason
    assert board.dispatcher.cache.tracked()[0].handle == "13"
    assert board.dispatcher.cache.run("13").reported is None


def test_deadline_deletion_does_not_depend_on_a_healthy_receipt_stream(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-key")
    for handle in ("13", "14"):
        record = seed(handle, target="vast", kind=Rented.name, verdict="running")
        board.dispatcher.cache.record(
            record.model_copy(
                update={
                    "lease": Lease(
                        offer=Offer(provider="vast", gpu="RTX 5080", rate_usd_hr=1),
                        release_by=datetime.now(UTC) - timedelta(seconds=1),
                    ),
                }
            )
        )
    monitor = board.monitor()

    def broken(*args, **kwargs) -> None:
        raise OSError("receipt directory is unavailable")

    monkeypatch.setattr(monitor, "evidence", broken)
    Rented.replies = [{"success": True}]
    failed = monitor.expired()
    assert len(failed) == 2
    assert all("release confirmed" in row.reason for row in failed)
    assert all(record.verdict == "timeout" for record in board.dispatcher.cache.tracked())


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
        "a failed pull keeps settlement pending",
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
    report = board.monitor().once()
    if fetch_path:
        assert not report.finished
        assert "transfer failed" in report.failed[0].reason
        assert board.dispatcher.cache.run("3").reported is None
    else:
        assert report.finished[0].pulled_path is None
        assert board.dispatcher.cache.run("3").reported == "ok"


def test_a_failed_job_carries_a_network_free_reason(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("5")
    probing(board, monkeypatch, finishing(verdict="failed", exit_code=137))
    report = board.monitor().once()
    assert [(item.handle, item.target) for item in report.failed] == [("5", _HOST)]
    assert "memory" in report.failed[0].reason


def test_a_run_that_died_mid_campaign_still_brings_its_receipts_home(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that stages and renames every fragment keeps 399 when trial 400 of 500 kills it.

    The pull used to sit inside the clean branch, so exactly the runs whose partial evidence is
    least reproducible, an OOM at trial 400 or a metered rental hitting its cap, were the ones
    it was thrown away for. The exit code decides the verdict here and nothing else.
    """
    seed("25", fetch_path="results/partial")
    probing(board, monkeypatch, finishing(verdict="failed", exit_code=137))
    pulled: list[str] = []
    monkeypatch.setattr(board.dispatcher, "fetch", lambda handle, **kw: pulled.append(handle.id))
    [item] = board.monitor().once().failed
    assert pulled == ["25"]
    assert (item.pulled_path, "memory" in item.reason) == ("results/partial", True)


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
    Rented.replies = [
        {"result_url": "https://s3.example/logs/24.log"},
        "partial output\n",
        {"success": True},
        {"success": True},
    ]
    settled = board.verdicts().cancel("24")
    assert settled.trials[0].verdict == "cancelled"
    assert [(call.full_url, call.get_method()) for call in Rented.calls][-2:] == [
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
    assert board.dispatcher.cache.run("16").reported is None


def test_empty_successful_transfer_retains_receipt_referenced_evidence(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for Vast 50237293: rsync exited zero against the wrong empty root."""
    record = seed("31", name="empty-transfer", fetch_path="research/project/datasets/node")
    probing(board, monkeypatch, finishing())
    content = b"measured data"
    receipt = json.dumps(
        {
            "trial_receipt": {
                "case_id": "case",
                "outcome": "passed",
                "verdict": "validated",
                "artifacts": {
                    "table": {
                        "path": "datasets/node/evidence/table.parquet",
                        "sha256": sha256(content).hexdigest(),
                        "size": len(content),
                    }
                },
            }
        }
    )
    monkeypatch.setattr(Job, "transcript", lambda job: receipt)
    monkeypatch.setattr(board.dispatcher, "fetch", lambda *args, **kwargs: None)
    released = []
    monkeypatch.setattr(Job, "release", lambda job: released.append(job.handle.id))
    report = board.monitor().once()
    assert not report.finished and len(report.failed) == 1
    assert not released
    assert board.dispatcher.cache.run(record.handle).reported is None
    assert board.dispatcher.cache.run(record.handle).evidence == "pending"
    assert board.verdicts().handled(record.handle).code == 2
    table = board.root / "research/project/datasets/node/evidence/table.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(content)
    assert board.monitor().once().finished[0].handle == record.handle
    assert released == [record.handle]
    assert board.verdicts().handled(record.handle).code == 0
    assert board.dispatcher.cache.run(record.handle).evidence == "verified"
    assert (directory(board, "empty-transfer") / "receipts.ndjson").read_text().count(receipt) == 1


def test_failed_release_retries_without_refetching_destroyed_evidence(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("32", name="retry-release", fetch_path="results/run")
    probing(board, monkeypatch, finishing())
    pulled = []
    monkeypatch.setattr(board.dispatcher, "fetch", lambda *a, **kw: pulled.append(True))
    calls = []

    def release(job: Job) -> None:
        calls.append(job.handle.id)
        if len(calls) == 1:
            raise MissionError("provider temporarily unavailable")

    monkeypatch.setattr(Job, "release", release)
    assert not board.monitor().once().finished
    assert board.dispatcher.cache.run("32").reported is None
    assert board.monitor().once().finished[0].handle == "32"
    assert pulled == [True]
    assert calls == ["32", "32"]


def test_competing_monitor_does_not_read_a_stale_settlement_cursor(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("33")
    trips = probing(board, monkeypatch, finishing())
    with FileLock(board.dispatcher.cache.path.with_suffix(".settlement.lock")):
        skipped = board.monitor().once()
        assert not skipped.changed
        assert skipped.running is None
        assert skipped.model_dump()["running"] is None
    assert not trips
    assert board.monitor().once().finished[0].handle == "33"


def test_a_native_job_without_any_captured_receipt_cannot_settle_an_empty_transfer(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = seed("34", fetch_path="research/project/datasets/node")
    board.dispatcher.cache.record(
        record.model_copy(
            update={"script": "research/project/experiments/node/test_law.py::test_law"}
        )
    )
    probing(board, monkeypatch, finishing())
    monkeypatch.setattr(board.dispatcher, "fetch", lambda *a, **kw: None)
    monkeypatch.setattr(Job, "transcript", lambda job: "")
    monkeypatch.setattr(Job, "release", lambda job: pytest.fail("evidence was not delivered"))
    report = board.monitor().once()
    assert not report.finished and "no captured receipt" in report.failed[0].reason


@pytest.mark.parametrize("kind", ["ssh", "pbs"])
def test_queued_native_submission_cannot_verify_an_empty_transfer(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Keep the native identity through real rendering, submission, and settlement."""
    manifest = lab.root / "mainboard.toml"
    lab.write(
        "mainboard.toml",
        manifest.read_text() + f'\n[hosts.miyabi-g]\nkind = "{kind}"\nroot = "/repo"\n',
    )
    file = "research/project with spaces/experiments/node/test_law.py"
    lab.write(file, "def test_law():\n    raise RuntimeError('must not execute')\n")
    lab.write("research/project with spaces/experiments/node/node.md", "# Software control\n")
    lab.commit()
    board = Board(lab.root)
    dispatcher = board.dispatcher
    scheduler = RecordingScheduler()
    monkeypatch.setattr(dispatch_module, "pick", lambda profile: scheduler)
    monkeypatch.setattr(dispatch_module, "connection", lambda host: machine_with())
    monkeypatch.setattr(dispatcher, "rsync_up", lambda *args, **kwargs: [])
    spelling = "'research/project with spaces/experiments/node/test_law.py::test_law' -- -q"
    shipment = board.shipment(spelling, board.plan())
    shipment.admit(lab.root)
    handle = dispatcher.run(
        plan(host=_HOST, profile=HostProfile(kind=kind, root="/repo")),
        shipment,
        root="/repo",
        resources=Resources(walltime="00:01:00"),
        fetch="research/project with spaces/datasets/node",
        name="native-empty-transfer",
    )
    [(_, generated, args)] = [call for name, call in scheduler.calls if name == "submit"]
    assert isinstance(generated, str)
    assert generated.startswith(".mainboard-jobs/job-") and generated.endswith(".sh")
    assert (
        board.root / ".mainboard/dispatch/jobs" / Path(generated).name
    ).is_file() and args == ()
    probing(board, monkeypatch, finishing())
    monkeypatch.setattr(Job, "pull", lambda job: None)
    monkeypatch.setattr(Job, "transcript", lambda job: "")
    released: list[str] = []
    monkeypatch.setattr(Job, "release", lambda job: released.append(job.handle.id))
    report = board.monitor().once()
    assert not report.finished and "no captured receipt" in report.failed[0].reason
    assert not released
    record = dispatcher.cache.run(handle.id)
    assert record.script == shipment.spelling == f"'{file}::test_law' -q"
    assert record.args == ""
    assert record.evidence == "pending" and record.reported is None
    assert board.verdicts().handled(handle.id).code == 2


@pytest.mark.parametrize("fault", ["torn", "capture"])
def test_bad_native_evidence_does_not_prevent_another_job_from_settling(
    board: Board, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    seed("40", name="good")
    seed("41", name="bad", fetch_path="results/bad")
    probing(board, monkeypatch, finishing())
    monkeypatch.setattr(board.dispatcher, "fetch", lambda *a, **kw: None)

    def transcript(job: Job) -> str:
        if job.handle.id == "41":
            if fault == "capture":
                raise OSError("cannot read native output")
            return '{"trial_receipt": {"run": "torn"'
        return "finished"

    released = []
    monkeypatch.setattr(Job, "transcript", transcript)
    monkeypatch.setattr(Job, "release", lambda job: released.append(job.handle.id))
    report = board.monitor().once()
    assert [item.handle for item in report.finished] == ["40"]
    assert [item.handle for item in report.failed] == ["41"]
    assert released == ["40"]
    assert board.dispatcher.cache.run("41").evidence == "pending"


def test_a_reused_stream_cannot_borrow_another_handles_copied_checkpoint(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = seed("35", name="reused", verdict="ok", reported="ok")
    board.monitor().evidence(old, (), status="copied")
    seed("36", name="reused", fetch_path="results/new")
    probing(board, monkeypatch, finishing())
    pulled = []
    monkeypatch.setattr(board.dispatcher, "fetch", lambda handle, **kw: pulled.append(handle.id))
    assert board.monitor().once().finished[0].handle == "36"
    assert pulled == ["36"]


def test_a_rented_workspace_is_fetched_before_the_provider_destroys_it(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The monitor uses the rental's SSH endpoint before the irreversible release."""
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("17", target="rented", kind=Rented.name, fetch_path="results/run")
    replies = rented()
    Rented.replies = [
        *replies[:3],
        {
            "instances": {
                "actual_status": "running",
                "ssh_host": "rental.example",
                "ssh_port": 2222,
            }
        },
        *replies[3:],
        {"success": True},
    ]
    monkeypatch.setattr(
        "mainboard.board.identity", lambda declared: Identity(private="/keys/id", public="pub")
    )
    monkeypatch.setattr(
        "mainboard.board.connection", lambda host, policy: machine_with("/rental/projects")
    )
    transfers = []

    def fetch(host: str, **fields) -> None:
        assert not any(call.get_method() == "DELETE" for call in Rented.calls)
        transfers.append((host, fields))

    monkeypatch.setattr(board.dispatcher, "fetch_path", fetch)
    [item] = board.monitor().once().finished
    assert item.pulled_path == "results/run"
    [(host, fields)] = transfers
    assert host == "root@rental.example"
    assert fields["root"] == "/rental/projects"
    assert fields["ssh"].endpoint.port == 2222
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


class Systemd:
    """A stand-in user manager: it records every command and answers as a machine in one state.

    The one seam between this suite and systemd. The unit files are real, written into a
    temporary unit directory, and only the four queries a settler makes are answered here, so
    the whole install path runs without arming anything on the machine running the tests.

    active: what `show` reports for the timer's ActiveState.
    last_run: what `show` reports for its LastTriggerUSec, `n/a` for a timer that never ran.
    lingering: what `loginctl` reports for this user.
    enable: the status and output the arming command answers with.
    refusing: answer every query nonzero, the machine whose user manager will not talk.
    """

    def __init__(
        self,
        *,
        active: bool = True,
        last_run: str = "n/a",
        lingering: bool = True,
        enable: tuple[int, str] = (0, ""),
        refusing: bool = False,
    ) -> None:
        self.active = active
        self.last_run = last_run
        self.lingering = lingering
        self.enable = enable
        self.refusing = refusing
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str]) -> tuple[int, str]:
        self.calls.append(tuple(command))
        if self.refusing:
            return 1, "Failed to connect to bus"
        if command[0] == "loginctl":
            return 0, f"Linger={'yes' if self.lingering else 'no'}"
        if "show" in command:
            state = "active" if self.active else "inactive"
            return 0, f"ActiveState={state}\nLastTriggerUSec={self.last_run}\ntimers.target\n"
        if "enable" in command:
            return self.enable
        return 0, ""


class Recorded(Settler):
    """A settler recording what the verb asked of it, so the CLI seam is tested on its own."""

    def __init__(self, root: Path, answer: Settling) -> None:
        super().__init__(root)
        self.answer = answer
        self.installed: list[str] = []
        self.removed = 0

    def install(self, every: Every) -> Settling:
        self.installed.append(every.written)
        return self.answer

    def remove(self) -> Settling:
        self.removed += 1
        return self.answer

    def state(self) -> Settling:
        return self.answer


def systemd(root: Path, units: Path, manager: Systemd) -> SystemdUser:
    """A user-timer settler for `root` over a temporary unit directory and a stand-in manager."""
    return SystemdUser(root, units=units, shell=manager)


@pytest.mark.parametrize(
    ("written", "seconds"),
    [("20m", 1200), ("1h", 3600), ("90s", 90), ("  20m  ", 1200), ("0", 0), ("0m", 0)],
    ids=[
        "the period the campaign cron ran at",
        "an hour",
        "a plain number of seconds",
        "whatever whitespace a shell left around it",
        "a bare zero removes the pass",
        "so does a zero with a unit",
    ],
)
def test_a_period_is_read_the_way_systemd_writes_one(written: str, seconds: int) -> None:
    """The spelling is kept as written, since it is what the installed unit carries."""
    found = Every.parse(written)
    assert found.seconds == seconds
    assert found.written == written.strip()


@pytest.mark.parametrize(
    "written",
    ["20", "", "twenty", "20d", "1.5h", "20m30s"],
    ids=[
        "a number with no unit is ambiguous and is refused rather than guessed",
        "nothing at all",
        "a word",
        "a unit systemd would take but this one does not",
        "a fraction",
        "two units",
    ],
)
def test_a_period_nobody_can_read_names_the_spellings_instead_of_guessing(written: str) -> None:
    with pytest.raises(MissionError, match="written like 20m"):
        Every.parse(written)


def test_installing_the_pass_writes_both_units_and_arms_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable form of the sweep: the machine's own manager runs it, not a session.

    Everything the unit says is read back off the disk, since a unit file is the whole contract
    between this tool and systemd.
    """
    monkeypatch.setattr("mainboard.durable.which", lambda name: f"/usr/bin/{name}")
    root, units = tmp_path / "workspace", tmp_path / "units"
    root.mkdir()
    manager = Systemd(last_run="Thu 2026-09-04 09:20:31 JST", lingering=False)
    settler = systemd(root, units, manager)
    found = settler.install(Every.parse("20m"))
    service = settler.service.read_text(encoding="utf-8")
    timer = settler.timer.read_text(encoding="utf-8")
    log = root / ".mainboard" / "monitor.log"
    assert re.fullmatch(r"mainboard-monitor-[0-9a-f]{8}\.service", settler.service.name)
    assert f"WorkingDirectory={root}" in service
    assert "ExecStart=/usr/bin/mainboard monitor --json" in service
    assert f"StandardOutput=append:{log}" in service
    assert "OnUnitActiveSec=20m" in timer
    assert f"Unit={settler.service.name}" in timer
    assert "WantedBy=timers.target" in timer
    assert ("systemctl", "--user", "daemon-reload") in manager.calls
    assert ("systemctl", "--user", "enable", "--now", settler.timer.name) in manager.calls
    assert (found.installed, found.active, found.every) == (True, True, "20m")
    assert (found.last_run, found.log, found.root) == (
        "Thu 2026-09-04 09:20:31 JST",
        str(log),
        str(root),
    )
    assert found.fix == f"loginctl enable-linger {getuser()}"
    assert "a reboot stops it" in found.detail


def test_a_machine_with_no_periodic_pass_names_the_command_that_installs_one(
    tmp_path: Path,
) -> None:
    """Nothing is asked of the manager, since a unit that is not there cannot be armed."""
    manager = Systemd()
    found = systemd(tmp_path / "workspace", tmp_path / "units", manager).state()
    assert (found.installed, found.active, found.detail.startswith("no periodic pass")) == (
        False,
        False,
        True,
    )
    assert found.fix == "mainboard monitor --every 20m"
    assert manager.calls == []


@pytest.mark.parametrize(
    ("manager", "silent", "fix", "fragment"),
    [
        (
            Systemd(active=False),
            False,
            "systemctl --user enable --now {timer}",
            "installed but not armed",
        ),
        (Systemd(), True, "systemctl --user enable --now {timer}", ""),
        (Systemd(last_run="Thu 2026-09-04 09:20:31 JST"), False, "", "last run Thu 2026-09-04"),
        (Systemd(), False, "", "last run never"),
    ],
    ids=[
        "a timer nothing armed is named by the command that arms it",
        "a user manager that stops talking is a timer that is not running",
        "an armed and lingering timer says when it last swept",
        "and says so when it never has",
    ],
)
def test_an_installed_timer_is_judged_by_what_the_manager_says_about_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manager: Systemd,
    silent: bool,
    fix: str,
    fragment: str,
) -> None:
    """What is on disk is only half the answer; the other half is whether it is running.

    silent: the manager goes quiet after the install, the machine whose user bus is gone.
    """
    monkeypatch.setattr("mainboard.durable.which", lambda name: f"/usr/bin/{name}")
    root, units = tmp_path / "workspace", tmp_path / "units"
    root.mkdir()
    settler = systemd(root, units, manager)
    settler.install(Every.parse("20m"))
    manager.refusing = silent
    found = settler.state()
    assert found.installed
    assert found.fix == fix.format(timer=settler.timer.name)
    assert fragment in found.detail


@pytest.mark.parametrize(
    ("output", "fragment"),
    [("Failed to enable: unit is masked.\n", "unit is masked"), ("  \n", "it said nothing")],
    ids=["the manager's own last line is the refusal", "a manager that refuses silently"],
)
def test_a_manager_that_refuses_the_arming_refuses_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str, fragment: str
) -> None:
    """A pass a person believes is running and is not is worse than no pass at all."""
    monkeypatch.setattr("mainboard.durable.which", lambda name: f"/usr/bin/{name}")
    root = tmp_path / "workspace"
    root.mkdir()
    settler = systemd(root, tmp_path / "units", Systemd(enable=(1, output)))
    with pytest.raises(MissionError, match=fragment):
        settler.install(Every.parse("20m"))


def test_a_workstation_with_no_snapshot_on_path_has_nothing_for_a_timer_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mainboard.durable.which", lambda name: None)
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(MissionError, match="no mainboard on PATH"):
        systemd(root, tmp_path / "units", Systemd()).install(Every.parse("20m"))


def test_removing_the_pass_disarms_it_before_taking_its_units_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit systemd no longer has a file for is one it cannot be told to stop."""
    monkeypatch.setattr("mainboard.durable.which", lambda name: f"/usr/bin/{name}")
    root, units = tmp_path / "workspace", tmp_path / "units"
    root.mkdir()
    manager = Systemd()
    settler = systemd(root, units, manager)
    settler.install(Every.parse("20m"))
    timer = settler.timer.name
    found = settler.remove()
    assert not settler.timer.exists() and not settler.service.exists()
    assert ("systemctl", "--user", "disable", "--now", timer) in manager.calls
    assert not found.installed
    assert found.fix == "mainboard monitor --every 20m"
    assert settler.remove().installed is False


def test_a_timer_whose_service_file_somebody_deleted_still_says_what_is_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row exists to say what is wrong, so half an installation cannot take it down."""
    monkeypatch.setattr("mainboard.durable.which", lambda name: f"/usr/bin/{name}")
    root, units = tmp_path / "workspace", tmp_path / "units"
    root.mkdir()
    settler = systemd(root, units, Systemd())
    settler.install(Every.parse("20m"))
    settler.service.unlink()
    found = settler.state()
    assert found.installed and found.log == ""


@pytest.mark.parametrize(
    ("system", "found", "expected"),
    [
        ("Linux", "/usr/bin/systemctl", SystemdUser),
        ("Linux", None, Unsupported),
        ("Darwin", None, Unsupported),
        ("Windows", None, Unsupported),
    ],
    ids=[
        "a workstation running systemd gets the user timer",
        "a Linux box without systemd has no periodic runner here",
        "macOS",
        "Windows",
    ],
)
def test_the_machine_picks_the_periodic_runner_it_can_actually_drive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    found: str | None,
    expected: type[Settler],
) -> None:
    """The seam a second platform fills is one registration ahead of the null answer."""
    monkeypatch.setattr("mainboard.durable.platform.system", lambda: system)
    monkeypatch.setattr("mainboard.durable.which", lambda name: found)
    assert isinstance(settler(tmp_path), expected)


def test_a_platform_with_no_periodic_runner_refuses_in_one_sentence_naming_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing beats pretending: an outcome nobody settles must not look settled."""
    monkeypatch.setattr("mainboard.durable.platform.system", lambda: "Darwin")
    machine = Unsupported(tmp_path)
    for asked in (lambda: machine.install(Every.parse("20m")), machine.remove):
        with pytest.raises(MissionError, match="Darwin has no service manager mainboard"):
            asked()
    found = machine.state()
    assert (found.installed, found.active, found.fix) == (False, False, "")
    assert "Darwin has no periodic runner" in found.detail


@pytest.mark.parametrize(
    ("command", "deadline", "status", "fragment"),
    [
        ((sys.executable, "-c", "print('swept')"), 10.0, 0, "swept"),
        ((sys.executable, "-c", "import time; time.sleep(30)"), 0.5, 1, "did not answer"),
        (("mainboard-no-such-manager",), 10.0, 1, "is not installed here"),
    ],
    ids=[
        "what a manager said comes back whole",
        "a manager that hangs is one that said nothing",
        "so is a manager this machine never installed",
    ],
)
def test_asking_the_machine_answers_rather_than_raising(
    command: tuple[str, ...], deadline: float, status: int, fragment: str
) -> None:
    """Silence already means `no periodic pass runs`, so neither refusal is worth an exception."""
    code, output = locally(command, deadline)
    assert code == status
    assert fragment in output


@pytest.mark.parametrize(
    ("every", "fix", "lines"),
    [
        (
            "20m",
            "loginctl enable-linger pedro",
            ("mainboard: the timer sweeps every 20m", "mainboard: run `loginctl enable-linger"),
        ),
        ("0", "", ("mainboard: nothing periodic runs here",)),
    ],
    ids=["installing prints what it installed and the step a reboot needs", "removing says so"],
)
def test_the_monitor_verb_hands_the_pass_to_this_machine_and_takes_it_back(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    every: str,
    fix: str,
    lines: tuple[str, ...],
) -> None:
    """`--every` is the whole of what the verb owns: the period reaches the machine's manager."""
    detail = "the timer sweeps every 20m" if fix else "nothing periodic runs here"
    recorder = Recorded(workspace, Settling(detail=detail, fix=fix))
    monkeypatch.setattr("mainboard.durable.settler", lambda root: recorder)
    with pytest.raises(SystemExit, match="0"):
        build(workspace)(["monitor", "--every", every])
    out = capsys.readouterr().out
    assert all(line in out for line in lines)
    assert out.count("mainboard: ") == len(lines)
    assert recorder.installed == (["20m"] if fix else [])
    assert recorder.removed == (0 if fix else 1)
