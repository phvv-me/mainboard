import logging
from itertools import islice
from typing import TYPE_CHECKING, ClassVar

import pytest
from plumbum.commands.processes import ProcessExecutionError

from mainboard import Board, Job
from mainboard.dispatch.backends import HpcAiBackend, VastBackend
from mainboard.dispatch.schedulers import HostUnreachable, JobState
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.experiments import StudyLedger
from mainboard.experiments.identity import study_label

from .dispatch.backends.conftest import FakeTransport, refused

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from urllib.request import Request

    from mainboard.dispatch import Handle

    from .dispatch.backends.conftest import Reply

_HOST = "miyabi-g"
_STUDY = "ec15c1b1e073"


@pytest.fixture
def board(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Board:
    """A real board over the fixture manifest, its dispatch cache isolated in the workspace."""
    monkeypatch.chdir(workspace)
    return Board(workspace)


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
    """The replies one Vast post-mortem takes, the instance row, its log url, and the log."""
    return [
        {"instances": {"id": 7, "actual_status": status}},
        {"result_url": "https://s3.example/logs/7.log"},
        f"training done\nmainboard-exit:{exit_code}\n",
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


def probing(board: Board, monkeypatch: pytest.MonkeyPatch, answer: Callable[[Handle], JobState]):
    """Pin the board's scheduler probe to `answer`, returning the handles it was asked about."""
    asked: list[str] = []

    def state(handle: Handle) -> JobState:
        asked.append(handle.id)
        return answer(handle)

    monkeypatch.setattr(board.dispatcher, "state", state)
    return asked


def finishing(verdict: str = "ok", exit_code: int | None = 0) -> Callable[[Handle], JobState]:
    """A probe answer settling every handle on the same terminal verdict."""
    return lambda handle: JobState(
        handle=handle.id, state="F", exit_code=exit_code, verdict=verdict
    )


def test_a_cache_with_nothing_tracked_reports_no_change(board: Board) -> None:
    report = board.monitor().once()
    assert report.running == 0
    assert not report.changed
    assert report.finished == [] and report.failed == [] and report.unreachable_hosts == []


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
    asked = probing(board, monkeypatch, finishing())
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
    assert asked == ["2"]  # the settled run is never probed a second time


def test_a_finished_job_without_a_results_path_pulls_nothing(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("3")
    probing(board, monkeypatch, finishing())
    [item] = board.monitor().once().finished
    assert item.pulled_path is None


def test_a_pull_that_fails_leaves_the_job_finished_without_a_path(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("4", fetch_path="results/run")
    probing(board, monkeypatch, finishing())

    def explode(handle: Handle, **kw: object) -> None:
        raise ProcessExecutionError(["rsync"], 23, "", "no such file")

    monkeypatch.setattr(board.dispatcher, "fetch", explode)
    [item] = board.monitor().once().finished
    assert item.pulled_path is None
    assert board.dispatcher.cache.run("4").reported == "ok"


def test_a_failed_job_carries_a_network_free_reason(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("5")
    probing(board, monkeypatch, finishing(verdict="failed", exit_code=137))
    report = board.monitor().once()
    assert [(item.handle, item.target) for item in report.failed] == [("5", _HOST)]
    assert "memory" in report.failed[0].reason


def test_a_terminal_verdict_already_cached_is_harvested_without_probing(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("6", verdict="ok")
    asked = probing(board, monkeypatch, finishing())
    report = board.monitor().once()
    assert [item.handle for item in report.finished] == ["6"]
    assert asked == []


def test_a_settled_run_the_sweep_already_reported_is_not_tracked_again(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed("7", verdict="ok", reported="ok")
    asked = probing(board, monkeypatch, finishing())
    report = board.monitor().once()
    assert not report.changed and asked == []


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

    asked = probing(board, monkeypatch, unreachable)
    report = board.monitor().once()
    assert [(host.host, "timed out" in host.reason) for host in report.unreachable_hosts] == [
        (_HOST, True)
    ]
    assert report.running == 0 and not report.changed
    assert len(asked) == 1  # the second job on the same host costs no further probe
    assert len(board.dispatcher.cache.tracked()) == 2


def test_a_host_the_manifest_can_no_longer_resolve_is_reported_not_raised(board: Board) -> None:
    seed("11", target="gold")  # the fixture's gold profile declares no root
    [host] = board.monitor().once().unreachable_hosts
    assert host.host == "gold"
    assert "root" in host.reason


# --- releasing what a settled run still holds ---


def test_a_finished_vast_rental_is_settled_and_then_cancelled(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vast restarts the exited container until someone cancels, so the sweep is what cancels."""
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    seed("13", target="rented", kind=Rented.name)
    Rented.replies = [*rented(), {"success": True}]
    report = board.monitor().once()
    assert [item.handle for item in report.finished] == ["13"]
    assert Rented.calls[-1].get_method() == "DELETE"
    assert Rented.calls[-1].full_url == "https://console.vast.ai/api/v0/instances/13/"
    assert board.dispatcher.cache.run("13").reported == "ok"


def test_a_finished_hpc_ai_instance_is_settled_and_then_terminated(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An HPC-AI instance runs until it is terminated, whatever its command already did."""
    monkeypatch.setenv("HPCAI_API_KEY", "key-123")
    seed("14", target="rented", kind=Instance.name)
    Instance.replies = [listed("14", "Stopped"), {}, {}]
    report = board.monitor().once()
    assert [item.handle for item in report.finished] == ["14"]
    assert [call.full_url for call in Instance.calls[-2:]] == [
        "https://www.hpc-ai.com/api/instance/stop",
        "https://www.hpc-ai.com/api/instance/terminate",
    ]


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
