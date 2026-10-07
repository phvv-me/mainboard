"""Inspecting one job collects its evidence without retrying unrelated unfinished deliveries."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mainboard.batch.receipts import Topic, publish
from mainboard.board import Board, Job
from mainboard.costs.catalog import Offer
from mainboard.dispatch.backends.vast import VastBackend
from mainboard.dispatch.dispatcher import Dispatcher, Handle
from mainboard.dispatch.lease import Lease
from mainboard.dispatch.shared import now
from mainboard.dispatch.state import RunRecord
from mainboard.dispatch.vocabulary import JobState
from mainboard.monitor import Monitor
from mainboard.verdicts import StreamVerdict, Verdicts


@pytest.fixture
def board(workspace: Path) -> Board:
    (workspace / "mb.toml").write_text(
        '[workspace]\nname = "it"\n'
        '[hosts.box]\nkind = "ssh"\nroot = "/srv/jobs"\n'
        '[hosts.other]\nkind = "ssh"\nroot = "/srv/jobs"\n',
        encoding="utf-8",
    )
    found = Board(workspace)
    for handle, target, verdict in (
        ("7", "box", "running"),
        ("8", "box", "ok"),
        ("held", "box", "held"),
        ("7", "other", "ok"),
    ):
        found.dispatcher.cache.record(
            RunRecord(
                handle=handle,
                target=target,
                kind="ssh",
                script="python train.py",
                args="",
                submitted_at=now(),
                fetch_path="outputs",
                verdict=verdict,
            )
        )
    return found


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    made: list[tuple[str, str, str]] = []

    def states(self: Dispatcher, handles: list[Handle]) -> dict[str, JobState]:
        made.extend(("poll", handle.host, handle.id) for handle in handles)
        return {
            handle.id: JobState(handle=handle.id, state="completed", exit_code=0, verdict="ok")
            for handle in handles
        }

    def transcript(self: Job) -> str:
        made.append(("transcript", self.handle.host, self.handle.id))
        return ""

    def held(self: Monitor, record: RunRecord) -> None:
        made.append(("held", record.target, record.handle))

    monkeypatch.setattr(Dispatcher, "states", states)
    monkeypatch.setattr(Job, "transcript", transcript)
    monkeypatch.setattr(
        Job, "pull", lambda self: made.append(("pull", self.handle.host, self.handle.id))
    )
    monkeypatch.setattr(Monitor, "asked", held)
    return made


@pytest.mark.parametrize("wait", [False, True])
def test_only_the_requested_submission_is_settled(board: Board, calls, wait: bool) -> None:
    (board.root / "7").mkdir()
    verdicts = board.verdicts()
    assert verdicts.of("7", host="box").code == 2
    assert calls == []
    result = (
        verdicts.wait("7", host="box", timeout=5, stall=0)
        if wait
        else verdicts.refresh("7", host="box")
    )
    assert result.code == 0
    assert calls == [("poll", "box", "7"), ("pull", "box", "7"), ("transcript", "box", "7")]
    cache = board.dispatcher.cache
    assert cache.run("7", "box").reported == "ok"
    assert cache.run("8", "box").reported is None
    assert cache.run("7", "other").reported is None
    assert cache.run("held", "box").verdict == "held"
    assert verdicts.refresh("7", host="box").code == 0
    assert len(calls) == 3


@pytest.mark.parametrize("wait", [False, True])
def test_a_batch_refreshes_all_its_submissions_only(board: Board, calls, wait: bool) -> None:
    (board.root / "cohort").mkdir()
    cache = board.dispatcher.cache
    for handle in ("7", "9"):
        record = cache.run("7", "box").model_copy(
            update={"handle": handle, "name": f"batch:cohort/cell-{handle}"}
        )
        cache.record(record)
        publish(
            board.receipts("cohort"),
            "cohort",
            Topic.SUBMITTED,
            job=f"cell-{handle}",
            data={"handle": handle, "target": "box", "kind": "ssh"},
        )
    verdicts = board.verdicts()
    result = verdicts.wait("cohort", timeout=5, stall=0) if wait else verdicts.refresh("cohort")
    assert result.code == 0
    assert {(host, handle) for _, host, handle in calls} == {("box", "7"), ("box", "9")}
    assert all(cache.run(handle, "box").reported == "ok" for handle in ("7", "9"))
    assert cache.run("8", "box").reported is None


def test_local_receipt_files_and_stores_stay_offline(
    board: Board, calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(self: Monitor, **kwargs) -> None:
        pytest.fail("a local receipt read attempted settlement")

    monkeypatch.setattr(Monitor, "once", unexpected)
    path = board.root / "receipts.jsonl"
    path.write_text("", encoding="utf-8")
    assert board.verdicts().refresh(str(path)).trials == ()
    assert board.verdicts().wait(str(path)).trials == ()
    stored = StreamVerdict(stream="stored", trials=())
    monkeypatch.setattr(Verdicts, "stored", lambda *args, **kwargs: stored)
    directory = board.root / "receipt-directory"
    directory.mkdir()
    assert board.verdicts().refresh(str(directory)) == stored
    assert board.verdicts().wait(str(directory), run="selected") == stored
    assert calls == []


def test_a_scoped_refresh_still_releases_an_unrelated_expired_rental(
    board: Board, calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = board.dispatcher.cache
    cache.record(
        RunRecord(
            handle="rented",
            target="vast",
            kind="vast",
            script="python train.py",
            args="",
            submitted_at=now(),
            verdict="running",
            lease=Lease(
                offer=Offer(provider="vast", gpu="test", rate_usd_hr=1),
                release_by=datetime.now(UTC) - timedelta(seconds=1),
            ),
        )
    )
    monkeypatch.setattr(
        VastBackend, "cancel", lambda self, handle: calls.append(("cancel", "vast", handle))
    )
    assert board.verdicts().refresh("7", host="box").code == 0
    assert calls[0] == ("cancel", "vast", "rented")
    ended = cache.run("rented", "vast")
    assert ended.verdict == ended.reported == "timeout"
    assert ended.evidence == "unverified"


def test_a_reused_handle_is_not_selected_by_an_old_submission(
    board: Board, calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Monitor.once

    def replaced(self: Monitor, **kwargs):
        cache = self.cache
        previous = cache.run("7", "box")
        cache.forget(previous)
        cache.record(previous.model_copy(update={"submitted_at": "2099-01-01T00:00:00+00:00"}))
        return original(self, **kwargs)

    monkeypatch.setattr(Monitor, "once", replaced)
    assert board.verdicts().refresh("7", host="box").code == 2
    assert calls == []


@pytest.mark.parametrize("wait", [False, True])
def test_a_selected_quota_hold_follows_its_new_handle(
    board: Board, calls, monkeypatch: pytest.MonkeyPatch, wait: bool
) -> None:
    def resumed(self: Monitor, record: RunRecord) -> Job:
        replacement = record.model_copy(update={"handle": "resumed", "verdict": "queued"})
        self.cache.record(replacement)
        return self.board.job(replacement.handle, host=replacement.target)

    monkeypatch.setattr(Monitor, "asked", resumed)
    verdicts = board.verdicts()
    result = (
        verdicts.wait("held", host="box", timeout=10, interval=0, stall=0)
        if wait
        else verdicts.refresh("held", host="box")
    )
    assert result.code == (0 if wait else 2)
    assert {trial.handle for trial in result.trials} == {"resumed"}
    assert all(handle == "resumed" for _, _, handle in calls)
    assert board.dispatcher.cache.run("7", "box").reported is None
