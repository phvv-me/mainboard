from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from mainboard import Fleet
from mainboard.dispatch import Handle, Verdict
from mainboard.dispatch.schedulers.base import POLL_SECONDS
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.experiments import Progress, Study, StudyLedger
from mainboard.experiments.identity import study_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Unpack

    from mainboard.experiments.fleet import ResourceOverrides


@dataclass
class FakeJob:
    """A `Job`-like stub: `Fleet` only ever reads `.handle` off what `submit` returns."""

    handle: Handle


class FakeBoundBoard:
    """A `Board.on(host)`-like stub recording every `submit` call onto its parent `FakeBoard`."""

    def __init__(self, board: FakeBoard, host: str) -> None:
        self.board = board
        self.host = host

    def submit(
        self, command: str, *, name: str = "", **kwargs: Unpack[ResourceOverrides]
    ) -> FakeJob:
        self.board.calls.append((self.host, command, name, kwargs))
        self.board.counter += 1
        handle = Handle(id=str(self.board.counter), host=self.host, root="/work/x", kind="pbs")
        return FakeJob(handle)


class FakeDispatcher:
    """A `Dispatcher`-like stub: `await_many` resolves handles from a pre-seeded verdict map.

    Carries a real, tmp-path-backed `cache` (exactly `Dispatcher.cache`'s shape), since
    `Fleet.progress` reads it to join a study's ledger against dispatch's own resolved verdicts.
    """

    def __init__(self, cache: Cache) -> None:
        self.cache = cache
        self.verdicts: dict[Handle, Verdict] = {}
        self.awaited: list[Handle] = []

    def await_many(
        self, handles: Sequence[Handle], *, interval: float = POLL_SECONDS
    ) -> dict[Handle, Verdict]:
        self.awaited.extend(handles)
        return {handle: self.verdicts[handle] for handle in handles}


class FakeBoard:
    """A `Board`-like stub: real `root`, fake `on`/`dispatcher`, no ssh or subprocess ever runs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, str, str, dict[str, object]]] = []
        self.counter = 0
        self.dispatcher = FakeDispatcher(Cache(root / "dispatch.sqlite"))

    def on(self, host: str) -> FakeBoundBoard:
        return FakeBoundBoard(self, host)


def make_run(name: str, *, handle: str, verdict: str | None = None) -> RunRecord:
    """A dispatch `RunRecord` labeled `name`, resolved to `verdict` when given."""
    return RunRecord(
        handle=handle,
        target="gold",
        kind="pbs",
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at="t0",
        name=name,
        verdict=verdict,
    )


@pytest.fixture
def board(tmp_path: Path) -> FakeBoard:
    return FakeBoard(tmp_path)


@pytest.fixture
def study() -> Study:
    return Study.create("joint-search", config_space={"bits": [1, 2]}, git_sha="abc123")


def test_submit_all_dispatches_every_pair_labeled_with_the_study(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    commands = [("gold", "cmd1"), ("miyabi-g", "cmd2")]
    jobs = fleet.submit_all(commands, study=study, mem_gb=64)
    assert len(jobs) == 2
    label = f"study:{study.study_id}"
    assert board.calls == [
        ("gold", "cmd1", label, {"mem_gb": 64}),
        ("miyabi-g", "cmd2", label, {"mem_gb": 64}),
    ]


def test_submit_all_records_a_submitted_event_per_trial(board: FakeBoard, study: Study) -> None:
    fleet = Fleet(board)
    jobs = fleet.submit_all([("gold", "cmd1"), ("gold", "cmd2")], study=study)
    ledger = StudyLedger(board.root, study.study_id)
    assert ledger.statuses() == {job.handle.id: "submitted" for job in jobs}


def test_submit_all_records_one_created_event_on_the_ledgers_first_touch(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    fleet.submit_all([("gold", "cmd1")], study=study)
    fleet.submit_all([("gold", "cmd2")], study=study)
    ledger = StudyLedger(board.root, study.study_id)
    created = [event for event in ledger.events() if event.kind == "created"]
    assert [event.name for event in created] == [study.name]


def test_statuses_reads_the_studys_ledger(board: FakeBoard, study: Study) -> None:
    fleet = Fleet(board)
    jobs = fleet.submit_all([("gold", "cmd1")], study=study)
    assert fleet.statuses(study) == {jobs[0].handle.id: "submitted"}


def test_progress_lets_a_resolved_dispatch_verdict_outrank_the_ledger(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    [job] = fleet.submit_all([("gold", "cmd1")], study=study)
    label = f"study:{study.study_id}"
    board.dispatcher.cache.record(make_run(label, handle=job.handle.id, verdict="ok"))
    assert fleet.progress(study) == Progress(submitted=1, running=0, ok=1, failed=0)


def test_wait_all_delegates_to_await_many_and_records_each_verdict(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    jobs = fleet.submit_all([("gold", "cmd1"), ("gold", "cmd2")], study=study)
    board.dispatcher.verdicts = {
        jobs[0].handle: Verdict(verdict="ok"),
        jobs[1].handle: Verdict(verdict="failed", reason="oom"),
    }
    verdicts = fleet.wait_all(jobs)
    assert verdicts == board.dispatcher.verdicts
    assert board.dispatcher.awaited == [job.handle for job in jobs]
    assert fleet.statuses(study) == {jobs[0].handle.id: "ok", jobs[1].handle.id: "failed"}


def test_wait_all_skips_ledger_recording_for_a_handle_it_never_submitted(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    foreign = Handle(id="ghost", host="gold", root="/work/x", kind="pbs")
    board.dispatcher.verdicts = {foreign: Verdict(verdict="ok")}
    verdicts = fleet.wait_all([FakeJob(foreign)])
    assert verdicts == {foreign: Verdict(verdict="ok")}
    assert StudyLedger(board.root, study.study_id).events() == []


def test_resubmit_redispatches_the_original_command_at_the_new_attempt(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    [job] = fleet.submit_all([("gold", "cmd1")], study=study)
    [resubmitted] = fleet.resubmit(study, [job.handle], attempt=2)
    label = f"study:{study.study_id}"
    assert board.calls[-1] == ("gold", "cmd1", label, {"attempt": 2})
    assert resubmitted.handle != job.handle


def test_resubmit_tracks_the_new_handle_for_a_further_resubmit(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    [job] = fleet.submit_all([("gold", "cmd1")], study=study)
    [resubmitted] = fleet.resubmit(study, [job.handle], attempt=2)
    [twice] = fleet.resubmit(study, [resubmitted.handle], attempt=3)
    assert board.calls[-1] == ("gold", "cmd1", f"study:{study.study_id}", {"attempt": 3})
    assert twice.handle not in (job.handle, resubmitted.handle)


def test_resubmit_raises_for_a_handle_this_fleet_never_submitted(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    foreign = Handle(id="ghost", host="gold", root="/work/x", kind="pbs")
    with pytest.raises(KeyError):
        fleet.resubmit(study, [foreign], attempt=2)


def test_overview_lists_every_study_found_under_the_board_root(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    fleet.submit_all([("gold", "cmd1")], study=study)
    [summary] = Fleet.overview(board.root, board.dispatcher.cache)
    assert summary.study_id == study.study_id
    assert summary.name == study.name
    assert summary.counts == {"submitted": 1}


def test_settle_records_a_verdict_for_a_handle_this_fleet_never_submitted(
    board: FakeBoard, study: Study
) -> None:
    handle = Handle(id="77", host="gold", root="/work/x", kind="pbs")
    board.dispatcher.cache.record(make_run(study_label(study.study_id), handle=handle.id))
    Fleet(board).settle({handle: Verdict(verdict="ok")})
    assert StudyLedger(board.root, study.study_id).statuses() == {"77": "ok"}


def test_settle_ignores_a_handle_belonging_to_no_study(board: FakeBoard) -> None:
    handle = Handle(id="88", host="gold", root="/work/x", kind="pbs")
    board.dispatcher.cache.record(make_run("ad-hoc", handle=handle.id))
    Fleet(board).settle({handle: Verdict(verdict="ok")})
    assert not list(board.root.glob("**/studies/*.jsonl"))


def test_owner_prefers_this_fleets_own_record_over_the_cache(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    [job] = fleet.submit_all([("gold", "cmd1")], study=study)
    assert fleet.owner(job.handle) == study.study_id


def test_owner_is_empty_for_a_handle_dispatch_never_recorded(board: FakeBoard) -> None:
    handle = Handle(id="ghost", host="gold", root="/work/x", kind="pbs")
    assert not Fleet(board).owner(handle)
