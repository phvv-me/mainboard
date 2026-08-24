from typing import TYPE_CHECKING

import pytest

from mainboard import Fleet
from mainboard.dispatch import Handle, Verdict
from mainboard.experiments import Progress, StudyLedger
from mainboard.experiments.identity import study_label

from .conftest import FakeBoard, make_run

if TYPE_CHECKING:
    from mainboard.experiments import Study


def stray(handle: str) -> Handle:
    """A dispatch handle no fleet in this test ever submitted."""
    return Handle(id=handle, host="gold", root="/work/x", kind="pbs")


def test_submit_all_dispatches_every_pair_under_the_studys_label_and_ledgers_each_trial(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    label = study_label(study.study_id)
    jobs = fleet.submit_all([("gold", "cmd1"), ("miyabi-g", "cmd2")], study=study, mem_gb=64)
    assert board.calls == [
        ("gold", "cmd1", label, {"mem_gb": 64}),
        ("miyabi-g", "cmd2", label, {"mem_gb": 64}),
    ]
    assert fleet.statuses(study) == {job.handle.id: "submitted" for job in jobs}
    fleet.submit_all([("gold", "cmd3")], study=study)
    events = StudyLedger(board.root, study.study_id).events()
    assert [event.name for event in events if event.kind == "created"] == [study.name]


def test_wait_all_awaits_every_handle_and_records_the_verdict_in_the_owning_ledger(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    jobs = fleet.submit_all([("gold", "cmd1"), ("gold", "cmd2")], study=study)
    board.dispatcher.verdicts = {
        jobs[0].handle: Verdict(verdict="ok"),
        jobs[1].handle: Verdict(verdict="failed", reason="oom"),
    }
    assert fleet.wait_all(jobs) == board.dispatcher.verdicts
    assert board.dispatcher.awaited == [job.handle for job in jobs]
    assert fleet.statuses(study) == {jobs[0].handle.id: "ok", jobs[1].handle.id: "failed"}


def test_settle_records_each_verdict_in_the_ledger_of_the_study_that_owns_its_handle(
    board: FakeBoard, study: Study
) -> None:
    owned, unowned = stray("77"), stray("88")
    board.dispatcher.cache.record(make_run(study_label(study.study_id), handle=owned.id))
    board.dispatcher.cache.record(make_run("ad-hoc", handle=unowned.id))
    Fleet(board).settle({owned: Verdict(verdict="ok"), unowned: Verdict(verdict="ok")})
    assert StudyLedger(board.root, study.study_id).statuses() == {"77": "ok"}
    assert [path.stem for path in board.root.glob("**/studies/*.jsonl")] == [study.study_id]


def test_owner_prefers_this_fleets_own_record_then_the_dispatch_label_then_nothing(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    [job] = fleet.submit_all([("gold", "cmd1")], study=study)
    assert fleet.owner(job.handle) == study.study_id
    recovered = stray("77")
    board.dispatcher.cache.record(make_run(study_label(study.study_id), handle=recovered.id))
    assert fleet.owner(recovered) == study.study_id
    assert not fleet.owner(stray("ghost"))


def test_resubmit_reissues_the_original_command_at_the_new_attempt_and_tracks_the_new_handle(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    label = study_label(study.study_id)
    [job] = fleet.submit_all([("gold", "cmd1")], study=study)
    [retried] = fleet.resubmit(study, [job.handle], attempt=2)
    assert board.calls[-1] == ("gold", "cmd1", label, {"attempt": 2})
    [again] = fleet.resubmit(study, [retried.handle], attempt=3)
    assert board.calls[-1] == ("gold", "cmd1", label, {"attempt": 3})
    assert len({job.handle, retried.handle, again.handle}) == 3
    with pytest.raises(KeyError):
        fleet.resubmit(study, [job.handle], attempt=4)


def test_a_fleet_reads_a_studys_live_progress_and_lists_every_study_under_the_board_root(
    board: FakeBoard, study: Study
) -> None:
    fleet = Fleet(board)
    [job] = fleet.submit_all([("gold", "cmd1")], study=study)
    assert fleet.progress(study) == Progress(submitted=1, running=1)
    label = study_label(study.study_id)
    board.dispatcher.cache.record(make_run(label, handle=job.handle.id, verdict="ok"))
    assert fleet.progress(study) == Progress(submitted=1, ok=1)
    [summary] = Fleet.overview(board.root, board.dispatcher.cache)
    assert (summary.study_id, summary.name, summary.counts) == (
        study.study_id,
        study.name,
        {"ok": 1},
    )
