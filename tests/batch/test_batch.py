from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from mainboard import Board, MissionError
from mainboard.batch import Batch, BatchStatus, Topic, Watch
from mainboard.batch.receipts import publish
from mainboard.batch.watch import _epoch
from mainboard.dispatch import Handle, HostUnreachable
from mainboard.dispatch.state import Failed, Finished, MonitorReport, RunRecord
from mainboard.monitor import Monitor

from .support import Recorder, published, spec

if TYPE_CHECKING:
    from mainboard.batch import BatchSpec

_TWO = ({"target": "gold", "command": "python -m a"}, {"target": "miyabi-g", "command": "b"})


def dispatching(board: Board, monkeypatch: pytest.MonkeyPatch, *handles: str) -> list[str]:
    """Answer every submit with the next handle, recording the commands that asked."""
    asked: list[str] = []
    queue = list(handles)

    def submit(self: Board, command: str, **options: str | int | float | bool) -> SimpleNamespace:
        asked.append(command)
        return SimpleNamespace(
            handle=Handle(id=queue.pop(0), host=self.host, root="/repo", kind="ssh")
        )

    monkeypatch.setattr(Board, "submit", submit)
    return asked


def refusing(monkeypatch: pytest.MonkeyPatch, refusal: BaseException) -> None:
    """Answer every submit with `refusal`, the way a target that will not take a job does."""

    def submit(self: Board, command: str, **options: str | int | float | bool) -> SimpleNamespace:
        raise refusal

    monkeypatch.setattr(Board, "submit", submit)


def sweeping(board: Board, monkeypatch: pytest.MonkeyPatch, report: MonitorReport) -> list[int]:
    """Answer the durable sweep with `report`, counting how often the watch drove it."""
    passes: list[int] = []

    def once(self: Monitor) -> MonitorReport:
        passes.append(1)
        return report

    del board
    monkeypatch.setattr(Monitor, "once", once)
    return passes


def recorded(board: Board, handle: str, *, target: str, verdict: str | None = None) -> RunRecord:
    """One dispatched run in the registry, as a submit would have left it."""
    run = RunRecord(
        handle=handle,
        target=target,
        kind="ssh",
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at="2026-08-20T00:00:00+00:00",
        state="Done" if verdict else "Running",
        verdict=verdict,
    )
    board.dispatcher.cache.record(run)
    return run


def batched(board: Board, bus: Recorder, declared: BatchSpec | None = None) -> Batch:
    """A batch over `board` publishing into `bus`."""
    return Batch(board, declared or spec(*_TWO), bus=bus)


def watching(board: Board, batch: Batch, bus: Recorder) -> Watch:
    """The live view over `batch`, reading and writing the same receipts it published."""
    return Watch(board, batch.id, bus=bus)


def test_the_first_verb_to_touch_a_batch_announces_it_and_the_next_one_does_not(
    lab: Board, bus: Recorder
) -> None:
    """The log is the state, so opening is written once however the flow is entered."""
    batch = batched(lab, bus)
    batch.prepare()
    batch.prepare()
    [opened] = published(bus, Topic.OPENED)
    assert opened.data == {
        "name": "smoke",
        "jobs": ["gold-1", "miyabi-g-2"],
        "root": str(lab.root),
    }
    assert opened.job == ""


def test_preparing_publishes_what_each_job_must_ship(lab: Board, bus: Recorder) -> None:
    (lab.root / "packages").mkdir()
    (lab.root / "packages" / "train.py").write_text("print(1)\n")
    measured = batched(lab, bus).prepare()
    assert [transfer.job for transfer in measured] == ["gold-1", "miyabi-g-2"]
    assert [event.data["files"] for event in published(bus, Topic.PREPARED)] == [1, 1]


def test_pricing_reads_back_what_preparing_measured_instead_of_measuring_again(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipts are the hand-off between the two verbs, which is what a broker would carry."""
    batch = batched(lab, bus, spec(_TWO[0]))
    batch.prepare()
    monkeypatch.setattr(
        "mainboard.batch.runner.Transfer.set_for", lambda self, job: pytest.fail("re-measured")
    )
    table = batch.estimate()
    assert [row.job for row in table.jobs] == ["gold-1"]
    assert [event.job for event in published(bus, Topic.ESTIMATED)] == ["gold-1"]


def test_pricing_a_job_nobody_prepared_measures_it_rather_than_leaving_the_row_blank(
    lab: Board, bus: Recorder
) -> None:
    (lab.root / "packages").mkdir()
    (lab.root / "packages" / "train.py").write_text("print(1)\n")
    [row] = batched(lab, bus, spec(_TWO[0])).estimate().jobs
    assert row.wire_bytes > 0
    assert published(bus, Topic.PREPARED) == []


def test_running_dispatches_every_job_to_its_own_target(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = batched(lab, bus)
    asked = dispatching(lab, monkeypatch, "77", "78")
    dispatched = batch.run()
    assert asked == ["python -m a", "b"]
    assert [(entry.job, entry.target, entry.handle) for entry in dispatched] == [
        ("gold-1", "gold", "77"),
        ("miyabi-g-2", "miyabi-g", "78"),
    ]
    assert [event.data["handle"] for event in published(bus, Topic.SUBMITTED)] == ["77", "78"]
    assert batch.labelling("gold-1") == f"batch:{batch.id}/gold-1"


@pytest.mark.parametrize(
    "refusal",
    [MissionError("host 'gold' declares no root"), HostUnreachable("connection timed out")],
    ids=["a target the manifest cannot resolve", "a machine that is asleep"],
)
def test_a_target_that_refuses_one_job_is_a_row_rather_than_the_end_of_the_batch(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch, refusal: BaseException
) -> None:
    """The other four jobs of a five-job sweep are still worth running."""
    refusing(monkeypatch, refusal)
    [entry] = batched(lab, bus, spec(_TWO[0])).run()
    assert (entry.handle, entry.job) == ("", "gold-1")
    assert str(refusal) in entry.reason
    [event] = published(bus, Topic.REFUSED)
    assert event.data["target"] == "gold"


def test_a_batch_keeps_its_receipts_in_its_own_directory(lab: Board) -> None:
    batch = Batch(lab, spec(_TWO[0]))
    assert batch.dir == lab.root / ".mainboard" / "batches" / batch.id
    batch.open()
    assert batch.bus.path.is_file()
    assert Watch(lab, batch.id).bus.path == batch.bus.path


def test_a_pass_reports_every_target_in_one_view_and_publishes_only_what_moved(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state that has not changed says nothing, which is what keeps the log a change log."""
    batch = batched(lab, bus)
    dispatching(lab, monkeypatch, "77", "78")
    batch.run()
    recorded(lab, "77", target="gold")
    recorded(lab, "78", target="miyabi-g")
    sweeping(lab, monkeypatch, MonitorReport(running=2))
    watch = watching(lab, batch, bus)
    first = watch.once()
    assert [(job.job, job.target, job.verdict) for job in first.jobs] == [
        ("gold-1", "gold", "running"),
        ("miyabi-g-2", "miyabi-g", "running"),
    ]
    assert (first.running, first.settled) == (2, False)
    assert len(published(bus, Topic.STATE)) == 2
    watch.once()
    assert len(published(bus, Topic.STATE)) == 2


def test_a_settled_job_is_settled_once_and_carries_what_the_sweep_found(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = batched(lab, bus, spec(*_TWO))
    dispatching(lab, monkeypatch, "77", "78")
    batch.run()
    recorded(lab, "77", target="gold", verdict="ok")
    recorded(lab, "78", target="miyabi-g", verdict="failed")
    sweeping(
        lab,
        monkeypatch,
        # The sweep covers the whole workspace, so it routinely carries jobs of other batches.
        MonitorReport(
            finished=[
                Finished(handle="12", target="crimson", pulled_path="results/elsewhere"),
                Finished(handle="77", target="gold", pulled_path="results/a"),
            ],
            failed=[
                Failed(handle="13", target="crimson", reason="exited 1"),
                Failed(handle="78", target="miyabi-g", reason="exited 137 (out of memory)"),
            ],
        ),
    )
    watch = watching(lab, batch, bus)
    status = watch.once()
    assert status.settled
    assert [(job.verdict, job.detail) for job in status.jobs] == [
        ("ok", "results/a"),
        ("failed", "exited 137 (out of memory)"),
    ]
    assert [event.job for event in published(bus, Topic.SETTLED)] == ["gold-1", "miyabi-g-2"]
    [closed] = published(bus, Topic.CLOSED)
    assert closed.data == {"jobs": 2, "ok": 1, "failed": 1}
    watch.once()
    assert len(published(bus, Topic.SETTLED)) == 2
    assert len(published(bus, Topic.CLOSED)) == 1


def test_the_same_batch_dispatched_again_settles_again(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cursor is the run, not the job, so last week's outcome cannot silence this one."""
    batch = batched(lab, bus, spec(_TWO[0]))
    dispatching(lab, monkeypatch, "77", "88")
    batch.run()
    recorded(lab, "77", target="gold", verdict="ok")
    sweeping(lab, monkeypatch, MonitorReport())
    watching(lab, batch, bus).once()
    batch.run()
    recorded(lab, "88", target="gold", verdict="failed")
    status = watching(lab, batch, bus).once()
    assert [(job.handle, job.verdict) for job in status.jobs] == [("88", "failed")]
    assert [event.data["handle"] for event in published(bus, Topic.SETTLED)] == ["77", "88"]
    assert len(published(bus, Topic.CLOSED)) == 2
    # Every duration is measured within one run: this dispatch against the last run's start is
    # not a duration at all, and negative setup times are what that mistake looks like live.
    assert all(event.data["setup_s"] >= 0.0 for event in published(bus, Topic.COST))


def test_a_job_the_run_registry_never_recorded_still_gets_a_row(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = batched(lab, bus, spec(_TWO[0]))
    dispatching(lab, monkeypatch, "77")
    batch.run()
    sweeping(lab, monkeypatch, MonitorReport())
    [job] = watching(lab, batch, bus).once().jobs
    assert (job.verdict, job.handle) == ("unknown", "77")
    assert "no record" in job.detail


def test_a_refused_job_still_appears_in_the_view_that_watches_the_batch(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job no target took is settled the moment it was refused, and the batch can still close."""
    batch = batched(lab, bus, spec(_TWO[0]))
    refusing(monkeypatch, MissionError("declares no root"))
    batch.run()
    sweeping(lab, monkeypatch, MonitorReport())
    status = watching(lab, batch, bus).once()
    assert [(job.job, job.verdict, job.detail) for job in status.jobs] == [
        ("gold-1", "vanished", "declares no root")
    ]
    assert status.settled


def test_a_job_dispatched_after_being_refused_is_reported_from_its_dispatch(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry that landed is the job's story now, so the earlier refusal is not a second row."""
    batch = batched(lab, bus, spec(_TWO[0]))
    refusing(monkeypatch, MissionError("asleep"))
    batch.run()
    dispatching(lab, monkeypatch, "79")
    batch.run()
    recorded(lab, "79", target="gold", verdict="ok")
    sweeping(lab, monkeypatch, MonitorReport())
    status = watching(lab, batch, bus).once()
    assert [(job.job, job.handle, job.verdict) for job in status.jobs] == [("gold-1", "79", "ok")]


def test_a_settled_run_that_was_seen_running_teaches_the_next_estimate_what_setup_costs(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timeline lives in the receipts, since the registry keeps no moment a command started."""
    batch = batched(lab, bus, spec(_TWO[0]))
    dispatching(lab, monkeypatch, "77")
    batch.run()
    recorded(lab, "77", target="gold")
    passes = sweeping(lab, monkeypatch, MonitorReport(running=1))
    watch = watching(lab, batch, bus)
    watch.once()
    recorded(lab, "77", target="gold", verdict="ok")
    watch.once()
    [cost] = published(bus, Topic.COST)
    assert cost.data["platform"] == "gold"
    assert cost.data["observed"] is True
    assert cost.data["setup_s"] >= 0.0
    assert len(passes) == 2
    assert [row.provider for row in watch.ledger.observations()] == ["gold"]


def test_a_settled_run_reports_what_it_was_quoted_beside_what_it_actually_came_to(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An estimate nobody checks against an outcome is a guess that never improves."""
    batch = batched(lab, bus, spec(_TWO[0]))
    dispatching(lab, monkeypatch, "77")
    batch.run()
    publish(
        bus,
        batch.id,
        Topic.ESTIMATED,
        job="gold-1",
        data={"rate_usd_hr": 3.6, "expected_usd": 1.0},
    )
    recorded(lab, "77", target="gold", verdict="ok")
    sweeping(lab, monkeypatch, MonitorReport())
    watch = watching(lab, batch, bus)
    watch.once()
    [cost] = published(bus, Topic.COST)
    assert cost.data["expected_usd"] == 1.0
    # The run is seconds old here, so what it came to is far under what it was quoted, and the
    # delta is what a later fit learns from.
    assert cost.data["actual_usd"] < 1.0
    assert cost.data["delta_usd"] == pytest.approx(cost.data["actual_usd"] - 1.0)


def test_a_stamp_with_no_offset_is_read_as_utc_not_as_this_machines_local_clock() -> None:
    """The same pinning the billing cycle already does, for the same reason.

    Reading a naive stamp locally would shift a setup time by the machine's whole UTC offset and
    teach every later estimate a wait that never happened.
    """
    assert _epoch("2026-08-25T12:00:00") == _epoch("2026-08-25T12:00:00+00:00")


def test_a_batch_nobody_priced_reports_a_zero_delta_rather_than_inventing_a_comparison(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = batched(lab, bus, spec(_TWO[0]))
    dispatching(lab, monkeypatch, "77")
    batch.run()
    recorded(lab, "77", target="gold", verdict="ok")
    sweeping(lab, monkeypatch, MonitorReport())
    watching(lab, batch, bus).once()
    [cost] = published(bus, Topic.COST)
    assert (cost.data["expected_usd"], cost.data["actual_usd"], cost.data["delta_usd"]) == (
        0.0,
        0.0,
        0.0,
    )


def test_a_run_no_pass_ever_caught_running_is_published_but_never_fitted(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run never caught running teaches the estimator nothing.

    A setup time inferred from a job already over would teach every estimate a wait that
    never happened, so the receipt says what was seen and the ledger stays empty.
    """
    batch = batched(lab, bus, spec(_TWO[0]))
    dispatching(lab, monkeypatch, "77")
    batch.run()
    recorded(lab, "77", target="gold", verdict="ok")
    sweeping(lab, monkeypatch, MonitorReport())
    watch = watching(lab, batch, bus)
    watch.once()
    [cost] = published(bus, Topic.COST)
    assert (cost.data["observed"], cost.data["setup_s"], cost.data["run_s"]) == (False, 0.0, 0.0)
    assert watch.ledger.observations() == []


def test_following_a_batch_repeats_at_its_interval_until_the_last_job_settles(
    lab: Board, bus: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = iter(
        [BatchStatus(batch="smoke-1", jobs=(), running=1), BatchStatus(batch="smoke-1", jobs=())]
    )
    monkeypatch.setattr(Watch, "once", lambda self: next(scripted))
    slept: list[float] = []
    monkeypatch.setattr("mainboard.batch.watch.sleep", slept.append)
    assert [status.settled for status in Watch(lab, "smoke-1", bus=bus).follow(3.0)] == [
        False,
        True,
    ]
    assert slept == [3.0]
