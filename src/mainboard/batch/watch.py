# The live view over a dispatched batch: every job on every target in one table, kept current by
# the same durable sweep a cron already runs.
#
# Nothing here polls a scheduler itself. Each pass runs `Monitor.once` (pulling results back,
# cancelling ended rentals) and reports this batch's jobs out of what it settled, so a watched
# batch settles exactly as an unwatched one does.

from datetime import UTC, datetime
from time import sleep
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from ..costs import Ledger, Observation
from ..dispatch import vocabulary
from ..dispatch.shared import now
from .estimate import platform
from .receipts import OFFERED, Receipts, Topic, latest, publish
from .runner import directory

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..board import Board
    from ..dispatch.state import MonitorReport
    from .receipts import Bus, Event

# Where the workspace keeps the observations a later estimate fits its setup times from.
_COSTS = "costs"

# What a handle the run registry has forgotten reads as, so a row always says something.
_UNKNOWN = "unknown"

# The verdicts a closing count must not read as a failure; a skipped job never ran.
_UNFAILED = frozenset({vocabulary.OK, vocabulary.SKIPPED})


class JobStatus(FrozenModel):
    """One batch job as the last sweep left it.

    handle: empty when the target refused it.
    state: the scheduler's own word for it, empty when nothing has reported one.
    verdict: the normalized outcome, `running` while it is still in flight.
    detail: where the results landed, why it failed, or why it was never dispatched.
    """

    job: str
    target: str
    handle: str = ""
    state: str = ""
    verdict: str = ""
    detail: str = ""


class BatchStatus(FrozenModel):
    """One pass over a batch: a row per job dispatched, refused or left out, and how many of
    them are still `running`."""

    batch: str
    jobs: tuple[JobStatus, ...]
    running: int = 0

    @property
    def settled(self) -> bool:
        return self.running == 0


class Watch:
    """A dispatched batch's live view, one pass at a time.

    Built from the batch id alone, since the receipts name its handles and the dispatch cache
    says what became of them, so any process can watch it, as a cron settles it.
    """

    def __init__(self, board: Board, batch_id: str, *, bus: Bus | None = None) -> None:
        self.board = board
        self.id = batch_id
        self.dir = directory(board, batch_id)
        self.bus = bus or Receipts(self.dir / "events.ndjson")
        self.ledger = Ledger(board.root / Project().out_dir / _COSTS)

    @staticmethod
    def detail(handle: str, swept: MonitorReport) -> str:
        """What this pass's sweep said about `handle`, its results path or why it failed."""
        said = [
            finished.pulled_path or "" for finished in swept.finished if finished.handle == handle
        ]
        said += [failed.reason for failed in swept.failed if failed.handle == handle]
        return next(iter(said), "")

    def close(self, status: BatchStatus, *, settled: bool) -> None:
        """Announce the batch's end only on a pass that `settled` something and left none running,
        keeping a quiet pass quiet while a re-dispatched batch can close again."""
        if not status.settled or not settled:
            return
        publish(
            self.bus,
            self.id,
            Topic.CLOSED,
            data={
                "jobs": len(status.jobs),
                "ok": sum(job.verdict == vocabulary.OK for job in status.jobs),
                "failed": sum(job.verdict not in _UNFAILED for job in status.jobs),
                "skipped": sum(job.verdict == vocabulary.SKIPPED for job in status.jobs),
            },
        )

    def follow(self, interval: float) -> Iterator[BatchStatus]:
        """Repeat `once` every `interval` seconds until every job has settled."""
        while True:
            status = self.once()
            yield status
            if status.settled:
                return
            sleep(interval)

    def observe(self, row: JobStatus, events: Sequence[Event]) -> None:
        """Record what `row` spent, as a receipt always and as a fitted observation when honest.

        The timeline exists only in this batch's lines (the registry never records when a queue
        started the command), matched on the handle since a re-dispatched batch shares its stream
        with earlier runs. A run no pass caught running stays out of the ledger, since its
        inferred setup would teach estimates a wait that never happened. The quote off this
        batch's `job.estimated` line is published beside the actual cost (a zero delta when
        unpriced), so the cost model learns from its misses.
        """
        mine = [
            event
            for event in events
            if event.job == row.job and event.data.get("handle") == row.handle
        ]
        submitted = [event for event in mine if event.topic is Topic.SUBMITTED]
        started = [
            event
            for event in mine
            if event.topic is Topic.STATE and event.data.get("verdict") == vocabulary.RUNNING
        ]
        key = platform(alias=row.target, kind=str(submitted[0].data["kind"]) if submitted else "")
        ended = _epoch(now())
        opened = _epoch(submitted[0].at) if submitted else ended
        running = _epoch(started[0].at) if started else 0.0
        quoted = latest(events, Topic.ESTIMATED).get(row.job)
        actual = _money(quoted, "rate_usd_hr") * (ended - opened) / 3600.0
        expected = _money(quoted, "expected_usd")
        publish(
            self.bus,
            self.id,
            Topic.COST,
            job=row.job,
            data={
                "platform": key,
                "setup_s": (running - opened) if running else 0.0,
                "run_s": (ended - running) if running else 0.0,
                "observed": bool(running),
                "expected_usd": round(expected, 4),
                "actual_usd": round(actual, 4),
                "delta_usd": round(actual - expected, 4),
            },
        )
        if running:
            self.ledger.record(
                Observation(
                    provider=key,
                    t_submit=opened,
                    t_running=running,
                    t_ended=ended,
                    billed_usd=round(actual, 4),
                )
            )

    def once(self) -> BatchStatus:
        """Settle whatever ended, then report every job of this batch as it now stands.

        The sweep covers the whole workspace, since another batch's rental still bills while this
        one watches. Each row comes from the newest of its target's three answers (`OFFERED`).
        """
        swept = self.board.monitor().once()
        events = self.bus.replay()
        rows = [
            self.answered(job, answer, swept) for job, answer in latest(events, *OFFERED).items()
        ]
        landed = [self.record(row, events) for row in rows]
        status = BatchStatus(
            batch=self.id,
            jobs=(*rows, *self.unselected(events, dispatched={row.job for row in rows})),
            running=sum(row.verdict not in vocabulary.TERMINAL for row in rows),
        )
        self.close(status, settled=any(landed))
        return status

    def record(self, row: JobStatus, events: Sequence[Event]) -> bool:
        """Publish whatever changed about `row` since the last pass, and say if it settled here.

        The cursor is the run (handle), not the job, so last week's settlement never silences a
        re-dispatch finishing now. The cost is published before `job.settled`, keeping the
        terminal line last.
        """
        seen = latest(events, Topic.STATE).get(row.job)
        reported = (seen.data.get("handle"), seen.data.get("verdict")) if seen else ()
        moved = seen is None or reported != (row.handle, row.verdict)
        if moved:
            publish(
                self.bus,
                self.id,
                Topic.STATE,
                job=row.job,
                data={"handle": row.handle, "state": row.state, "verdict": row.verdict},
            )
        settled = latest(events, Topic.SETTLED).get(row.job)
        if row.verdict not in vocabulary.TERMINAL or (
            settled is not None and settled.data.get("handle") == row.handle
        ):
            return False
        self.observe(row, events)
        publish(
            self.bus,
            self.id,
            Topic.SETTLED,
            job=row.job,
            data={"handle": row.handle, "verdict": row.verdict, "detail": row.detail},
        )
        return True

    def answered(self, job: str, answer: Event, swept: MonitorReport) -> JobStatus:
        """One job's row from the newest of its `OFFERED` answers.

        Ranking the answers by topic instead rendered a job taken and then refused as still flying.
        """
        target = str(answer.data["target"])
        if answer.topic is Topic.REFUSED:
            return JobStatus(
                job=job,
                target=target,
                verdict=vocabulary.VANISHED,
                detail=str(answer.data["reason"]),
            )
        # A held job is still coming (the sweep asks again every pass), so it counts as in flight.
        if answer.topic is Topic.HELD:
            return JobStatus(
                job=job,
                target=target,
                state=vocabulary.HELD,
                verdict=vocabulary.HELD,
                detail=f"waiting on the target's quota: {answer.data['reason']}",
            )
        return self.status(job, answer, swept)

    def status(self, job: str, submitted: Event, swept: MonitorReport) -> JobStatus:
        """One dispatched job's row, from the run registry and this pass's own sweep."""
        handle = str(submitted.data["handle"])
        target = str(submitted.data["target"])
        try:
            record = self.board.dispatcher.cache.run(handle, target)
        except LookupError:
            return JobStatus(
                job=job,
                target=target,
                handle=handle,
                verdict=_UNKNOWN,
                detail="the run registry has no record of this handle",
            )
        return JobStatus(
            job=job,
            target=target,
            handle=handle,
            state=record.state or "",
            verdict=record.verdict or vocabulary.RUNNING,
            detail=self.detail(handle, swept),
        )

    @staticmethod
    def unselected(events: Sequence[Event], *, dispatched: set[str]) -> list[JobStatus]:
        """Every job a run left out and no later wave dispatched, as a row already over.

        Shown so a wave is read against the plan, but settling nothing: an undispatched job has
        nothing to pull back, release or bill.
        """
        return [
            JobStatus(
                job=job,
                target=str(event.data["target"]),
                verdict=vocabulary.SKIPPED,
                detail=str(event.data["reason"]),
            )
            for job, event in latest(events, Topic.SKIPPED).items()
            if job not in dispatched
        ]


def _epoch(stamp: str) -> float:
    """An ISO-8601 instant as epoch seconds, a naive one read as UTC as the billing cycle does.

    Only a foreign or older receipts file is naive; reading it as local time would add the UTC
    offset (nine hours here) of imaginary provisioning to every later estimate.
    """
    read = datetime.fromisoformat(stamp)
    return (read if read.tzinfo else read.replace(tzinfo=UTC)).timestamp()


def _money(event: Event | None, field: str) -> float:
    """A dollar figure off a free-form event payload, zero when absent or unreadable rather than
    raising mid-settle."""
    amount = event.data.get(field) if event is not None else None
    return float(amount) if isinstance(amount, int | float) else 0.0
