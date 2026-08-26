# The live view over a dispatched batch: every job on every target in one table, kept current by
# the same durable sweep a cron already runs.
#
# Nothing here polls a scheduler itself. Each pass runs `Monitor.once`, which is what pulls a
# finished job's results back and cancels a rental whose command has ended, and then reports this
# batch's own jobs out of the state that sweep just settled. That is the whole point of driving
# the sweep rather than reimplementing it: a batch nobody is watching still settles, and a batch
# somebody is watching settles the same way.

from datetime import datetime
from time import sleep
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from ..costs import Ledger, Observation
from ..dispatch import vocabulary
from ..dispatch.shared import now
from .estimate import platform
from .receipts import Receipts, Topic, latest, publish
from .runner import directory

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..board import Board
    from ..dispatch.state import MonitorReport
    from .receipts import Bus, Event

# Where the workspace keeps the observations a later estimate fits its setup times from.
_COSTS = "costs"

# What a job reads as before any sweep has resolved it, and what a handle the run registry has
# forgotten reads as, so a row always says something.
_UNKNOWN = "unknown"


class JobStatus(FrozenModel):
    """One batch job as the last sweep left it.

    job: the job's name inside the batch.
    target: the alias it runs on.
    handle: its scheduler or provider handle, empty when the target refused it.
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
    """One pass over a batch: every job's row and what the batch adds up to.

    batch: the batch id.
    jobs: one row per job the batch ever dispatched or was refused for.
    running: how many are still in flight.
    """

    batch: str
    jobs: tuple[JobStatus, ...]
    running: int = 0

    @property
    def settled(self) -> bool:
        """Whether every job has reached a terminal verdict, so watching can stop."""
        return self.running == 0


class Watch:
    """A dispatched batch's live view, one pass at a time.

    Built from the batch id alone, since everything it needs is durable: the receipts say which
    handles belong to this batch and the dispatch cache says what became of them. A process that
    never dispatched the batch can therefore watch it, which is the same property that lets a
    cron settle it.
    """

    def __init__(self, board: Board, batch_id: str, *, bus: Bus | None = None) -> None:
        """board: the workspace whose dispatch cache and sweep the pass reads.

        batch_id: the batch to watch.
        bus: where receipts go, the batch's own NDJSON file when None.
        """
        self.board = board
        self.id = batch_id
        self.dir = directory(board, batch_id)
        self.bus = bus or Receipts(self.dir / "events.ndjson")
        self.ledger = Ledger(board.root / Project().out_dir / _COSTS)

    @staticmethod
    def detail(handle: str, swept: MonitorReport) -> str:
        """What this pass's sweep said about `handle`, its results path or why it failed."""
        for finished in swept.finished:
            if finished.handle == handle:
                return finished.pulled_path or ""
        for failed in swept.failed:
            if failed.handle == handle:
                return failed.reason
        return ""

    def close(self, status: BatchStatus, *, settled: bool) -> None:
        """Announce the batch's end on the pass that settles its last job, and only then.

        Tying the announcement to a settlement this pass made is what keeps a quiet pass quiet
        and still lets a re-dispatched batch close a second time.

        status: this pass's rows.
        settled: whether this pass settled anything at all.
        """
        if not status.settled or not settled:
            return
        publish(
            self.bus,
            self.id,
            Topic.CLOSED,
            data={
                "jobs": len(status.jobs),
                "ok": sum(1 for job in status.jobs if job.verdict == vocabulary.OK),
                "failed": sum(1 for job in status.jobs if job.verdict != vocabulary.OK),
            },
        )

    def follow(self, interval: float) -> Iterator[BatchStatus]:
        """Repeat `once` every `interval` seconds until every job has settled.

        interval: seconds between passes.
        """
        while True:
            status = self.once()
            yield status
            if status.settled:
                return
            sleep(interval)

    def observe(self, row: JobStatus, events: Sequence[Event]) -> None:
        """Record what `row` spent, as a receipt always and as a fitted observation when honest.

        The timeline comes from this batch's own lines, which is the only place it exists: the
        run registry keeps a submit time and a verdict, never the moment a queue actually started
        the command. Every line is matched on the handle, since a batch dispatched again shares
        its stream with the runs before it and a dispatch from this run against a start from the
        last one is not a duration at all. A run no pass ever caught running is published all the
        same and kept out of the ledger, since a setup time inferred from a job that was already
        over would teach every later estimate to expect a wait that never happened.

        What this run was quoted at is published beside what it actually came to, because an
        estimate nobody ever checks against an outcome is a guess that never improves. The quote
        is read back off this batch's own `job.estimated` line, so a batch nobody priced reports
        a zero delta rather than inventing a comparison, and the figure lands on the observation
        as well, so a later fit stands on money as well as on seconds.
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
        kind = str(submitted[0].data["kind"]) if submitted else ""
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
                "platform": platform(alias=row.target, kind=kind),
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
                    provider=platform(alias=row.target, kind=kind),
                    t_submit=opened,
                    t_running=running,
                    t_ended=ended,
                    billed_usd=round(actual, 4),
                )
            )

    def once(self) -> BatchStatus:
        """Settle whatever ended, then report every job of this batch as it now stands.

        The sweep runs first and over the whole workspace rather than over this batch, since a
        rental this batch does not own still bills while this one watches, and settling it costs
        one probe that was going to happen anyway.
        """
        swept = self.board.monitor().once()
        events = self.bus.replay()
        rows = [
            self.status(job, event, swept)
            for job, event in latest(events, Topic.SUBMITTED).items()
        ]
        rows += [
            JobStatus(
                job=job,
                target=str(event.data["target"]),
                verdict=vocabulary.VANISHED,
                detail=str(event.data["reason"]),
            )
            for job, event in latest(events, Topic.REFUSED).items()
            if job not in {row.job for row in rows}
        ]
        landed = [self.record(row, events) for row in rows]
        status = BatchStatus(
            batch=self.id,
            jobs=tuple(rows),
            running=sum(1 for row in rows if row.verdict not in vocabulary.TERMINAL),
        )
        self.close(status, settled=any(landed))
        return status

    def record(self, row: JobStatus, events: Sequence[Event]) -> bool:
        """Publish whatever changed about `row` since the last pass, and say if it settled here.

        The cursor is the run rather than the job, since a batch re-dispatched under the same
        declaration is the same batch with new handles, and a job that settled last week must
        not silence the run of it that is finishing now.

        What the run spent is published before it settles, so `job.settled` really is the last
        line a subscriber sees about this job. That is what lets a sink close its own record of
        the run on the terminal line without losing the cost that follows it.
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


def _epoch(stamp: str) -> float:
    """An ISO-8601 instant as epoch seconds, the footing an observation is recorded on."""
    return datetime.fromisoformat(stamp).timestamp()


def _money(event: Event | None, field: str) -> float:
    """A dollar figure off an event payload, zero when no line ever carried one.

    A payload is free-form JSON, so a batch nobody priced and a batch whose estimate wrote
    something unreadable both answer zero rather than raising in the middle of a settle.
    """
    amount = event.data.get(field) if event is not None else None
    return float(amount) if isinstance(amount, int | float) else 0.0
