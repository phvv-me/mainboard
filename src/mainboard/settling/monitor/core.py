"""The durable pass over every dispatched job the shared cache still owes an outcome for."""

from collections.abc import Mapping
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Final

from filelock import Timeout
from patos import FrozenModel

from ...batch.receipts import Topic
from ...core.errors import MissionError
from ...dispatch.jobs import vocabulary
from ...dispatch.jobs.evidence import receipts_in
from ...dispatch.schedulers import short_reason
from ...dispatch.state import DownHost, Failed, Finished, Held, MonitorReport, Resumed
from ...dispatch.state.monitor import SettlementBusy
from ...log import logger
from ...tracking import streamed
from ..events import Events
from .constants import COPIED
from .deadlines import Deadlines, bounded_pass
from .settlement import Settlement, Unverifiable, failure, identity, owes_receipt
from .sweep import Sweep, uncreated

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ...batch.receipts import Event
    from ...dispatch.jobs.vocabulary import JobState
    from ...dispatch.state import RunRecord
    from ...fleet.board import Board, Run

# How long one process leaves a target alone after it did not answer a pass.
_SILENT_SECONDS: Final = 30.0


class Collected(FrozenModel):
    """What a finished run left: its receipts, where its results landed, and its log."""

    harvested: tuple[str, ...]
    pulled: str | None
    transcript: str


class Monitor:
    """The durable pass over every dispatched job still owed an outcome.

    One `once` probes each unsettled run, pulls back whatever just finished, releases what a
    settled run still holds, and advances the run's reported cursor so the next pass says
    nothing. A host that does not answer is reported once and its jobs are left for the next
    pass. Releasing is why this is worth running unattended: a queue stops charging when the job
    ends, but a provider keeps the instance billing after its command exits, so a terminal
    verdict here is followed by a cancel the scheduler path deliberately does not make.
    """

    def __init__(self, board: Board) -> None:
        self.board = board
        self.cache = board.dispatcher.cache
        self.settlement = Settlement(board)
        self.silent: dict[str, tuple[float, str]] = {}

    def _bounded(self, scope: Callable[[RunRecord], bool] | None) -> MonitorReport:
        """One pass claimed and run under the pass deadline (`bounded_pass`)."""
        with bounded_pass() as deadline:
            return self._claimed(scope, deadline=deadline)

    def _claimed(
        self, scope: Callable[[RunRecord], bool] | None, *, deadline: str
    ) -> MonitorReport:
        expired = Deadlines(self.settlement).expired()
        lock = self.cache.settlement
        try:
            lock.acquire(timeout=0)
        except Timeout:
            logger.debug("another monitor owns settlement; leaving its cursor untouched")
            return MonitorReport(busy=SettlementBusy.read(Path(lock.lock_file)), failed=expired)
        try:
            report = self._settled(scope, deadline=deadline)
        finally:
            lock.release()
        return report.model_copy(update={"failed": [*expired, *report.failed]})

    def _collected(self, record: RunRecord, job: Run, *, stream: str) -> Collected:
        """Pull a finished run's results home and capture its log, unless its host went quiet."""
        self.settlement.answering(job)
        pulled = self.settlement.pull(job)
        self.settlement.answering(job)
        harvested = self.settlement.capture(record, job)
        transcript = self.settlement.captured.transcript(stream, handle=record.handle) or ""
        return Collected(harvested=harvested, pulled=pulled, transcript=transcript)

    def _concluded(
        self, record: RunRecord, job: Run, state: JobState, *, collected: Collected, why: str
    ) -> Finished | Failed:
        """Release a run whose evidence was copied, then record how it ended and move its cursor.

        why: why its evidence can never be verified, empty when it was.
        """
        harvested, pulled = collected.harvested, collected.pulled
        self.settlement.evidence(record, harvested, status="copied")
        if not self.settlement.release(job):
            detail = "settlement pending; release failed and will be retried"
            self.settlement.evidence(record, harvested, status="copied", detail=detail)
            return failure(record, detail)
        status = "unverified" if why else "verified"
        self.settlement.evidence(record, harvested, status=status, detail=why)
        if state.verdict == vocabulary.Word.OK:
            detail = pulled or ""
            outcome: Finished | Failed = Finished(
                handle=record.handle, target=record.target, pulled_path=pulled
            )
        else:
            detail = short_reason(state.verdict, state.exit_code)
            outcome = Failed(
                handle=record.handle, target=record.target, reason=detail, pulled_path=pulled
            )
        self.settlement.track(record, state, detail=detail)
        self.cache.report(self.cache.run(record.handle, record.target), state.verdict)
        return outcome

    def _harvested(
        self, record: RunRecord, job: Run, state: JobState, current: RunRecord
    ) -> Finished | Failed:
        """Collect what a finished run left, verify it, then release the run and settle it.

        A run whose evidence an earlier pass copied is read back from the local log instead.
        """
        stream, name = streamed(record.name or "", handle=record.handle)
        self.cache.settlement.phase(f"reading the receipts of {stream}")
        previous = self.settlement.stream(stream).get(
            Topic.EVIDENCE, name, submission=identity(record)
        )
        copied = current.evidence in COPIED or (
            previous is not None and previous.data.get("status") in COPIED
        )
        try:
            collected = (
                self._recovered(record, job, stream=stream, previous=previous)
                if copied
                else self._collected(record, job, stream=stream)
            )
        except (MissionError, OSError, ValueError) as fault:
            return self._pending(record, (), fault=fault)
        owed = owes_receipt(record, transcript=collected.transcript, verdict=state.verdict)
        why = ""
        try:
            self.settlement.verify(job, collected.pulled, collected.harvested, owed=owed)
        except Unverifiable as fault:
            why = str(fault)
        except (MissionError, OSError, ValueError) as fault:
            return self._pending(record, collected.harvested, fault=fault)
        return self._concluded(record, job, state, collected=collected, why=why)

    def _heard(self, down: Mapping[str, str]) -> None:
        """Remember the targets that did not answer this pass, so the next ones leave them be."""
        asked = monotonic()
        self.silent = {target: (asked, why) for target, why in down.items()} | self.silent

    def _once(self, scope: Callable[[RunRecord], bool] | None = None) -> MonitorReport:
        """Resolve selected unsettled runs, harvest the newly terminal ones, report the changes.

        Resolving goes by target (see `Sweep`); the harvest then walks runs in cache order, so
        what a run causes does not depend on which target answered first. A run in flight is only
        counted. A run that ended has its results pulled, its output and trial receipts captured
        beside its receipts stream, whatever it holds released, and only then its cursor
        advanced, so a sweep killed halfway repeats work rather than losing an outcome. Capture
        must precede release, which destroys a rented instance and its log; release must precede
        the cursor, so a pass dying between them cancels the rental again next time.

        The pull happens whatever the exit code said. A sweep of 500 trials dying at 400 leaves
        399 immutable receipt fragments on the host, exactly the crash safety the staged store
        exists to give, and a partial sweep is also how a metered rental that hit its cap ends.
        """
        self.settlement.quiet.clear()
        self.settlement.fetched.clear()
        self.cache.settlement.phase("resuming held jobs")
        resumed, waiting, failed = self.held(scope)
        records = [
            record for record in self.cache.listed("tracked") if scope is None or scope(record)
        ]
        self.cache.settlement.phase("probing hosts")
        resolved = Sweep(self.board, records, down=self._silenced())
        self._heard(resolved.down)
        outcomes = [
            self._settle(record, resolved.runs[record], state)
            for record in records
            if (state := resolved.states.get(record)) is not None
        ]
        return MonitorReport(
            running=outcomes.count(None) + len(waiting),
            resumed=resumed,
            held=waiting,
            finished=[outcome for outcome in outcomes if isinstance(outcome, Finished)],
            failed=[
                *failed,
                *(failure(record, why) for record in records if (why := uncreated(record))),
                *(outcome for outcome in outcomes if isinstance(outcome, Failed)),
            ],
            unreachable_hosts=[
                DownHost(host=host, reason=why) for host, why in resolved.down.items()
            ],
        )

    def _pending(
        self, record: RunRecord, harvested: tuple[str, ...], *, fault: Exception
    ) -> Failed:
        """Keep a run whose evidence could not be settled pending, its remote copy retained."""
        detail = f"settlement pending; remote evidence retained: {fault}"
        self.settlement.evidence(record, harvested, status="pending", detail=detail)
        logger.error("{} on {}: {}", record.handle, record.target, detail)
        return failure(record, detail)

    def _recovered(
        self, record: RunRecord, job: Run, *, stream: str, previous: Event | None
    ) -> Collected:
        """The evidence an earlier pass already copied, read back from the local receipt log."""
        kept = self.settlement.captured.transcript(stream, handle=record.handle)
        if previous is None and kept is None:
            raise MissionError("copied evidence has no recoverable local receipt log")
        harvested = receipts_in(kept or "")
        if previous is not None and previous.data.get("trials") and not harvested:
            raise MissionError("copied trial receipts are missing from the local log")
        return Collected(harvested=harvested, pulled=job.handle.fetch_path, transcript=kept or "")

    def _settle(self, record: RunRecord, job: Run, state: JobState) -> Finished | Failed | None:
        """One resolved run: counted while in flight (None), else carried to its outcome."""
        self.cache.settlement.phase(f"settling {record.target}/{record.handle}")
        current = self.cache.resolve(record, state.state, state.exit_code, state.verdict)
        if state.verdict not in vocabulary.Word.settled():
            if state.stage == vocabulary.Word.RUNNING:
                self.settlement.pull(job)
            self.settlement.track(record, state, detail="")
            return None
        discarded = state.state == vocabulary.Word.CANCELLED and current.evidence == "unverified"
        if current.evidence == "not_started" or discarded:
            return self._unlaunched(record, job, state, discarded=discarded)
        return self._harvested(record, job, state, current)

    def _settled(
        self, scope: Callable[[RunRecord], bool] | None, *, deadline: str
    ) -> MonitorReport:
        """Run one pass under the claimed settlement lock and mark the lock's work done."""
        self.cache.settlement.phase("settling", deadline=deadline)
        report = self._once(scope)
        self.cache.settlement.succeeded()
        return report

    def _silenced(self) -> dict[str, str]:
        """The targets that did not answer a pass of this process too recently to ask again.

        A wait sweeps every few seconds, and a dead host cost each of its passes a connect
        timeout, whichever host the waited job ran on.
        """
        now = monotonic()
        self.silent = {
            target: (since, why)
            for target, (since, why) in self.silent.items()
            if now - since < _SILENT_SECONDS
        }
        return {target: why for target, (_, why) in self.silent.items()}

    def _unlaunched(
        self, record: RunRecord, job: Run, state: JobState, *, discarded: bool
    ) -> Failed:
        """Settle a run that ended with nothing to collect, released before its cursor moves.

        discarded: an explicit cancellation whose evidence already reads unverified, rather than
            a provisioning that ended before a native launch was attempted.
        """
        detail = (
            "explicit cancellation; evidence remains unverified"
            if discarded
            else "provisioning ended before a native launch was attempted"
        )
        if not self.settlement.release(job):
            return failure(record, f"{detail}; release failed and will be retried")
        if not discarded:
            self.settlement.evidence(record, (), status="not_started", detail=detail)
        self.settlement.track(record, state, detail=detail)
        self.cache.report(record, state.verdict)
        return failure(record, detail)

    def held(
        self, scope: Callable[[RunRecord], bool] | None = None
    ) -> tuple[list[Resumed], list[Held], list[Failed]]:
        """Ask every quota-held dispatch's target for room again, in the order they were held.

        This makes a hold a delay rather than a loss: a wave meeting a group's job limit used to
        drop what the queue would not take, found by counting logs hours later (miyabi-g njobs-g,
        2026-09-04). A request that goes through replaces its placeholder row under the new
        handle and tells the batch that asked, through its own receipts.
        """
        outcomes = [
            self.settlement.resumed(record)
            for record in reversed(self.cache.listed("live"))
            if record.verdict == vocabulary.Word.HELD and (scope is None or scope(record))
        ]
        return (
            [outcome for outcome in outcomes if isinstance(outcome, Resumed)],
            [outcome for outcome in outcomes if isinstance(outcome, Held)],
            [outcome for outcome in outcomes if isinstance(outcome, Failed)],
        )

    def once(self, *, scope: Callable[[RunRecord], bool] | None = None) -> MonitorReport:
        """Settle selected submissions under the shared lock; rental deadlines remain global.

        scope: selects fresh registry rows inside the lock, all unsettled runs when omitted.

        Another process holding the lock ends the pass at once with `busy` naming it. A pass that
        outlives its deadline, or whose parent session dies, kills this whole process: the lock
        is the OS's to free, and the next pass retries from the durable checkpoints.
        """
        try:
            report = self._bounded(scope)
        finally:
            self.settlement.streams.clear()
        Events(self.board.root).record(self.board.dispatcher.cache, report)
        return report

    def watch(self, interval: float) -> Iterator[MonitorReport]:
        """Repeat `once` every `interval` seconds, yielding each pass's report as it lands.

        The foreground loop a person watches; nothing durable depends on it.
        """
        while True:
            yield self.once()
            sleep(interval)
