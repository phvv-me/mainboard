# The anti-fabrication read behind `mb job show` and the block behind `mb job show --wait`.
# Everything printed here comes from recorded receipts and the durable run registry, never from a
# dashboard, a digest or a live session's memory: a notification says a job probably ended, this
# module reads its outcome.
#
# Three targets resolve to one settled view. A receipts file is read line by line in both shapes
# the workspace writes, the batch `Event` envelope and the printed `trial_receipt` line. A stream
# id reads that stream's events and captured receipts in the lake. A handle resolves through
# the run registry to the stream its dispatch was tracked under. Under every target the registry
# row is the floor: a row the receipts left in flight is re-read from the registry, where the
# unattended sweep records an outcome nobody was watching for, and a run whose workspace tracks
# nothing still answers.

import json
from collections.abc import Sequence
from time import monotonic, sleep
from typing import TYPE_CHECKING

from filelock import Timeout
from patos import FrozenModel, Model
from pydantic import ValidationError

from ..batch import receipts as receipts_module
from ..batch.receipts import Event, Journal, Topic, latest
from ..core.errors import MissionError
from ..dispatch.jobs import vocabulary
from ..dispatch.jobs.vocabulary import JobState
from ..dispatch.schedulers import HostUnreachable, short_reason
from ..dispatch.state.captured import Captured
from ..log import logger
from ..settling.monitor import Settlement, owes_receipt
from ..tracking import streamed
from ..trials.store.dataset import Dataset
from .diagnosis import reason
from .pulse import Pulses
from .vigil import Patience, Vigil

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pydantic import JsonValue

    from ..dispatch.state import RunRecord
    from ..fleet.board import Board, Run
    from ..settling.monitor import Monitor

# The key a printed trial receipt carries its payload under, spelled here rather than imported so
# reading a receipt never drags the lab machinery in.
_RECEIPT = "trial_receipt"

# The code while any row is still running or legitimately waiting, the answer a waiter loops on.
_IN_FLIGHT = 2
# A wait whose job went silent on an idle card, distinct from a timeout's 2 so a script can tell
# a job still working from one that stopped doing anything.
_STALLED = 4
# How long a cancel waits for the settlement claim another process holds before it refuses.
_SETTLEMENT_SECONDS = 120.0
# Settled word to exit code: 0 ok, 1 failed, 2 in flight, 3 vanished or unknown. A cancel exits 1
# because a completion check must never call a stopped run complete, however deliberate the stop;
# a skip exits 0 because nothing was dispatched, so no run is incomplete. A quota hold is 2: the
# sweep still offers it, which keeps a batch of thirteen from reading as a finished nine.
_EXITS = {
    vocabulary.Word.OK: 0,
    "passed": 0,
    vocabulary.Word.FAILED: 1,
    "refused": 1,
    vocabulary.Word.TIMEOUT: 1,
    vocabulary.Word.CANCELLED: 1,
    vocabulary.Word.RUNNING: 2,
    vocabulary.Word.HELD: 2,
    vocabulary.Word.SKIPPED: 0,
    "blocked": 2,
    "": 2,
    vocabulary.Word.QUEUED: 2,
    vocabulary.Word.PREPARED: 2,
    vocabulary.Word.SUBMITTING: 2,
}
_INCOMPLETE = "evidence settlement is incomplete"


# A wait that holds on until its target settles, at the vigil's own stall threshold.
_PATIENCE = Patience()


class TrialVerdict(FrozenModel):
    """One trial or job as its receipts left it; every text field is empty when unknown.

    job: the trial's name inside its stream, or a trial receipt's case id.
    handle: the scheduler or provider handle, empty for an in-process trial.
    node: the ledger slug the run serves.
    state: the scheduler's own word.
    verdict: the settled word the exit code derives from, `running` while nothing is terminal.
    settled: the word a trial receipt's own vocabulary settled on. Kept apart from `verdict`
        because a harness may settle `refuted` or `abandoned` on a reading taken perfectly well,
        which a completion check must read as the success it is.
    detail: where results landed, why it failed, or a trial receipt's own reason.
    gates: the gate sweep summarized.
    producer: the harness that stamped a trial receipt.
    contended: what the machine was already doing when this run started, empty for an idle node
        or no attestation; it tells a measurement from one taken while another job held the GPU.
    cause: a failed run's own last meaningful output line, where `detail` only says it failed.
    commit: the commit the dispatching tree was at, named even for a mirror with no history.
    digest: the content digest of that tree, what a run seals against where there is no git: the
        mirror carries the bytes and not the history.
    """

    job: str
    run: str = ""
    handle: str = ""
    target: str = ""
    node: str = ""
    state: str = ""
    verdict: str = ""
    settled: str = ""
    exit_code: int | None = None
    detail: str = ""
    cause: str = ""
    gates: str = ""
    producer: str = ""
    contended: str = ""
    commit: str = ""
    digest: str = ""

    @property
    def code(self) -> int:
        """This trial's exit code, 3 for a word outside the settled table."""
        return _EXITS.get(self.verdict, 3)


class StreamVerdict(FrozenModel):
    """The settled truth of one stream, every trial's row and what they add up to.

    trials: one row per trial or job, in first-seen order.
    note: why there are no rows, empty whenever there are. A bare empty table cannot say whether
        the run has not started, the evidence went elsewhere, or the harness wrote a shape this
        verb does not read.
    stalled: why a wait stopped on a job that went silent on an idle card, empty otherwise.
    """

    stream: str
    trials: tuple[TrialVerdict, ...]
    note: str = ""
    stalled: str = ""

    @property
    def code(self) -> int:
        """The exit status: failure, then stall, then in flight, then vanished; 0 only all clean.

        An empty stream is 3, since receipts that do not exist prove nothing.
        """
        codes = {trial.code for trial in self.trials}
        if 1 in codes:
            return 1
        if self.stalled:
            return _STALLED
        if _IN_FLIGHT in codes:
            return _IN_FLIGHT
        if 3 in codes or not codes:
            return 3
        return 0


class _Watched(Model):
    """What one wait follows: its run or batch, which may move to the run resumed in its place,
    and the hosts it already said went quiet."""

    handle: str
    host: str
    stream: bool
    silent: set[str] = set()


class Verdicts:
    """The receipts-derived outcomes of a workspace's runs, read fresh on every ask."""

    @staticmethod
    def __events(events: Iterable[Event], record: RunRecord) -> list[Event]:
        """Select one submission, refusing ambiguous legacy events without a host."""
        targets = {
            str(event.data.get("target", ""))
            for event in events
            if event.topic == Topic.SUBMITTED and event.data.get("handle") == record.handle
        }
        return [
            event
            for event in events
            if event.data.get("handle") == record.handle
            and event.at >= record.submitted_at
            and event.data.get("submitted_at", record.submitted_at) == record.submitted_at
            and (
                event.data.get("target") == record.target
                or (not event.data.get("target") and targets == {record.target})
            )
        ]

    @staticmethod
    def __preserved(record: RunRecord, run: Run, settlement: Settlement) -> tuple[str, ...]:
        """Capture and verify what `run` left before it stops, recording how far that got."""
        receipts: tuple[str, ...] = ()
        try:
            receipts = settlement.capture(record, run, required=False)
            settlement.verify(run, settlement.pull(run), receipts, owed=owes_receipt(record))
        except (MissionError, OSError, ValueError) as fault:
            settlement.evidence(
                record, receipts, status="unverified", detail=f"cancelled: {fault}"
            )
        else:
            settlement.evidence(record, receipts, status="copied")
        return receipts

    @staticmethod
    def __unconfirmed(record: RunRecord) -> None:
        raise MissionError(
            f"creation {record.creation} has no confirmed provider handle; reconcile "
            "its provider label before cancellation; absence from one listing is not proof"
        )

    @staticmethod
    def _killed(run: Run, settlement: Settlement) -> str:
        """Stop `run` on its host, answering what to add to its record when the host is silent.

        A run on a host that does not answer still settles here, which is all a cancel can do
        about it, and the record and a warning say the job was never stopped there: the fault
        used to leave this verb as a traceback with the run half settled (gold, 2026-09-30). A
        host that went quiet while the evidence was pulled is not knocked on again.
        """
        target = run.handle.host
        quiet = settlement.quiet.get(target, "")
        if not quiet:
            try:
                run.kill()
            except HostUnreachable as fault:
                quiet = str(fault)
        if not quiet:
            return ""
        logger.warning(
            "{} did not answer, so {} settles here but was not stopped there and may still be "
            "running; stop it on {} by hand once it answers: {}",
            target,
            run.handle.id,
            target,
            quiet,
        )
        return f"; {target} did not answer, so the job was not stopped there"

    @staticmethod
    def _registered(record: RunRecord, *, job: str) -> TrialVerdict:
        """The run registry's own row as a settled row, what a receiptless run answers."""
        trial = TrialVerdict(
            job=job,
            handle=record.handle,
            target=record.target,
            node=record.node,
            state=record.state or "",
            verdict=record.verdict or vocabulary.Word.RUNNING,
            exit_code=record.exit_code,
            commit=record.commit,
            digest=record.digest,
        )
        return _delivery(trial, record)

    @staticmethod
    def _silent(stream: str, events: list[Event], trials: Sequence[TrialVerdict]) -> str:
        """Why stream `stream` yielded no rows, empty when it yielded some.

        A stream kept in the lake rather than a file reads as `unreadable`.
        """
        if trials:
            return ""
        if not events:
            return f"stream {stream} has recorded nothing yet"
        return (
            f"stream {stream} holds {len(events)} event(s), none of which settles a trial yet: a "
            "trial settles on its job.settled event or on a `trial_receipt` line its log prints"
        )

    @staticmethod
    def _stopped(ended: str, exit_code: int | None) -> str:
        """Why a stopped run settled where it did, the detail its settled receipt carries."""
        if ended == vocabulary.Word.CANCELLED:
            return short_reason(vocabulary.Word.CANCELLED, None)
        return f"pytest session ended with exit {exit_code}; the lingering process was stopped"

    @staticmethod
    def _unreadable(path: Path, trials: Sequence[TrialVerdict]) -> str:
        """Why `path` yielded no rows, empty when it yielded some.

        A silent empty table reads as a failure and usually is not one, and guessing at it is the
        fabrication this module exists to prevent. The two shapes are named rather than the tools
        that write them: this verb reads a contract, so any harness printing one of them is read.
        """
        if trials:
            return ""
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return f"{path} is empty: nothing has been recorded for this run yet"
        return (
            f"{path} holds {len(lines)} line(s), none of which is evidence this verb reads. It "
            "reads the batch event envelope and the `trial_receipt` line, so a harness writing "
            "another shape settles here as soon as it prints one of those two per trial"
        )

    def __explained(self, trial: TrialVerdict, record: RunRecord) -> TrialVerdict:
        """A failed `trial` with its cause read off the log the sweep brought home."""
        if trial.code != 1:
            return trial
        return trial.model_copy(update={"cause": reason(self.board, record)})

    def __floor(self, record: RunRecord, *, job: str) -> TrialVerdict:
        """`record`'s own row, the answer for a run whose stream holds no receipts at all."""
        return self.__explained(Verdicts._registered(record, job=job), record)

    def __init__(self, board: Board) -> None:
        self.board = board
        self.kept = Captured(board.dispatcher.cache.session)

    def __looked(
        self, watched: _Watched, vigil: Vigil, *, patience: Patience
    ) -> StreamVerdict | None:
        """Settle the watched runs whose session ended while their process lingers; the
        answer once one stalled, after a pause when nothing lingered."""
        scope = self.__scope(watched)
        look = vigil.look(
            [
                record
                for record in self.board.dispatcher.cache.listed("live")
                if record.verdict == vocabulary.Word.RUNNING and scope(record)
            ]
        )
        for linger in look.lingering:
            self._conclude(linger.handle, host=linger.target, session=linger.session)
        if look.stalled:
            answer = self.__standing(watched)
            return answer.model_copy(update={"stalled": look.stalled})
        if not look.lingering:
            sleep(patience.interval)
        return None

    def __of(
        self, target: str, *, host: str, run: str, monitor: Monitor | None = None
    ) -> StreamVerdict:
        path = self.board.dispatcher.local(target)
        stored = self.stored(path, stream=target, run=run)
        if stored is not None:
            return stored
        if path.is_file():
            read = _parsed(path.read_text(encoding="utf-8").splitlines(), source=str(path))
            return StreamVerdict(stream=target, trials=read, note=Verdicts._unreadable(path, read))
        events = Journal(self.board.dispatcher.cache.session, target).replay()
        harvested = self._harvest(target)
        if monitor is not None:
            stream = bool(events or harvested)
            watched = _Watched(handle=target, host=host, stream=stream)
            report = monitor.once(scope=self.__scope(watched))
            if not stream and report.resumed:
                resumed = report.resumed[0]
                return self.of(resumed.handle, host=resumed.target, run=run)
            return self.of(target, host=host, run=run)
        if events or harvested:
            recorded = _eventful(events)
            seen = {(trial.target, trial.handle) for trial in recorded}
            missing = tuple(
                self.__floor(record, job=job)
                for record in self.board.dispatcher.cache.listed("tracked")
                for stream, job in [streamed(record.name or "", handle=record.handle)]
                if stream == target and (record.target, record.handle) not in seen
            )
            found = (*self._swept(recorded), *missing, *harvested)
            return StreamVerdict(
                stream=target,
                trials=_qualified(found, events),
                note=Verdicts._silent(target, events, found),
            )
        return self._handled(target, host=host)

    def __pass(
        self,
        watched: _Watched,
        monitor: Monitor,
        vigil: Vigil,
        *,
        patience: Patience,
        say: Callable[[str], None],
    ) -> StreamVerdict | None:
        """One pass of a wait: settle what ended, say what changed; the answer once it stalls."""
        scope = self.__scope(watched)
        report = monitor.once(scope=scope)
        if report.busy is not None:
            # The holder moves the states being waited on, so they are cached here: it is said
            # every pass in place of a heartbeat for a job that may already have ended.
            say(report.busy.note)
            sleep(patience.interval)
            return None
        if not watched.stream and report.resumed:
            watched.handle, watched.host = report.resumed[0].handle, report.resumed[0].target
        for down in report.unreachable_hosts:
            # Said once: a wait on a run whose host went silent otherwise reads as a job that
            # is merely slow, until the timeout.
            if down.host not in watched.silent:
                watched.silent.add(down.host)
                asked = self.board.declares(down.host)
                say(f"{down.host} is not answering: {down.reason}" if asked else down.reason)
        if self.__settled(watched) is not None:
            return None
        return self.__looked(watched, vigil, patience=patience)

    def __registered(self, trial: TrialVerdict) -> TrialVerdict:
        """`trial` under its registry row: its provenance always, its outcome while it flies.

        Provenance joins settled rows too, since what a row was measured from stays true after
        the job ends and a mirror has no history to read it from later. The outcome joins only
        rows the receipts left in flight: a settled line carries a detail and exit code the
        registry has no column for, written by the pass that read the same probe.
        """
        try:
            record = self.board.dispatcher.cache.run(trial.handle, trial.target or None)
        except LookupError:
            return trial
        outcome = (
            {
                "state": record.state or trial.state,
                "verdict": record.verdict or trial.verdict,
                "exit_code": record.exit_code,
            }
            if trial.code == _IN_FLIGHT
            else {}
        )
        joined = trial.model_copy(
            update={"commit": record.commit, "digest": record.digest, **outcome}
        )
        return self.__explained(_delivery(joined, record), record)

    def __released(self, record: RunRecord, *, ended: str, exit_code: int | None) -> None:
        """Stop a live run once its evidence is kept, and record the verdict after the release.

        A stop its scheduler refused raises before any verdict is recorded.
        """
        cache = self.board.dispatcher.cache
        run = self.board.job(record.handle, host=record.target)
        settlement = Settlement(self.board)
        receipts = Verdicts.__preserved(record, run, settlement)
        verdict = (
            record.verdict
            if record.verdict is not None and record.verdict in vocabulary.Word.settled()
            else ended
        )
        code = record.exit_code if exit_code is None else exit_code
        word = (
            vocabulary.Word.CANCELLED
            if ended == vocabulary.Word.CANCELLED
            else vocabulary.Word.FINISHED
        )
        state = JobState(handle=record.handle, state=word, exit_code=code, verdict=verdict)
        unheard = self._killed(run, settlement)
        stored = cache.resolve(record, word, code, verdict)
        if settlement.release(run):
            if cache.run(record.handle, record.target).evidence == "copied":
                settlement.evidence(record, receipts, status="verified")
            settlement.track(record, state, detail=Verdicts._stopped(ended, code) + unheard)
            cache.report(stored, verdict)

    def __scope(self, watched: _Watched) -> Callable[[RunRecord], bool]:
        """Match a whole stream or one host-qualified submission, never a recycled handle."""
        handle = watched.handle
        if watched.stream:
            return lambda record: streamed(record.name or "", handle=record.handle)[0] == handle
        selected = self.record(handle, host=watched.host)
        return lambda record: (
            record.handle == selected.handle
            and record.target == selected.target
            and record.submitted_at == selected.submitted_at
        )

    def __settled(self, watched: _Watched) -> StreamVerdict | None:
        """The watched run's final answer, None while any of it is still in flight.

        A batch has no record of its own and settles when its stream verdict does.
        """
        if watched.stream:
            answered = self.of(watched.handle)
            return None if answered.code == _IN_FLIGHT else answered
        if self.record(watched.handle, host=watched.host).reported in vocabulary.Word.settled():
            return self._handled(watched.handle, host=watched.host)
        return None

    def __standing(self, watched: _Watched) -> StreamVerdict:
        """What the watched run reads as right now, settled or not."""
        if watched.stream:
            return self.of(watched.handle)
        return self._handled(watched.handle, host=watched.host)

    def __unheld(self, record: RunRecord) -> None:
        """Cancel a held request: nothing ever took it, so the sweep stops offering it."""
        cache = self.board.dispatcher.cache
        cancelled = vocabulary.Word.CANCELLED
        cache.report(cache.resolve(record, cancelled, None, cancelled), cancelled)

    def __unprepared(self, record: RunRecord) -> None:
        """Cancel a creation never attempted, refusing one that changed meanwhile."""
        try:
            self.board.dispatcher.cache.creations.prepared(record, vocabulary.Word.CANCELLED)
        except ValueError as changed:
            raise MissionError(
                f"creation {record.creation} changed during cancellation; reconcile "
                "its provider label before retrying"
            ) from changed

    def __unreached(self, record: RunRecord) -> bool:
        """Settle a run that has no machine to stop, and answer whether it had none.

        That is a creation never confirmed (refused: its provider must be reconciled first),
        one never attempted, a held request, a run already settled and reported, and a run whose
        host is no longer declared.
        """
        unreached: dict[str, Callable[[RunRecord], None]] = {
            vocabulary.Word.SUBMITTING: self.__unconfirmed,
            vocabulary.Word.PREPARED: self.__unprepared,
            vocabulary.Word.HELD: self.__unheld,
        }
        if (settle := unreached.get(record.verdict or "")) is not None:
            settle(record)
            return True
        if record.verdict in vocabulary.Word.settled() and record.reported == record.verdict:
            return True
        if not self.board.declares(record.target):
            self._abandon(record)
            return True
        return False

    def _abandon(self, record: RunRecord) -> None:
        """Settle a run whose host the manifest no longer declares, asking that host nothing.

        Its output went with the machine, so the evidence is recorded unverified and says why.
        """
        cache = self.board.dispatcher.cache
        settlement = Settlement(self.board)
        detail = f"cancelled: {record.target} is no longer declared, so its output is lost"
        settlement.evidence(record, (), status="unverified", detail=detail)
        stored = cache.resolve(record, vocabulary.Word.CANCELLED, None, vocabulary.Word.CANCELLED)
        verdict = stored.verdict or vocabulary.Word.CANCELLED
        state = JobState(
            handle=record.handle, state=stored.state, exit_code=stored.exit_code, verdict=verdict
        )
        settlement.track(record, state, detail=detail)
        cache.report(stored, verdict)

    def _conclude(self, handle: str, *, host: str = "", session: int) -> StreamVerdict:
        """Settle a run whose pytest session ended while its process did not, on the session.

        The outcome is written and the process only fails to exit while holding an allocation, so
        it is stopped the way a cancel stops it, evidence first, and settled `ok` or `failed` on
        the session's exit status rather than `cancelled`. Under another process's claim it is
        left for the next look.

        session: the pytest session's exit status, as its beacon reported it.
        """
        try:
            claim = self.board.dispatcher.cache.settlement.acquire(timeout=_SETTLEMENT_SECONDS)
        except Timeout:
            return self._handled(handle, host=host)
        ended = vocabulary.Word.OK if session == 0 else vocabulary.Word.FAILED
        with claim:
            return self._stop(handle, host=host, ended=ended, exit_code=session)

    def _handled(self, handle: str, *, host: str = "") -> StreamVerdict:
        """The settled truth of one dispatched run, its registry row under its receipts."""
        try:
            record = self.board.dispatcher.cache.run(handle, host or None)
        except LookupError as missing:
            raise MissionError(
                f"{handle!r} is not a receipts file, a stream, or a recorded handle: {missing}"
            ) from None
        stream, job = streamed(record.name or "", handle=record.handle)
        history = Journal(self.board.dispatcher.cache.session, stream).replay()
        events = self.__events(history, record)
        mine = tuple(trial for trial in _eventful(events) if trial.handle == record.handle)
        cases = {
            case for event in events if event.topic == Topic.EVIDENCE for case in _cases(event)
        }
        harvested = tuple(
            trial for trial in self._harvest(stream) if (trial.run, trial.job) in cases
        )
        floor = self._swept(mine) or (self.__floor(record, job=job),)
        return StreamVerdict(stream=stream, trials=_qualified((*floor, *harvested), events))

    def _harvest(self, stream: str) -> tuple[TrialVerdict, ...]:
        """The trial receipts a settle brought home for `stream`, empty when it brought none.

        They are kept apart from the stream's events, so no reader of the envelopes ever meets a
        line it was never promised.
        """
        return _parsed(self.kept.receipts(stream))

    def _stop(
        self,
        handle: str,
        *,
        host: str = "",
        ended: str = vocabulary.Word.CANCELLED,
        exit_code: int | None = None,
    ) -> StreamVerdict:
        """Preserve available evidence, stop the run, and advance the cursor only after release.

        A cancel may discard incomplete work but records that loss rather than calling the
        evidence complete. A held dispatch, and a run whose host is no longer declared, have no
        machine to contact. A stop its scheduler refused raises before any verdict is recorded,
        so the run is still in flight for the next ask.

        ended: the verdict a run still in flight settles on.
        exit_code: the exit status that verdict carries, the recorded one when None.
        """
        record = self.record(handle, host=host)
        if not self.__unreached(record):
            self.__released(record, ended=ended, exit_code=exit_code)
        return self._handled(handle, host=host)

    def _swept(self, trials: Iterable[TrialVerdict]) -> tuple[TrialVerdict, ...]:
        """`trials` joined onto the run registry: their provenance, and any in-flight outcome.

        A batched job's settled line is published only by its batch's watch; the unattended sweep
        probes, pulls the log home and memoizes the verdict in the registry but tells no stream.
        Reading the stream alone showed thirteen finished miyabi-g jobs as running beside their
        thirteen pulled logs (gigatoken-shootout rep92, 2026-09-04).
        """
        return tuple(self.__registered(trial) if trial.handle else trial for trial in trials)

    def cancel(self, handle: str, *, host: str = "") -> StreamVerdict:
        """Cancel under the same claim as automatic settlement, preserving cleanup failures.

        The claim wait is bounded: a sweep or a dispatch still landing holds it for minutes, and
        an unbounded cancel slept seven minutes on 2026-09-21 on an instance already gone.
        """
        try:
            claim = self.board.dispatcher.cache.settlement.acquire(timeout=_SETTLEMENT_SECONDS)
        except Timeout:
            raise MissionError(
                f"another mainboard process has held settlement for {_SETTLEMENT_SECONDS:g}s, a "
                "sweep or a dispatch still landing; nothing was cancelled, ask again when it ends"
            ) from None
        with claim:
            return self._stop(handle, host=host)

    def captured(self, handle: str, *, host: str = "") -> str:
        """`handle`'s output: the tail a settle brought home, else whatever the backend still has.

        The stored copy wins because it is the one that still exists: a settled run's log lives
        under a state dir a cleanup takes, or on a rented disk destroyed with the rental. The live
        read still answers for a run that ended before this workspace captured anything.
        """
        record = self.record(handle, host=host)
        stream, _ = streamed(record.name or "", handle=record.handle)
        stored = self.kept.transcript(stream, handle=record.handle)
        if stored is not None:
            return stored
        return self.board.job(record.handle, host=record.target).transcript() or ""

    def of(self, target: str, *, host: str = "", run: str = "") -> StreamVerdict:
        """The settled truth of `target`: a receipts store, file, stream, or handle, in that order.

        host: the alias narrowing a handle recorded on several hosts.
        run: which run of a receipts store to score, its newest when empty; the other targets
            carry one run's evidence by construction.
        """
        return self.__of(target, host=host, run=run)

    def record(self, handle: str, *, host: str = "") -> RunRecord:
        """The run registry's row for `handle`, refusing a handle nothing ever dispatched."""
        try:
            return self.board.dispatcher.cache.run(handle, host or None)
        except LookupError as missing:
            raise MissionError(f"nothing to wait on: {missing}") from None

    def refresh(self, target: str, *, host: str = "", run: str = "") -> StreamVerdict:
        """Refresh only this dispatch or stream before reading it; receipt files stay offline."""
        return self.__of(target, host=host, run=run, monitor=self.board.monitor())

    def stored(self, path: Path, *, stream: str, run: str) -> StreamVerdict | None:
        """One run of the receipts store at `path` scored, None when `path` holds no store.

        A store holds every run a harness ever took, so read flat a lane broken in one campaign
        would condemn a clean re-run months later. The newest run answers unless `run` names one.

        path: the partition root or the evidence directory above it.
        stream: what the caller asked for, which the heading names.
        """
        if not path.is_dir() or not any(path.rglob("*.parquet")):
            return None
        store = Dataset.holding(path)
        if store is None:
            return None
        chosen = run or store.newest
        trials = tuple(_receipted(row) for row in store.rows(chosen))
        note = (
            ""
            if trials
            else f"{store.root} holds no receipts for run {chosen!r}; it holds {store.runs}"
        )
        return StreamVerdict(stream=f"{stream} run {chosen}", trials=trials, note=note)

    def wait(
        self,
        handle: str,
        *,
        host: str = "",
        run: str = "",
        patience: Patience = _PATIENCE,
        say: Callable[[str], None] = logger.debug,
    ) -> StreamVerdict:
        """Block until `handle` settles through the monitor's durable settlement path.

        Each pass selects only this dispatch or batch, while rental deadlines remain global.
        Waiting pulls results, releases rentals and writes receipts like an unattended sweep.
        Interruption stops this waiter, not the job or its
        billing. The answer's code is the normalized receipt outcome, not the process exit status.
        Between passes a vigil says each cell as it lands plus a heartbeat, settles a job whose
        pytest session ended while its process lingers, and stops with `STALLED` on a job silent
        past its stall threshold on an idle card.

        handle: the dispatched run, or a batch id as `submit --batch` printed it, which waits for
            every job of the batch and answers with the batch's verdict. Local receipt paths
            are read once without contacting hosts.
        run: which run to read when the target is a local receipts store.
        patience: when to give up with the run reported in flight (exit 2), and what silence
            on an idle card stops the wait.
        say: where the cells and the heartbeat go.
        """
        path = self.board.dispatcher.local(handle)
        stored = self.stored(path, stream=handle, run=run)
        if stored is not None:
            return stored
        if path.is_file():
            return self.of(handle, host=host, run=run)
        stream = bool(Journal(self.board.dispatcher.cache.session, handle).replay())
        watched = _Watched(handle=handle, host=host, stream=stream)
        deadline = monotonic() + patience.timeout if patience.timeout else None
        monitor = self.board.monitor()
        vigil = Vigil(Pulses(self.board), stall=patience.stall, say=say)
        # A caller re-reading a finished batch owes no remote probe.
        while (settled := self.__settled(watched)) is None:
            if deadline is not None and monotonic() >= deadline:
                return self.__standing(watched)
            if answer := self.__pass(watched, monitor, vigil, patience=patience, say=say):
                return answer
        return settled


def _eventful(events: Iterable[Event]) -> tuple[TrialVerdict, ...]:
    """Every job's settled row out of one stream's event envelopes.

    The cursor is `latest` per topic per job, the read every resumed pass uses, so a re-dispatched
    job answers with its newest run and a refused or quota-held job still has a row; `_joined`
    decides between those by the clock alone. A job never dispatched answers from its skip, so a
    `--only` wave's unselected jobs stay visible to the verb that reports the batch.
    """
    recorded = list(events)
    targets: dict[str, set[str]] = {}
    for event in recorded:
        if event.topic == Topic.SUBMITTED:
            handle = str(event.data.get("handle", ""))
            targets.setdefault(handle, set()).add(str(event.data.get("target", "")))
    identified = [
        event
        for event in recorded
        if event.data.get("target")
        or len(targets.get(str(event.data.get("handle", "")), set()) - {""}) <= 1
    ]
    answers = receipts_module.offered(recorded)
    skipped = latest(recorded, Topic.SKIPPED)
    states = latest(identified, Topic.STATE)
    settled = latest(identified, Topic.SETTLED)
    attested = latest(recorded, Topic.ATTESTED)
    return tuple(
        _joined(
            job,
            answer=answers[job],
            state=states.get(job),
            ended=settled.get(job),
            attestation=attested.get(job),
        )
        if job in answers
        else _unselected(job, skipped[job])
        for job in dict.fromkeys([*answers, *skipped])
    )


def _qualified(
    trials: Iterable[TrialVerdict], events: Iterable[Event]
) -> tuple[TrialVerdict, ...]:
    """Overlay append-only delivery corrections; leave original computation receipts intact."""
    statuses = {
        (
            str(event.data.get("handle", "")),
            str(event.data.get("target", "")),
            str(event.data.get("submitted_at", "")),
        ): event
        for event in sorted(events, key=lambda event: event.at)
        if event.topic == Topic.EVIDENCE
    }
    handles = {
        (str(event.data.get("target", "")), str(event.data.get("handle", ""))): event
        for event in statuses.values()
    }
    cases = {case: event for event in statuses.values() for case in _cases(event)}
    return tuple(
        _corrected(
            trial,
            handles.get((trial.target, trial.handle))
            or handles.get(("", trial.handle))
            or cases.get((trial.run, trial.job)),
        )
        for trial in trials
    )


def _corrected(trial: TrialVerdict, status: Event | None) -> TrialVerdict:
    """`trial` under its newest delivery status, withheld unless that status is benign."""
    if status is None or status.data.get("status") in {"verified", "not_started"}:
        return trial
    word = "unverified" if status.data.get("status") == "unverified" else "blocked"
    detail = str(status.data.get("detail") or _INCOMPLETE)
    return trial.model_copy(update={"verdict": word, "detail": detail})


def _cases(event: Event) -> list[tuple[str, str]]:
    """The (run, case) pairs an evidence line names, skipping any entry torn out of shape."""
    group = event.data.get("trials")
    if not isinstance(group, list):
        return []
    return [
        (str(case[0]), str(case[1])) for case in group if isinstance(case, list) and len(case) == 2
    ]


def _parsed(
    lines: Iterable[str], corrections: Iterable[Event] = (), *, source: str = "the lake"
) -> tuple[TrialVerdict, ...]:
    """Every row `lines` hold, in either written shape, under `corrections` and their own events.

    source: where the lines came from, named when one is neither shape.
    """
    events: list[Event] = []
    trials: list[TrialVerdict] = []
    for payload in filter(None, map(_payload, lines)):
        if _RECEIPT in payload:
            trials.append(_receipted(payload[_RECEIPT]))
            continue
        try:
            events.append(Event.model_validate(payload))
        except ValidationError:
            logger.debug("{} carries a line that is neither shape this verb reads", source)
    return _qualified((*_eventful(events), *trials), [*events, *corrections])


def _payload(line: str) -> dict[str, JsonValue] | None:
    """`line` as a JSON object, None for a blank line, broken JSON or any other value."""
    try:
        payload = json.loads(line) if line.strip() else None
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _receipted(payload: JsonValue) -> TrialVerdict:
    """One printed `trial_receipt` payload as a settled row, every field read leniently.

    The contract makes `case_id`, `outcome`, `producer`, `node` and `gates` optional and lets a
    harness add its own, so an absent field is an empty cell and a non-mapping an empty row.
    `run_id` is read ONLY where `case_id` is absent: the trials harness spelled the test case
    `run_id` beside a `run` column holding the run, so a row carrying both is a new row and the
    old meaning never enters a new join.
    """
    data = payload if isinstance(payload, dict) else {}
    return TrialVerdict(
        job=str(data.get("case_id") or data.get("run_id", "")),
        run=str(data.get("run", "")),
        node=str(data.get("node", "")),
        verdict=str(data.get("outcome", "")) or vocabulary.Word.OK,
        settled=str(data.get("verdict", "")),
        detail=str(data.get("reason", "")),
        gates=_gated(data.get("gates")),
        producer=str(data.get("producer", "")),
    )


def _gated(sweep: JsonValue) -> str:
    """A trial receipt's gate sweep as one cell, the first non-passing gate named.

    sweep: the receipt's `gates` list, entries of `status` and `reason`.
    """
    if not isinstance(sweep, list) or not sweep:
        return ""
    checks = [entry for entry in sweep if isinstance(entry, dict)]
    for check in checks:
        status = str(check.get("status", ""))
        if status and status != "passed":
            return f"{status}: {check.get('reason', '')}"
    return f"{len(checks)} passed"


def _delivery(trial: TrialVerdict, record: RunRecord) -> TrialVerdict:
    """Fail closed on the cache checkpoint even if its next event was never published."""
    if record.evidence not in {"pending", "copied", "unverified"}:
        return trial
    unverified = record.evidence == "unverified"
    return trial.model_copy(
        update={
            "verdict": "unverified" if unverified else "blocked",
            "detail": "evidence is unverified" if unverified else _INCOMPLETE,
        }
    )


def _contention(attestation: Event | None) -> str:
    """What a job's attestation says it started under, empty for an idle node or no attestation.

    Only the unwelcome half is rendered, so a column of `idle` never buries the row that matters,
    and the busy figure rides along so a reader can weigh it.
    """
    if attestation is None or attestation.data.get("idle"):
        return ""
    return f"gpu {attestation.data.get('gpu_pct', 0)}% busy at start"


def _unselected(job: str, skip: Event) -> TrialVerdict:
    """One job a wave was told to leave out, as a row that is already over.

    A skip says no offer was made in this wave, so it decides a row only when nothing was ever
    dispatched for the job: nine jobs run at 13:43 and left out of an 18:43 `--only` wave
    (2026-09-04) are runs that happened, which ranking the skip by its clock would have thrown
    away. It is shown so a plan worked in waves reads against the plan, and terminal because
    nothing never dispatched can move, so a completion check neither waits on it nor fails it.
    """
    return TrialVerdict(
        job=job,
        target=str(skip.data.get("target", "")),
        state=vocabulary.Word.SKIPPED,
        verdict=vocabulary.Word.SKIPPED,
        detail=str(skip.data.get("reason", "")),
    )


def _joined(
    job: str,
    *,
    answer: Event,
    state: Event | None,
    ended: Event | None,
    attestation: Event | None = None,
) -> TrialVerdict:
    """One job's row, folded from the newest answer about it and its latest line per topic.

    Taken, refused and held on a quota answer the same offer, so the newest stands whatever its
    topic, the shared `OFFERED` cursor's ranking: four jobs miyabi-g's `njobs-g` limit refused at
    13:43 went out at 18:43 (2026-09-04), and reading the refusal for being a refusal buried the
    younger run. A refusal after a submission is terminal, in the target's words. Past the answer
    a settled line wins the verdict and a state line stands in while the job flies; both must
    match the submission, so a stale settlement never silences a re-dispatched run. An
    attestation rides on every row, true of the finished measurement as of the running one.
    """
    target = str(answer.data.get("target", ""))
    if answer.topic in (Topic.HELD, Topic.REFUSED):
        held = answer.topic is Topic.HELD
        return TrialVerdict(
            job=job,
            target=target,
            state=vocabulary.Word.HELD if held else "",
            verdict=vocabulary.Word.HELD if held else "refused",
            detail=str(answer.data.get("reason", "")),
        )
    state = _matching(state, answer)
    ended = _matching(ended, answer)
    flying = str(state.data.get("verdict", "")) if state else ""
    row = TrialVerdict(
        job=job,
        handle=str(answer.data.get("handle", "")),
        target=target,
        node=str(answer.data.get("node", "")),
        state=str(state.data.get("state", "")) if state else "",
        verdict=flying or vocabulary.Word.RUNNING,
        contended=_contention(attestation),
    )
    if ended is None:
        return row
    code = ended.data.get("exit_code")
    return row.model_copy(
        update={
            "verdict": str(ended.data.get("verdict", "")),
            "exit_code": code if isinstance(code, int) else None,
            "detail": str(ended.data.get("detail", "")),
        }
    )


def _matching(event: Event | None, submission: Event) -> Event | None:
    """Keep a state only when its identity and time belong to this submission."""
    if event is None or event.at < submission.at:
        return None
    for field in ("handle", "target", "submitted_at"):
        expected = submission.data.get(field)
        observed = event.data.get(field)
        if (field == "handle" or (expected and observed)) and expected != observed:
            return None
    return event
