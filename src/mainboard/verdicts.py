# The anti-fabrication read behind `mainboard verdict` and the block behind `mainboard wait`.
# Everything printed here comes from on-disk receipts and the durable run registry, never from a
# dashboard, a digest or a live session's memory: a notification says a job probably ended, this
# module reads its outcome.
#
# Three targets resolve to one settled view. A receipts file is read line by line in both shapes
# the workspace writes, the batch `Event` envelope and the printed `trial_receipt` line. A stream
# id reads that stream's `events.ndjson`. A handle resolves through the run registry to the stream
# its dispatch was tracked under. Under every target the registry row is the floor: a row the
# receipts left in flight is re-read from the registry, where the unattended sweep records an
# outcome nobody was watching for, and a run whose workspace tracks nothing still answers.

import json
from time import monotonic, sleep
from typing import TYPE_CHECKING

from filelock import Timeout
from patos import FrozenModel
from pydantic import ValidationError

from .batch.receipts import OFFERED, Event, Receipts, Topic, latest
from .batch.runner import directory
from .core.errors import MissionError
from .diagnosis import reason
from .dispatch import vocabulary
from .dispatch.schedulers import short_reason
from .dispatch.shared import logger
from .dispatch.vocabulary import JobState
from .pulse import Pulses
from .tracking import streamed
from .vigil import STALL_SECONDS, Vigil

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pydantic import JsonValue

    from .board import Board
    from .dispatch.state import RunRecord

# The key a printed trial receipt carries its payload under, spelled here rather than imported so
# reading a receipt never drags the lab machinery in.
_RECEIPT = "trial_receipt"

# The code while any row is still running or legitimately waiting, the answer a waiter loops on.
_IN_FLIGHT = 2
# A wait whose job went silent on an idle card, distinct from a timeout's 2 so a script can tell
# a job still working from one that stopped doing anything.
STALLED = 4
# How long a cancel waits for the settlement claim another process holds before it refuses.
SETTLEMENT_SECONDS = 120.0
# Settled word to exit code: 0 ok, 1 failed, 2 in flight, 3 vanished or unknown. A cancel exits 1
# because a completion check must never call a stopped run complete, however deliberate the stop;
# a skip exits 0 because nothing was dispatched, so no run is incomplete. A quota hold is 2: the
# sweep still offers it, which keeps a batch of thirteen from reading as a finished nine.
_EXITS = {
    vocabulary.OK: 0,
    "passed": 0,
    vocabulary.FAILED: 1,
    "refused": 1,
    vocabulary.TIMEOUT: 1,
    vocabulary.CANCELLED: 1,
    vocabulary.RUNNING: 2,
    vocabulary.HELD: 2,
    vocabulary.SKIPPED: 0,
    "blocked": 2,
    "": 2,
    vocabulary.QUEUED: 2,
    vocabulary.PREPARED: 2,
    vocabulary.SUBMITTING: 2,
}
_INCOMPLETE = "evidence settlement is incomplete"


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
            return STALLED
        if _IN_FLIGHT in codes:
            return _IN_FLIGHT
        if 3 in codes or not codes:
            return 3
        return 0


class Verdicts:
    """The receipts-derived outcomes of a workspace's runs, read fresh on every ask."""

    def __init__(self, board: Board) -> None:
        self.board = board

    def cancel(self, handle: str, *, host: str = "") -> StreamVerdict:
        """Cancel under the same claim as automatic settlement, preserving cleanup failures.

        The claim wait is bounded: a sweep or a dispatch still landing holds it for minutes, and
        an unbounded cancel slept seven minutes on 2026-09-21 on an instance already gone.
        """
        try:
            claim = self.board.dispatcher.cache.settlement.acquire(timeout=SETTLEMENT_SECONDS)
        except Timeout:
            raise MissionError(
                f"another mainboard process has held settlement for {SETTLEMENT_SECONDS:g}s, a "
                "sweep or a dispatch still landing; nothing was cancelled, ask again when it ends"
            ) from None
        with claim:
            return self._stop(handle, host=host)

    def conclude(self, handle: str, *, host: str = "", session: int) -> StreamVerdict:
        """Settle a run whose pytest session ended while its process did not, on the session.

        The outcome is written and the process only fails to exit while holding an allocation, so
        it is stopped the way a cancel stops it, evidence first, and settled `ok` or `failed` on
        the session's exit status rather than `cancelled`. Under another process's claim it is
        left for the next look.

        session: the pytest session's exit status, as its beacon reported it.
        """
        try:
            claim = self.board.dispatcher.cache.settlement.acquire(timeout=SETTLEMENT_SECONDS)
        except Timeout:
            return self.handled(handle, host=host)
        ended = vocabulary.OK if session == 0 else vocabulary.FAILED
        with claim:
            return self._stop(handle, host=host, ended=ended, exit_code=session)

    def _stop(
        self,
        handle: str,
        *,
        host: str = "",
        ended: str = vocabulary.CANCELLED,
        exit_code: int | None = None,
    ) -> StreamVerdict:
        """Preserve available evidence, stop the run, and advance the cursor only after release.

        A cancel may discard incomplete work but records that loss rather than calling the
        evidence complete. A held dispatch, and a run whose host is no longer declared, have no
        machine to contact.

        ended: the verdict a run still in flight settles on.
        exit_code: the exit status that verdict carries, the recorded one when None.
        """
        cache = self.board.dispatcher.cache
        record = self.record(handle, host=host)
        if record.verdict == vocabulary.SUBMITTING:
            raise MissionError(
                f"creation {record.creation} has no confirmed provider handle; reconcile "
                "its provider label before cancellation; absence from one listing is not proof"
            )
        if record.verdict in vocabulary.TERMINAL and record.reported == record.verdict:
            return self.handled(handle, host=host)
        if record.verdict == vocabulary.PREPARED:
            try:
                cache.leave_prepared(record, vocabulary.CANCELLED)
            except ValueError as changed:
                raise MissionError(
                    f"creation {record.creation} changed during cancellation; reconcile "
                    "its provider label before retrying"
                ) from changed
            return self.handled(handle, host=host)
        if record.verdict == vocabulary.HELD:
            # Nothing ever took it, so cancelling is the sweep never offering the request again.
            cache.report(
                cache.resolve(record, vocabulary.CANCELLED, None, vocabulary.CANCELLED),
                vocabulary.CANCELLED,
            )
            return self.handled(handle, host=host)
        if not self.board.declares(record.target):
            self._abandon(record)
            return self.handled(handle, host=host)
        run = self.board.job(record.handle, host=record.target)
        monitor = self.board.monitor()
        receipts: tuple[str, ...] = ()
        try:
            receipts = monitor.capture(record, run)
            monitor.verify(record, run, monitor.pull(run), receipts)
        except (MissionError, OSError, ValueError) as fault:
            monitor.evidence(record, receipts, status="unverified", detail=f"cancelled: {fault}")
        else:
            monitor.evidence(record, receipts, status="copied")
        verdict = (
            record.verdict
            if record.verdict is not None and record.verdict in vocabulary.TERMINAL
            else ended
        )
        code = record.exit_code if exit_code is None else exit_code
        word = vocabulary.CANCELLED if ended == vocabulary.CANCELLED else vocabulary.FINISHED
        state = JobState(handle=record.handle, state=word, exit_code=code, verdict=verdict)
        stored = cache.resolve(record, word, code, verdict)
        run.kill()
        if monitor.release(run):
            if cache.run(handle, record.target).evidence == "copied":
                monitor.evidence(record, receipts, status="verified")
            monitor.track(record, state, detail=stopped(ended, code))
            cache.report(stored, verdict)
        return self.handled(handle, host=host)

    def _abandon(self, record: RunRecord) -> None:
        """Settle a run whose host the manifest no longer declares, asking that host nothing.

        Its output went with the machine, so the evidence is recorded unverified and says why.
        """
        cache = self.board.dispatcher.cache
        monitor = self.board.monitor()
        detail = f"cancelled: {record.target} is no longer declared, so its output is lost"
        monitor.evidence(record, (), status="unverified", detail=detail)
        stored = cache.resolve(record, vocabulary.CANCELLED, None, vocabulary.CANCELLED)
        verdict = stored.verdict or vocabulary.CANCELLED
        state = JobState(
            handle=record.handle, state=stored.state, exit_code=stored.exit_code, verdict=verdict
        )
        monitor.track(record, state, detail=detail)
        cache.report(stored, verdict)

    def captured(self, handle: str, *, host: str = "") -> str:
        """`handle`'s output: the tail a settle brought home, else whatever the backend still has.

        The stored copy wins because it is the one that still exists: a settled run's log lives
        under a state dir a cleanup takes, or on a rented disk destroyed with the rental. The live
        read still answers for a run that ended before this workspace captured anything.
        """
        record = self.record(handle, host=host)
        stream, _ = streamed(record.name or "", handle=record.handle)
        stored = directory(self.board, stream) / f"{record.handle}.log"
        if stored.is_file():
            return stored.read_text(encoding="utf-8")
        return self.board.job(record.handle, host=record.target).transcript()

    def handled(self, handle: str, *, host: str = "") -> StreamVerdict:
        """The settled truth of one dispatched run, its registry row under its receipts."""
        try:
            record = self.board.dispatcher.cache.run(handle, host or None)
        except LookupError as missing:
            raise MissionError(
                f"{handle!r} is not a receipts file, a stream, or a recorded handle: {missing}"
            ) from None
        stream, job = streamed(record.name or "", handle=record.handle)
        under = directory(self.board, stream)
        stream_file = under / "events.ndjson"
        history = Receipts(stream_file).replay() if stream_file.is_file() else []
        events = self.__events(history, record)
        mine = tuple(trial for trial in eventful(events) if trial.handle == record.handle)
        cases = {
            case for event in events if event.topic == Topic.EVIDENCE for case in _cases(event)
        }
        harvested = tuple(trial for trial in harvest(under) if (trial.run, trial.job) in cases)
        floor = self.swept(mine) or (self.__floor(record, job=job),)
        return StreamVerdict(stream=stream, trials=qualified((*floor, *harvested), events))

    @staticmethod
    def __events(events: list[Event], record: RunRecord) -> list[Event]:
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

    def of(self, target: str, *, host: str = "", run: str = "") -> StreamVerdict:
        """The settled truth of `target`: a receipts store, file, stream, or handle, in that order.

        host: the alias narrowing a handle recorded on several hosts.
        run: which run of a receipts store to score, its newest when empty; the other targets
            carry one run's evidence by construction.
        """
        path = self.board.dispatcher.local(target)
        stored = self.stored(path, stream=target, run=run)
        if stored is not None:
            return stored
        if path.is_file():
            read = lined(path)
            return StreamVerdict(stream=target, trials=read, note=unreadable(path, read))
        under = directory(self.board, target)
        stream_file = under / "events.ndjson"
        if stream_file.is_file() or (under / "receipts.ndjson").is_file():
            events = Receipts(stream_file).replay()
            recorded = eventful(events)
            seen = {(trial.target, trial.handle) for trial in recorded}
            missing = tuple(
                self.__floor(record, job=job)
                for record in self.board.dispatcher.cache.tracked()
                for stream, job in [streamed(record.name or "", handle=record.handle)]
                if stream == target and (record.target, record.handle) not in seen
            )
            found = (*self.swept(recorded), *missing, *harvest(under))
            return StreamVerdict(
                stream=target, trials=qualified(found, events), note=unreadable(stream_file, found)
            )
        return self.handled(target, host=host)

    def record(self, handle: str, *, host: str = "") -> RunRecord:
        """The run registry's row for `handle`, refusing a handle nothing ever dispatched."""
        try:
            return self.board.dispatcher.cache.run(handle, host or None)
        except LookupError as missing:
            raise MissionError(f"nothing to wait on: {missing}") from None

    def stored(self, path: Path, *, stream: str, run: str) -> StreamVerdict | None:
        """One run of the receipts store at `path` scored, None when `path` holds no store.

        A store holds every run a harness ever took, so read flat a lane broken in one campaign
        would condemn a clean re-run months later. The newest run answers unless `run` names one.
        The reader is imported here because its dataframe engine costs a fifth of a second and
        this module is on every command's path.

        path: the partition root or the evidence directory above it.
        stream: what the caller asked for, which the heading names.
        """
        from .trials.dataset import Dataset

        store = Dataset.holding(path)
        if store is None:
            return None
        chosen = run or store.newest
        trials = tuple(receipted(row) for row in store.rows(chosen))
        note = (
            ""
            if trials
            else f"{store.root} holds no receipts for run {chosen!r}; it holds {store.runs}"
        )
        return StreamVerdict(stream=f"{stream} run {chosen}", trials=trials, note=note)

    def swept(self, trials: tuple[TrialVerdict, ...]) -> tuple[TrialVerdict, ...]:
        """`trials` joined onto the run registry: their provenance, and any in-flight outcome.

        A batched job's settled line is published only by its batch's watch; the unattended sweep
        probes, pulls the log home and memoizes the verdict in the registry but tells no stream.
        Reading the stream alone showed thirteen finished miyabi-g jobs as running beside their
        thirteen pulled logs (gigatoken-shootout rep92, 2026-09-04).
        """
        return tuple(self.__registered(trial) if trial.handle else trial for trial in trials)

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
        return self.__explained(delivery(joined, record), record)

    def __floor(self, record: RunRecord, *, job: str) -> TrialVerdict:
        """`record`'s own row, the answer for a run whose stream holds no receipts at all."""
        return self.__explained(registered(record, job=job), record)

    def __explained(self, trial: TrialVerdict, record: RunRecord) -> TrialVerdict:
        """A failed `trial` with its cause read off the log the sweep brought home."""
        if trial.code != 1:
            return trial
        return trial.model_copy(update={"cause": reason(self.board, record)})

    def wait(
        self,
        handle: str,
        *,
        host: str = "",
        timeout: float = 0.0,
        interval: float = vocabulary.POLL_SECONDS,
        stall: float = STALL_SECONDS,
        say: Callable[[str], None] = logger.debug,
        poll: Callable[[float], None] = sleep,
    ) -> StreamVerdict:
        """Block until `handle` settles, sweeping the same durable path the monitor cron runs.

        Every pass is one `Monitor.once`, so waiting pulls results, releases rentals and writes
        receipts like an unattended sweep. Interruption stops this waiter, not the job or its
        billing. The answer's code is the normalized receipt outcome, not the process exit status.
        Between passes a vigil says each cell as it lands plus a heartbeat, settles a job whose
        pytest session ended while its process lingers, and stops with `STALLED` on a job silent
        past `stall` on an idle card.

        handle: the dispatched run, or a batch id as `batch run` printed it, which waits for
            every job of the batch and answers with the batch's verdict.
        timeout: wall seconds before giving up with the run reported in flight (exit 2), 0 never.
        interval: seconds between sweeps.
        stall: seconds of silence on an idle card that stop the wait, 0 never.
        say: where the cells and the heartbeat go.
        """
        deadline = monotonic() + timeout if timeout else None
        monitor = self.board.monitor()
        stream = (directory(self.board, handle) / "events.ndjson").is_file()
        vigil = Vigil(Pulses(self.board), stall=stall, say=say)
        # What already settled answers before any pass runs, since a pass settles the whole
        # workspace and a caller re-reading a finished batch owes it nothing.
        while (settled := self.__settled(handle, host=host, stream=stream)) is None:
            if deadline is not None and monotonic() >= deadline:
                return self.__standing(handle, host=host, stream=stream)
            monitor.once()
            if self.__settled(handle, host=host, stream=stream) is not None:
                continue
            look = vigil.look(self.__running(handle, host=host, stream=stream))
            for linger in look.lingering:
                self.conclude(linger.handle, host=linger.target, session=linger.session)
            if look.stalled:
                answer = self.__standing(handle, host=host, stream=stream)
                return answer.model_copy(update={"stalled": look.stalled})
            if not look.lingering:
                poll(interval)
        return settled

    def __standing(self, handle: str, *, host: str, stream: bool) -> StreamVerdict:
        """What `handle` reads as right now, settled or not."""
        return self.of(handle) if stream else self.handled(handle, host=host)

    def __running(self, handle: str, *, host: str, stream: bool) -> list[RunRecord]:
        """The dispatched runs behind `handle` that are running now, as the last pass left them."""
        cache = self.board.dispatcher.cache
        behind = (
            [
                record
                for record in cache.live()
                if streamed(record.name or "", handle=record.handle)[0] == handle
            ]
            if stream
            else [cache.run(handle, host or None)]
        )
        return [record for record in behind if record.verdict == vocabulary.RUNNING]

    def __settled(self, handle: str, *, host: str, stream: bool) -> StreamVerdict | None:
        """`handle`'s final answer, None while any of it is still in flight.

        A batch has no record of its own and settles when its stream verdict does.
        """
        if stream:
            answered = self.of(handle)
            return None if answered.code == _IN_FLIGHT else answered
        if self.record(handle, host=host).reported in vocabulary.TERMINAL:
            return self.handled(handle, host=host)
        return None


def stopped(ended: str, exit_code: int | None) -> str:
    """Why a stopped run settled where it did, the detail its settled receipt carries."""
    if ended == vocabulary.CANCELLED:
        return short_reason(vocabulary.CANCELLED, None)
    return f"pytest session ended with exit {exit_code}; the lingering process was stopped"


def eventful(events: Iterable[Event]) -> tuple[TrialVerdict, ...]:
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
    answers = latest(recorded, *OFFERED)
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


def unreadable(path: Path, trials: tuple[TrialVerdict, ...]) -> str:
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
        f"{path} holds {len(lines)} line(s), none of which is evidence this verb reads. It reads "
        "the batch event envelope and the `trial_receipt` line, so a harness writing another "
        "shape settles here as soon as it prints one of those two per trial"
    )


def harvest(under: Path) -> tuple[TrialVerdict, ...]:
    """The trial receipts a settle brought home for `under`'s stream, empty when it brought none.

    They live in their own file, not the event log, so no reader of the envelopes ever meets a
    line it was never promised.
    """
    path = under / "receipts.ndjson"
    return lined(path) if path.is_file() else ()


def qualified(
    trials: tuple[TrialVerdict, ...], events: Iterable[Event]
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


def lined(path: Path) -> tuple[TrialVerdict, ...]:
    """Every row a receipts file holds, whichever of the two written shapes each line is.

    An `Event` line joins its stream's per-job cursor; a `trial_receipt` line is one whole trial.
    A line that is neither is skipped, torn or whole, the tolerance the receipts replay extends to
    a torn log: another tool's well-formed evidence once reached `Event` and answered with a
    pydantic traceback instead of the empty table the caller could be told about.
    """
    events: list[Event] = []
    trials: list[TrialVerdict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if _RECEIPT in payload:
            trials.append(receipted(payload[_RECEIPT]))
            continue
        try:
            events.append(Event.model_validate(payload))
        except ValidationError:
            logger.debug("%s carries a line that is neither shape this verb reads", path)
    companion = path.parent / "events.ndjson"
    corrections = Receipts(companion).replay() if companion != path and companion.is_file() else []
    return qualified((*eventful(events), *trials), [*events, *corrections])


def receipted(payload: JsonValue) -> TrialVerdict:
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
        verdict=str(data.get("outcome", "")) or vocabulary.OK,
        settled=str(data.get("verdict", "")),
        detail=str(data.get("reason", "")),
        gates=gated(data.get("gates")),
        producer=str(data.get("producer", "")),
    )


def gated(sweep: JsonValue) -> str:
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


def registered(record: RunRecord, *, job: str) -> TrialVerdict:
    """The run registry's own row as a settled row, the floor a receiptless run answers from."""
    trial = TrialVerdict(
        job=job,
        handle=record.handle,
        target=record.target,
        node=record.node,
        state=record.state or "",
        verdict=record.verdict or vocabulary.RUNNING,
        exit_code=record.exit_code,
        commit=record.commit,
        digest=record.digest,
    )
    return delivery(trial, record)


def delivery(trial: TrialVerdict, record: RunRecord) -> TrialVerdict:
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


def contention(attestation: Event | None) -> str:
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
        state=vocabulary.SKIPPED,
        verdict=vocabulary.SKIPPED,
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
            state=vocabulary.HELD if held else "",
            verdict=vocabulary.HELD if held else "refused",
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
        verdict=flying or vocabulary.RUNNING,
        contended=contention(attestation),
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
