# The anti-fabrication read behind `mainboard verdict` and the block behind `mainboard wait`.
# Everything printed here is derived from on-disk receipts and the durable run registry, never
# from a dashboard, a digest or anything a live session remembered. A notification says a job
# probably ended; this module is where its outcome is actually read.
#
# Three targets resolve to one settled view. A receipts file is read line by line, accepting
# both shapes the workspace writes, the batch `Event` envelope and the printed `trial_receipt`
# line, so a study's events stream and a harness's own receipts file answer through one verb. A
# stream id reads the workspace's own `events.ndjson` for that stream. A handle resolves through
# the run registry to the stream its dispatch was tracked under, and the registry row itself is
# the floor the receipts overlay, so a run whose workspace tracks nothing still answers. That
# floor is under every target rather than only under a handle: a row the receipts left in flight
# is re-read from the registry, which is where the unattended sweep records an outcome nobody
# was watching for.

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
from .tracking import streamed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pydantic import JsonValue

    from .board import Board
    from .dispatch.state import RunRecord

# The key a printed trial receipt line carries its payload under, the shape any experiment
# harness may write; spelled here rather than imported so reading a receipt never drags the lab
# machinery in.
_RECEIPT = "trial_receipt"

# How a settled word maps to a process exit code, the same table `Verdict.code` answers from:
# 0 ok, 1 failed, 2 still running or legitimately waiting, 3 vanished or unknown. A cancel exits
# 1 because a completion check must never call a stopped run complete, however deliberate the
# stop was; the word in the row is what says it was a decision rather than a crash. A skip exits
# 0 for the opposite reason: nothing was ever dispatched, so there is no run to be incomplete,
# which is the same reading the batch's own closing count already takes of it.
# The code a stream answers while any of its rows is still running or legitimately waiting,
# named once because a waiter loops on exactly this answer.
_IN_FLIGHT = 2
# How long a cancel waits for the settlement claim another process holds before it refuses.
SETTLEMENT_SECONDS = 120.0
_EXITS = {
    vocabulary.OK: 0,
    "passed": 0,
    vocabulary.FAILED: 1,
    "refused": 1,
    vocabulary.TIMEOUT: 1,
    vocabulary.CANCELLED: 1,
    vocabulary.RUNNING: 2,
    # A dispatch a target's quota is holding has not run, has not failed, and is not settled: the
    # sweep is still offering it. A completion check must wait for it exactly as it waits for a
    # queued job, which is what keeps a batch of thirteen from reading as a finished nine.
    vocabulary.HELD: 2,
    vocabulary.SKIPPED: 0,
    "blocked": 2,
    "": 2,
    vocabulary.QUEUED: 2,
    vocabulary.PREPARED: 2,
    vocabulary.SUBMITTING: 2,
}


class TrialVerdict(FrozenModel):
    """One trial or job as its receipts left it.

    job: the trial's name inside its stream, or a trial receipt's run id.
    handle: the scheduler or provider handle, empty for an in-process trial.
    target: the alias it ran on, empty when the receipt never named one.
    node: the ledger slug the run serves, empty when none was declared.
    state: the scheduler's own word, empty when nothing reported one.
    verdict: the settled word, `running` while nothing terminal is on file.
    settled: the word a trial receipt's own vocabulary settled on, empty when it named none.
        Separate from `verdict` because only `verdict` is what an exit code is derived from: a
        harness may settle `refuted` or `abandoned` on a reading that was taken perfectly well,
        and a completion check must read that as the success it is.
    exit_code: the process exit status, when a receipt recorded one.
    detail: where results landed, why it failed, or a trial receipt's own reason.
    gates: the gate sweep summarized, empty when the receipts carry none.
    producer: the harness that stamped a trial receipt, empty for event streams.
    contended: what the machine was already doing when this run started, empty when it attested
        an idle node and empty when nothing attested at all. A cell here is the difference
        between a measurement and a measurement taken while another job held the GPU.
    cause: why a failed run failed, its own last meaningful output line, empty for a run that
        did not fail and for one whose output never came home. `detail` says a job failed and
        with what status; this says what it said on the way out.
    commit: the commit the dispatching tree was at, so a row measured on a mirror with no
        history still names one. Empty for a run this workspace did not dispatch.
    digest: the content digest of that tree, which is what a run seals against where there is no
        git to ask: the mirror carries the bytes and not the history, so the claim a preflight
        can still make is that these bytes are the ones the dispatch shipped.
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

    stream: the stream these rows were read from.
    trials: one row per trial or job, in first-seen order.
    note: why there are no rows, empty whenever there are. An empty table is the one answer a
        reader cannot act on, because nothing about it says whether the run has not started,
        the evidence went somewhere else, or the harness wrote a shape this verb does not read.
    """

    stream: str
    trials: tuple[TrialVerdict, ...]
    note: str = ""

    @property
    def code(self) -> int:
        """The one exit status a completion check branches on.

        A failure anywhere outranks everything, then anything still in flight, then a trial
        that vanished, and only a stream whose every row settled clean exits zero. An empty
        stream is unknown rather than clean, since receipts that do not exist prove nothing.
        """
        codes = {trial.code for trial in self.trials}
        if 1 in codes:
            return 1
        if _IN_FLIGHT in codes:
            return _IN_FLIGHT
        if 3 in codes or not codes:
            return 3
        return 0


class Verdicts:
    """The receipts-derived outcomes of a workspace's runs, read fresh on every ask."""

    def __init__(self, board: Board) -> None:
        """board: the workspace whose receipts, run registry and sweep this reads."""
        self.board = board

    def cancel(self, handle: str, *, host: str = "") -> StreamVerdict:
        """Cancel under the same claim as automatic settlement, preserving cleanup failures.

        The claim is waited for and not forever. A sweep, or a dispatch still landing on the very
        rental being cancelled, holds it for minutes, and a cancel that waited without a bound
        sat in a sleep loop for seven minutes on 2026-09-21 while the instance it was asked to
        end was already gone.
        """
        claim = self.board.dispatcher.cache.settlement
        try:
            claim.acquire(timeout=SETTLEMENT_SECONDS)
        except Timeout:
            raise MissionError(
                f"another mainboard process has held settlement for {SETTLEMENT_SECONDS:g}s, a "
                "sweep or a dispatch still landing; nothing was cancelled, ask again when it ends"
            ) from None
        try:
            return self._cancel(handle, host=host)
        finally:
            claim.release()

    def _cancel(self, handle: str, *, host: str = "") -> StreamVerdict:
        """Preserve available evidence, cancel, and advance the cursor only after release.

        Explicit cancellation may discard incomplete work, but records that loss rather than
        calling the evidence complete. A held dispatch has no machine to contact.
        """
        record = self.record(handle, host=host)
        if record.verdict == vocabulary.SUBMITTING:
            raise MissionError(
                f"creation {record.creation} has no confirmed provider handle; reconcile "
                "its provider label before cancellation; absence from one listing is not proof"
            )
        if record.verdict in vocabulary.TERMINAL and record.reported == record.verdict:
            return self.handled(handle, host=host)
        if record.verdict == vocabulary.PREPARED:
            cache = self.board.dispatcher.cache
            try:
                cache.leave_prepared(record, vocabulary.CANCELLED)
            except ValueError as changed:
                raise MissionError(
                    f"creation {record.creation} changed during cancellation; reconcile "
                    "its provider label before retrying"
                ) from changed
            return self.handled(handle, host=host)
        if record.verdict == vocabulary.HELD:
            # Nothing ever took this one, so there is nothing to kill and no backend to ask.
            # What exists is the request, and cancelling it is the sweep never offering it again.
            cache = self.board.dispatcher.cache
            cache.report(
                cache.resolve(record, vocabulary.CANCELLED, None, vocabulary.CANCELLED),
                vocabulary.CANCELLED,
            )
            return self.handled(handle, host=host)
        run = self.board.job(record.handle, host=record.target)
        monitor = self.board.monitor()
        receipts: tuple[str, ...] = ()
        try:
            receipts = monitor.capture(record, run)
            pulled = monitor.pull(run)
            monitor.verify(record, run, pulled, receipts)
        except (MissionError, OSError, ValueError) as fault:
            monitor.evidence(record, receipts, status="unverified", detail=f"cancelled: {fault}")
        else:
            monitor.evidence(record, receipts, status="copied")
        verdict = (
            record.verdict
            if record.verdict is not None and record.verdict in vocabulary.TERMINAL
            else vocabulary.CANCELLED
        )
        state = JobState(handle=record.handle, state=vocabulary.CANCELLED, verdict=verdict)
        stored = self.board.dispatcher.cache.resolve(
            record, state.state, record.exit_code, verdict
        )
        run.kill()
        if monitor.release(run):
            if self.board.dispatcher.cache.run(handle, record.target).evidence == "copied":
                monitor.evidence(record, receipts, status="verified")
            monitor.track(record, state, detail=short_reason(vocabulary.CANCELLED, None))
            self.board.dispatcher.cache.report(stored, verdict)
        return self.handled(handle, host=host)

    def captured(self, handle: str, *, host: str = "") -> str:
        """`handle`'s output: the tail a settle brought home, else whatever the backend still has.

        The stored copy is preferred because it is the one that still exists. A settled run's log
        lives under the host's state dir, which a cleanup eventually takes, or on a rented disk
        that was destroyed the moment the rental ended, so the live read is the fallback and not
        the other way round. A run that ended before this workspace knew to capture anything, and
        whose host is still up, therefore still answers.

        handle: the dispatched run to read.
        host: the alias narrowing a handle recorded on several hosts.
        """
        record = self.record(handle, host=host)
        stream, _ = streamed(record.name or "", handle=record.handle)
        stored = directory(self.board, stream) / f"{record.handle}.log"
        if stored.is_file():
            return stored.read_text(encoding="utf-8")
        return self.board.job(record.handle, host=record.target).transcript()

    def handled(self, handle: str, *, host: str = "") -> StreamVerdict:
        """The settled truth of one dispatched run, its registry row under its receipts.

        The registry row is durable dispatch state and always exists for a real handle, so a
        workspace that tracks nothing still gets an answer, and the receipts overlay it with
        whatever richer truth they hold.
        """
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
        recorded = eventful(events)
        mine = [trial for trial in recorded if trial.handle == record.handle]
        cases = {
            (str(case[0]), str(case[1]))
            for event in events
            if event.topic == Topic.EVIDENCE
            for group in [event.data.get("trials")]
            if isinstance(group, list)
            for case in group
            if isinstance(case, list) and len(case) == 2
        }
        harvested = tuple(trial for trial in harvest(under) if (trial.run, trial.job) in cases)
        floor = self.swept(tuple(mine)) or (self.__floor(record, job=job),)
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
        """The settled truth of `target`, a receipts store, a file, a stream, or a handle.

        target: what to read, tried in that order.
        host: the alias narrowing a handle recorded on several hosts.
        run: which run of a receipts store to score, its newest when empty. Meaningless for the
            other three targets, which carry one run's evidence by construction.
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
            missing = tuple(
                self.__floor(record, job=job)
                for record in self.board.dispatcher.cache.tracked()
                for stream, job in [streamed(record.name or "", handle=record.handle)]
                if stream == target
                and (record.target, record.handle)
                not in {(trial.target, trial.handle) for trial in recorded}
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
        """One receipts STORE scored a run at a time, None when `path` holds no store at all.

        A store holds every run a harness ever took, so reading them as one flat stream lets a
        lane that broke in one campaign condemn a clean re-run months later, with no flag able to
        dig it out. The newest run answers by default and `--run` names an older one.

        The reader is imported here rather than at the top of this module because it carries a
        dataframe engine costing a fifth of a second to import, and this module is on the path of
        every command this tool runs. It is the charge-on-touch rule the package facade already
        states, spent on the one branch that needs the engine.

        path: the directory to read, the partition root or the evidence directory above it.
        stream: what the caller asked for, which the heading names.
        run: which run to score, the newest when empty.
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
        """`trials` joined onto the durable run registry: their provenance, and any in-flight
        outcome the registry has already settled.

        A batch's own watch is the only thing that publishes a batched job's settled line, and
        the unattended sweep deliberately writes none, so the two can never double each other.
        Kill the session holding that watch and the sweep still does everything else: it probes
        the job, pulls its log home beside the stream, and memoizes the terminal verdict in the
        registry. Nothing tells the stream. So this verb read thirteen finished miyabi-g jobs as
        running, with their thirteen pulled logs sitting in the same directory it was reading
        (gigatoken-shootout rep92, 2026-09-04), and would have gone on saying it until a watch
        nobody was going to start again said otherwise.

        The registry row is that outcome written down, which is the same floor `handled` already
        answers a receiptless run from, so it is joined on here too. Only onto rows the receipts
        left in flight: a settled line carries a detail and an exit code the registry has no
        column for, and it was written by the pass that read the same probe.

        trials: the stream's own rows, in the order they will be reported.
        """
        return tuple(self.__registered(trial) if trial.handle else trial for trial in trials)

    def __registered(self, trial: TrialVerdict) -> TrialVerdict:
        """`trial` under its registry row: its provenance always, its outcome while it flies.

        The provenance is joined onto every dispatched row, settled ones included, because what
        a row was measured from does not stop being true when the job ends and a mirror carries
        no history to read it from later. The outcome is joined only where the receipts left the
        row in flight, since a settled line carries a detail and an exit code this registry has
        no column for and was written by the pass that read the same probe.
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
        joined = delivery(joined, record)
        if joined.code != 1:
            return joined
        return joined.model_copy(update={"cause": self.why(record)})

    def __floor(self, record: RunRecord, *, job: str) -> TrialVerdict:
        """`record`'s own row, the answer for a run whose stream holds no receipts at all."""
        alone = registered(record, job=job)
        return alone if alone.code != 1 else alone.model_copy(update={"cause": self.why(record)})

    def why(self, record: RunRecord) -> str:
        """Why `record` failed, off the log the sweep brought home; empty when it brought none."""
        return reason(self.board, record)

    def wait(
        self,
        handle: str,
        *,
        host: str = "",
        timeout: float = 0.0,
        interval: float = vocabulary.POLL_SECONDS,
        poll: Callable[[float], None] = sleep,
    ) -> StreamVerdict:
        """Block until `handle` settles, sweeping the same durable path the monitor cron runs.

        Every pass is one `Monitor.once`, so waiting here pulls results back, releases rentals
        and writes receipts through the same path as an unattended sweep. Interruption stops
        this waiter, not the submitted job or provider billing. The answer's code is the
        normalized receipt outcome, not the original process exit status. A batch id waits for
        every job of the batch and answers with the batch's verdict.

        handle: the dispatched run to wait on, or a batch id as `batch run` printed it.
        host: the alias narrowing a handle recorded on several hosts.
        timeout: give up after this many wall seconds, 0 to wait as long as it takes; the
            answer then reports the run still in flight and exits 2.
        interval: seconds between sweeps.
        poll: the sleeper between sweeps, injectable for tests.
        """
        deadline = monotonic() + timeout if timeout else None
        monitor = self.board.monitor()
        stream = (directory(self.board, handle) / "events.ndjson").is_file()
        # What already settled is answered off its receipts before any pass runs, since a pass
        # settles the whole workspace and a caller re-reading a finished batch owes it nothing.
        while (settled := self.__settled(handle, host=host, stream=stream)) is None:
            if deadline is not None and monotonic() >= deadline:
                return self.of(handle) if stream else self.handled(handle, host=host)
            monitor.once()
            if self.__settled(handle, host=host, stream=stream) is None:
                poll(interval)
        return settled

    def __settled(self, handle: str, *, host: str, stream: bool) -> StreamVerdict | None:
        """`handle`'s final answer, None while any of it is still in flight.

        A batch settles when every job's row has, which is what its stream verdict already adds
        up, so a batch is asked that rather than a record it has none of.
        """
        if stream:
            answered = self.of(handle)
            return None if answered.code == _IN_FLIGHT else answered
        if self.record(handle, host=host).reported in vocabulary.TERMINAL:
            return self.handled(handle, host=host)
        return None


def eventful(events: Iterable[Event]) -> tuple[TrialVerdict, ...]:
    """Every job's settled row out of one stream's event envelopes.

    The cursor logic is `latest` per topic per job, the same read every resumed pass uses, so a
    re-dispatched job answers with its newest run, and a job that was refused or is being held
    on a target's quota still has a row rather than vanishing from the stream it was declared in.
    Which of those three the row settles on is decided by the clock alone, in `_joined`.

    A job nothing was ever dispatched for answers from its skip instead, which is the one row
    shape a target never spoke about. Every job the stream mentions therefore has a row, and a
    `--only` wave's unselected jobs stop being invisible to the verb that reports the batch.
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

    A silent empty table reads as a failure, and it is usually not one: the run may not have
    started, or the harness may be writing a shape nobody told this verb about. Saying which
    costs one line and saves the reader from guessing at an outcome, which is the exact
    fabrication this whole module exists to prevent.

    The two shapes are named rather than the tools that write them. This verb reads a contract,
    not a producer, so a harness earns the same reading by printing the same line and nothing
    here has to learn what that harness is called.

    path: the file that was read.
    trials: what reading it produced.
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

    A run's own receipts land in their own file rather than in the stream's event log, because
    the two are different shapes and the log is read as envelopes. Keeping them apart is what
    lets a settle append a trial without any reader of the event stream having to tolerate a
    line it was never promised.
    """
    path = under / "receipts.ndjson"
    return lined(path) if path.is_file() else ()


def qualified(
    trials: tuple[TrialVerdict, ...], events: Iterable[Event]
) -> tuple[TrialVerdict, ...]:
    """Overlay append-only delivery corrections; leave original computation receipts intact."""
    statuses: dict[tuple[str, str, str], Event] = {}
    for event in sorted(events, key=lambda event: event.at):
        if event.topic == Topic.EVIDENCE:
            identity = (
                str(event.data.get("handle", "")),
                str(event.data.get("target", "")),
                str(event.data.get("submitted_at", "")),
            )
            statuses[identity] = event
    updates: dict[tuple[str, str], Event] = {}
    handles: dict[tuple[str, str], Event] = {}
    for event in statuses.values():
        cases = event.data.get("trials", [])
        handles[(str(event.data.get("target", "")), str(event.data.get("handle", "")))] = event
        if isinstance(cases, list):
            updates.update(
                {
                    (str(case[0]), str(case[1])): event
                    for case in cases
                    if isinstance(case, list) and len(case) == 2
                }
            )
    result: list[TrialVerdict] = []
    for trial in trials:
        status = (
            handles.get((trial.target, trial.handle))
            or handles.get(("", trial.handle))
            or updates.get((trial.run, trial.job))
        )
        if status is None or status.data.get("status") in {"verified", "not_started"}:
            result.append(trial)
            continue
        word = "unverified" if status.data.get("status") == "unverified" else "blocked"
        result.append(
            trial.model_copy(
                update={
                    "verdict": word,
                    "detail": str(
                        status.data.get("detail") or "evidence settlement is incomplete"
                    ),
                }
            )
        )
    return tuple(result)


def lined(path: Path) -> tuple[TrialVerdict, ...]:
    """Every row a receipts file holds, whichever of the two written shapes each line is.

    An `Event` line joins its stream's per-job cursor; a `trial_receipt` line is one trial,
    whole. A line that is neither readable JSON nor either shape is skipped rather than fatal,
    the same tolerance the receipts replay itself extends to a torn log. That promise used to
    hold for a torn line and break for a whole one: any JSON object without a `trial_receipt`
    key was handed straight to `Event`, so a file of some other tool's evidence answered this
    verb with a pydantic traceback rather than with the empty table the caller could then be
    told about. A well-formed line of a shape this verb does not read is exactly as skippable
    as a truncated one.
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
    """One printed `trial_receipt` payload as a settled row.

    The contract names `case_id`, `outcome`, `producer`, `node` and `gates` as optional fields
    and any harness may add its own, so everything is read leniently and an absent field is an
    empty cell rather than a refusal.

    `run_id` IS READ ONLY WHERE `case_id` IS ABSENT, and that is the whole of the compatibility.
    The trials harness spelled the test case `run_id` beside a `run` column already holding the
    run, so the name promised a join nobody could make; a dispatched job printing its own receipt
    still names its job there and is still read. A row carrying both is a new row and its
    `case_id` is what this reads, so the old meaning never enters a new join.
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
    return trial.model_copy(
        update={
            "verdict": "unverified" if record.evidence == "unverified" else "blocked",
            "detail": "evidence is unverified"
            if record.evidence == "unverified"
            else "evidence settlement is incomplete",
        }
    )


def contention(attestation: Event | None) -> str:
    """What a job's attestation says it started under, empty for an idle node or no attestation.

    Only the unwelcome half is rendered, since a clean measurement's whole point is that there is
    nothing to say about it, and a column full of the word `idle` would bury the one row that
    matters. The busy figure rides along so a reader can weigh it rather than take the flag's
    word for it.

    attestation: the run's `job.attested` line, None when nothing attested.
    """
    if attestation is None or attestation.data.get("idle"):
        return ""
    return f"gpu {attestation.data.get('gpu_pct', 0)}% busy at start"


def _unselected(job: str, skip: Event) -> TrialVerdict:
    """One job a run was told to leave out, as the row that is already over.

    A skip is not a fourth answer to an offer, it is the statement that no offer was made in
    this wave, so it never outranks a dispatch however much newer it is. The nine jobs this
    workspace ran at 13:43 and left out of an `--only` wave at 18:43 (2026-09-04) are nine runs
    that happened, not nine that unhappened, and a rule ranking the skip by its clock alone
    would have thrown their outcomes away. So the skip decides a row exactly when nothing was
    ever dispatched for the job.

    Shown rather than dropped, because a plan worked through in waves is read against the plan
    and a reader has to see which jobs were not asked for rather than wonder where they went.
    Terminal from the start, because nothing that was never dispatched can move, so a completion
    check neither waits on it nor counts it as a failure.

    job: the job's name inside the stream.
    skip: its newest `job.skipped` line.
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

    Taken, turned away, and kept waiting on a quota are three answers to the same offer, so the
    newest of the three is the one that stands and none of them outranks the others by being a
    particular topic. That ranking is the shared `OFFERED` cursor's, so a re-dispatch supersedes
    an earlier refusal here and in the live watch alike: four jobs miyabi-g's `njobs-g` limit
    turned away at 13:43 went out at 18:43 (2026-09-04), and reading the refusal because it was
    a refusal buried the run five hours younger than it. A refusal recorded after a submission
    is terminal for the same reason, in the target's own words.

    Past the answer, a settled line wins the verdict and a state line stands in while the job
    flies. A job submitted again after settling compares handles, so a stale settlement never
    silences the run of it that is still going. An attestation is carried onto every row the run
    has, since what the machine was doing at the start is as true of the finished measurement as
    it was of the running one.
    """
    contended = contention(attestation)
    target = str(answer.data.get("target", ""))
    if answer.topic is Topic.HELD:
        return TrialVerdict(
            job=job,
            target=target,
            state=vocabulary.HELD,
            verdict=vocabulary.HELD,
            detail=str(answer.data.get("reason", "")),
        )
    if answer.topic is Topic.REFUSED:
        return TrialVerdict(
            job=job,
            target=target,
            verdict="refused",
            detail=str(answer.data.get("reason", "")),
        )
    handle = str(answer.data.get("handle", ""))
    node = str(answer.data.get("node", ""))
    state = _matching(state, answer)
    ended = _matching(ended, answer)
    current = str(state.data.get("state", "")) if state else ""
    verdict = str(state.data.get("verdict", "")) if state else ""
    if ended is not None:
        code = ended.data.get("exit_code")
        return TrialVerdict(
            job=job,
            handle=handle,
            target=target,
            node=node,
            state=current,
            verdict=str(ended.data.get("verdict", "")),
            exit_code=code if isinstance(code, int) else None,
            detail=str(ended.data.get("detail", "")),
            contended=contended,
        )
    return TrialVerdict(
        job=job,
        handle=handle,
        target=target,
        node=node,
        state=current,
        verdict=verdict or vocabulary.RUNNING,
        contended=contended,
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
