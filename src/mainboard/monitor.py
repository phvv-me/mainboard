# The durable sweep behind `mainboard monitor`: one pass over every dispatched job the shared
# cache still owes an outcome for. Everything it reads is durable state, so a periodic cron
# closes out jobs this process never submitted and a remote job's result never depends on the
# agent that dispatched it staying alive to see it end.

import json
import os
import shlex
from pathlib import Path
from sqlite3 import Error as SQLiteError
from time import sleep
from typing import TYPE_CHECKING, Final

from filelock import Timeout
from plumbum.commands.processes import ProcessExecutionError

from .batch.receipts import Topic, latest, publish
from .batch.runner import directory
from .core.errors import MissionError
from .dispatch import vocabulary
from .dispatch.backends.base import route
from .dispatch.dispatcher import Verdict
from .dispatch.evidence import covered_in, receipts_in
from .dispatch.schedulers import HostUnreachable, is_quota_refusal, short_reason
from .dispatch.shared import logger
from .dispatch.state import DownHost, Failed, Finished, Held, MonitorReport, Resumed
from .dispatch.vocabulary import JobState
from .tracking import is_batched, streamed

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from pydantic import JsonValue

    from .batch.receipts import Bus
    from .board import Board, Run
    from .dispatch.state import RunRecord

# The routing answer for the schedulers reached over ssh, the one family whose whole host can be
# asked about in a single query. A provider has no such listing and is asked run by run.
_QUEUED: Final = "ssh-family"

# What a resubmission may fail with before it becomes this run's row rather than the end of the
# sweep, the set a batch dispatch absorbs: one target refusing says nothing about the next run.
_REFUSALS = (MissionError, HostUnreachable, OSError, LookupError, SystemExit)

# A provider creation with no confirmed handle is neither probed nor resubmitted, only reported.
_UNCREATED = {
    vocabulary.PREPARED: "provider creation prepared; no create attempted; cancel if abandoned",
    vocabulary.SUBMITTING: "provider creation has no confirmed handle; reconcile its exact "
    "label before retrying; no automatic resubmission or release",
}


class Sweep:
    """Every tracked run rebuilt and resolved, one query per target rather than one per run.

    A cache holding a thousand runs on one box costs one query, not a thousand round trips, and
    a run whose verdict is already terminal costs nothing. A target that cannot be resolved (no
    declared root, a host that will not answer, a provider API that refuses, or one that goes
    quiet halfway) is recorded once with why; its unanswered runs simply have no state here and
    are left for the next pass.

    runs: each record's rebuilt run.
    states: each record's current state, absent where its target could not be resolved.
    down: why a target could not be resolved, one entry per target.
    """

    def __init__(self, board: Board, records: Sequence[RunRecord]) -> None:
        """records: every run the sweep still owes an outcome for, in the order it reports them."""
        self.board = board
        self.runs: dict[RunRecord, Run] = {}
        self.states: dict[RunRecord, JobState] = {}
        self.down: dict[str, str] = {}
        # The kind rides in the key because it picks the scheduler and was recorded at dispatch,
        # so a run is asked about the way it was submitted even if its host was redeclared. A
        # held or uncreated dispatch has no handle any target heard of; `Monitor.held` asks again.
        groups: dict[tuple[str, str], list[RunRecord]] = {}
        for record in records:
            if record.verdict not in {vocabulary.HELD, *_UNCREATED}:
                groups.setdefault((record.target, record.kind), []).append(record)
        for (target, kind), owned in groups.items():
            if target in self.down:
                continue
            try:
                self.settle(kind, owned)
            except (HostUnreachable, MissionError, OSError) as fault:
                self.down[target] = str(fault)

    def settle(self, kind: str, records: Sequence[RunRecord]) -> None:
        """Rebuild every run in `records` and resolve the ones still owed a probe.

        Every record is rebuilt, settled ones included, since a target with no declared root
        refuses at the rebuild. A terminal verdict can never change, so it is read from the cache
        without touching the target, which also keeps a finished job the queue already forgot
        from reading back as vanished. The rest cost one query for the whole host, or one call
        per run where a provider, which has no listing, owns them.
        """
        pending: list[RunRecord] = []
        for record in records:
            self.runs[record] = self.board.job(record.handle, host=record.target)
            if record.verdict is not None and record.verdict in vocabulary.TERMINAL:
                self.states[record] = JobState(
                    handle=record.handle,
                    state=record.state,
                    exit_code=record.exit_code,
                    verdict=record.verdict,
                )
            else:
                pending.append(record)
        if not pending:
            return
        if route(kind) != _QUEUED:
            self.states.update({record: self.runs[record].poll() for record in pending})
            return
        found = self.board.dispatcher.states([self.runs[record].handle for record in pending])
        self.states.update({record: found[record.handle] for record in pending})


class Monitor:
    """The durable pass over every dispatched job still owed an outcome.

    One `once` probes each unsettled run, pulls back whatever just finished, releases what a
    settled run still holds, records each fresh verdict in the study ledger that owns it, and
    advances the run's reported cursor so the next pass says nothing. A host that does not answer
    is reported once and its jobs are left for the next pass. Releasing is why this is worth
    running unattended: a queue stops charging when the job ends, but a provider keeps the
    instance billing after its command exits, so a terminal verdict here is followed by a cancel
    the scheduler path deliberately does not make.
    """

    def __init__(self, board: Board) -> None:
        self.board = board
        self.cache = board.dispatcher.cache
        self.streams: dict[str, Bus] = {}
        self.quiet: dict[str, str] = {}

    def capture(self, record: RunRecord, job: Run) -> tuple[str, ...]:
        """Save the native log and deduplicated receipts before a rental can be destroyed.

        A retry can read the saved log after the provider disappears. Files are flushed before
        the delivery checkpoint is published; receipt references are verified separately.
        """
        stream, name = streamed(record.name or "", handle=record.handle)
        under = directory(self.board, stream)
        under.mkdir(parents=True, exist_ok=True)
        log = under / f"{record.handle}.log"
        transcript = job.transcript()
        if not transcript:
            return receipts_in(log.read_text(encoding="utf-8")) if log.is_file() else ()
        _synced(log, transcript, "w")
        harvested = receipts_in(transcript)
        path = under / "receipts.ndjson"
        known = harvested and path.is_file()
        seen = set(path.read_text(encoding="utf-8").splitlines()) if known else set()
        if fresh := [line for line in harvested if line not in seen]:
            _synced(path, "\n".join(fresh) + "\n", "a")
        logger.info(
            "captured %d log lines for %s (%s)", transcript.count("\n"), record.handle, name
        )
        return harvested

    def evidence(
        self, record: RunRecord, receipts: tuple[str, ...], *, status: str, detail: str = ""
    ) -> None:
        """Append transfer status without rewriting the computational receipt or exit code."""
        self.cache.delivery(record, status)
        stream, job = streamed(record.name or "", handle=record.handle)
        bus = self.streams.setdefault(stream, self.board.receipts(stream))
        data: dict[str, JsonValue] = {
            **_identity(record),
            "status": status,
            "detail": detail,
            "trials": _cases(receipts),
        }
        seen = latest(bus.replay(), Topic.EVIDENCE).get(job)
        if seen is None or seen.data != data:
            try:
                publish(bus, stream, Topic.EVIDENCE, job=job, data=data)
            except OSError as fault:
                logger.error(
                    "evidence status is cached but its event could not be saved: %s", fault
                )

    def verify(
        self,
        record: RunRecord,
        job: Run,
        pulled: str | None,
        receipts: tuple[str, ...],
        *,
        covered: bool = False,
        verdict: str = vocabulary.OK,
    ) -> None:
        """A successful transport is not proof that its declared artifacts arrived.

        covered: the transcript is a trials session whose every cell was already complete and
            skipped, the one native run that legitimately captures no receipt.
        verdict: what the run ended on. Only a run claiming success owes a receipt: holding a
            crash pending re-pulled it every sweep for the cache's lifetime, and ten such runs
            cost every `wait` a minute per pass.
        """
        # Loaded only when evidence needs checking, so its dataframe engine never taxes startup.
        from .trials.artifacts import Artifacts

        artifacts: list[JsonValue] = []
        for line in receipts:
            payload = _receipt(line)
            if not isinstance(payload, dict) or not isinstance(
                declared := payload.get("artifacts", {}), dict
            ):
                raise ValueError("malformed trial receipt; raw line preserved in captured log")
            artifacts.extend(declared.values())
        if job.handle.fetch_path and pulled is None:
            raise MissionError("result transfer failed; remote evidence retained")
        native = any(
            Path(token.partition("::")[0]).name.startswith("test_")
            for token in shlex.split(record.script)
        )
        claimed = verdict == vocabulary.OK
        if native and claimed and job.handle.fetch_path and not receipts and not covered:
            raise MissionError("native trial has no captured receipt; evidence is unverified")
        if not receipts:
            return
        if pulled is not None:
            Artifacts.verify(
                receipts, directory=self.board.dispatcher.local(pulled), boundary=self.board.root
            )
        elif any(isinstance(value, dict) for value in artifacts):
            raise MissionError("receipt references artifacts but no fetch was declared")

    def asked(self, record: RunRecord) -> Run | Failed | None:
        """Ask `record`'s target for its held dispatch again, None while the quota is still full.

        A quota refusal leaves the row as it was for the next sweep. Any other refusal (a queue
        that does not exist, an account without permission) answers the same way forever, so the
        row settles failed and that failure is returned for the only pass that still sees it.
        """
        if record.request is None:
            return None
        try:
            return self.board.dispatch(record.request)
        except _REFUSALS as refusal:
            if is_quota_refusal(str(refusal)):
                return None
            self.cache.report(
                self.cache.resolve(record, vocabulary.FAILED, None, vocabulary.FAILED),
                vocabulary.FAILED,
            )
            logger.warning("held dispatch for %s refused: %s", record.target, refusal)
            return _failed(record, f"held dispatch refused: {refusal}")

    def held(self) -> tuple[list[Resumed], list[Held], list[Failed]]:
        """Ask every quota-held dispatch's target for room again, in the order they were held.

        This makes a hold a delay rather than a loss: a wave meeting a group's job limit used to
        drop what the queue would not take, found by counting logs hours later (miyabi-g njobs-g,
        2026-09-04). A request that goes through replaces its placeholder row under the new
        handle and tells the batch that asked, through its own receipts.
        """
        resumed: list[Resumed] = []
        waiting: list[Held] = []
        refused: list[Failed] = []
        for record in reversed(self.cache.live()):
            if record.verdict != vocabulary.HELD:
                continue
            run = self.asked(record)
            if run is None:
                waiting.append(
                    Held(handle=record.handle, target=record.target, reason=record.reason)
                )
                continue
            if isinstance(run, Failed):
                refused.append(run)
                continue
            self.cache.forget(record)
            self.submitted(record, run)
            resumed.append(Resumed(handle=run.handle.id, target=record.target, name=record.name))
            logger.info("held dispatch went through as %s on %s", run.handle.id, record.target)
        return resumed, waiting, refused

    def submitted(self, record: RunRecord, run: Run) -> None:
        """Publish the submission a batch would have written for its held job that went out here.

        Without it a watch reading submissions out of the receipts shows the job held forever
        while it runs.
        """
        if not is_batched(record.name):
            return
        stream, job = streamed(record.name, handle=run.handle.id)
        bus = self.streams.setdefault(stream, self.board.receipts(stream))
        publish(
            bus,
            stream,
            Topic.SUBMITTED,
            job=job,
            data={
                "handle": run.handle.id,
                "target": record.target,
                "kind": run.handle.kind,
                "command": record.script,
                **({"node": record.node} if record.node else {}),
            },
        )

    def expired(self) -> list[Failed]:
        """Release due rentals before probing hosts or waiting for the settlement lock.

        At this boundary the reserved deletion interval has begun, so no unbounded transfer
        starts. Previously copied evidence survives; missing evidence stays explicitly
        unverified. Provider failures keep the exact handle tracked for retry.
        """
        failed: list[Failed] = []
        for record in self.cache.tracked():
            if record.lease is None or not record.lease.expired or record.verdict in _UNCREATED:
                continue
            try:
                backend = route(record.kind)
                if backend == _QUEUED:
                    continue
                current = self.cache.run(record.handle, record.target)
            except (MissionError, OSError, ValueError, LookupError, SQLiteError) as fault:
                failed.append(
                    _failed(
                        record, f"deadline identity check failed; release remains pending: {fault}"
                    )
                )
                continue
            if (
                current.submitted_at != record.submitted_at
                or current.lease is None
                or not current.lease.expired
            ):
                continue
            copied = current.evidence in {"copied", "verified"}
            detail = "rental release deadline reached"
            try:
                self.evidence(
                    current, (), status="copied" if copied else "unverified", detail=detail
                )
            except (MissionError, OSError, ValueError, SQLiteError) as fault:
                detail += f"; evidence recording failed: {fault}"
                logger.error("%s: %s", record.handle, detail)
            try:
                backend().cancel(record.handle)
            except (MissionError, OSError, ValueError) as fault:
                detail += f"; release failed and will be retried: {fault}"
            else:
                try:
                    current = self.cache.resolve(
                        current, vocabulary.TIMEOUT, None, vocabulary.TIMEOUT
                    )
                    self.evidence(
                        current, (), status="verified" if copied else "unverified", detail=detail
                    )
                    self.cache.report(current, current.verdict or vocabulary.TIMEOUT)
                except (MissionError, OSError, ValueError, SQLiteError) as fault:
                    detail += f"; release confirmed but bookkeeping needs repair: {fault}"
            failed.append(_failed(record, detail))
        return failed

    def once(self) -> MonitorReport:
        """Serialize settlement before reading its cursor, including across monitor processes."""
        expired = self.expired()
        lock = self.cache.settlement
        try:
            lock.acquire(timeout=0)
        except Timeout:
            logger.debug("another monitor owns settlement; leaving its cursor untouched")
            return MonitorReport(running=None, failed=expired)
        try:
            report = self._once()
            return report.model_copy(update={"failed": [*expired, *report.failed]})
        finally:
            lock.release()

    def _once(self) -> MonitorReport:
        """Resolve every unsettled run once, harvest the newly terminal ones, report the changes.

        Resolving goes by target (see `Sweep`); the harvest then walks runs in cache order, so
        what a run causes does not depend on which target answered first. A run in flight is only
        counted. A run that ended has its results pulled, its output and trial receipts captured
        beside its receipts stream, whatever it holds released, its verdict recorded in the
        owning study ledger, and only then its cursor advanced, so a sweep killed halfway repeats
        work rather than losing an outcome. Capture must precede release, which destroys a rented
        instance and its log; release must precede the cursor, so a pass dying between them
        cancels the rental again next time.

        The pull happens whatever the exit code said. A sweep of 500 trials dying at 400 leaves
        399 immutable receipt fragments on the host, exactly the crash safety the staged store
        exists to give, and a partial sweep is also how a metered rental that hit its cap ends.
        """
        running = 0
        finished: list[Finished] = []
        self.quiet.clear()
        resumed, waiting, failed = self.held()
        fleet = self.board.fleet()
        records = self.cache.tracked()
        failed.extend(
            _failed(record, reason)
            for record in records
            if (reason := _UNCREATED.get(record.verdict or ""))
        )
        resolved = Sweep(self.board, records)
        for record in records:
            state = resolved.states.get(record)
            if state is None:
                continue
            job = resolved.runs[record]
            current = self.cache.resolve(record, state.state, state.exit_code, state.verdict)
            if state.verdict not in vocabulary.TERMINAL:
                if state.stage == vocabulary.RUNNING:
                    self.pull(job)
                self.track(record, state, detail="")
                running += 1
                continue
            discarded = state.state == vocabulary.CANCELLED and current.evidence == "unverified"
            if current.evidence == "not_started" or discarded:
                detail = (
                    "explicit cancellation; evidence remains unverified"
                    if discarded
                    else "provisioning ended before a native launch was attempted"
                )
                if self.release(job):
                    if not discarded:
                        self.evidence(record, (), status="not_started", detail=detail)
                    self.track(record, state, detail=detail)
                    fleet.settle(
                        {job.handle: Verdict(verdict=state.verdict, exit_code=state.exit_code)}
                    )
                    self.cache.report(record, state.verdict)
                else:
                    detail += "; release failed and will be retried"
                failed.append(_failed(record, detail))
                continue
            stream, name = streamed(record.name or "", handle=record.handle)
            bus = self.streams.setdefault(stream, self.board.receipts(stream))
            ours = _identity(record)
            history = (
                event
                for event in bus.replay()
                if all(event.data.get(key) == value for key, value in ours.items())
            )
            previous = latest(history, Topic.EVIDENCE).get(name)
            copied = current.evidence in {"copied", "verified"} or (
                previous is not None and previous.data.get("status") in {"copied", "verified"}
            )
            harvested: tuple[str, ...] = ()
            pulled = None
            try:
                log = directory(self.board, stream) / f"{record.handle}.log"
                if copied:
                    if previous is None and not log.is_file():
                        raise MissionError("copied evidence has no recoverable local receipt log")
                    transcript = log.read_text(encoding="utf-8") if log.is_file() else ""
                    harvested = receipts_in(transcript)
                    if previous is not None and previous.data.get("trials") and not harvested:
                        raise MissionError("copied trial receipts are missing from the local log")
                    pulled = job.handle.fetch_path
                else:
                    self.answering(job)
                    pulled = self.pull(job)
                    self.answering(job)
                    harvested = self.capture(record, job)
                    transcript = log.read_text(encoding="utf-8") if log.is_file() else ""
                self.verify(
                    record,
                    job,
                    pulled,
                    harvested,
                    covered=covered_in(transcript),
                    verdict=state.verdict,
                )
            except (MissionError, OSError, ValueError) as fault:
                detail = f"settlement pending; remote evidence retained: {fault}"
                self.evidence(record, harvested, status="pending", detail=detail)
                logger.error("%s on %s: %s", record.handle, record.target, detail)
                failed.append(_failed(record, detail))
                continue
            self.evidence(record, harvested, status="copied")
            if not self.release(job):
                detail = "settlement pending; release failed and will be retried"
                self.evidence(record, harvested, status="copied", detail=detail)
                failed.append(_failed(record, detail))
                continue
            self.evidence(record, harvested, status="verified")
            if state.verdict == vocabulary.OK:
                finished.append(
                    Finished(handle=record.handle, target=record.target, pulled_path=pulled)
                )
                detail = pulled or ""
            else:
                detail = short_reason(state.verdict, state.exit_code)
                failed.append(
                    Failed(
                        handle=record.handle,
                        target=record.target,
                        reason=detail,
                        pulled_path=pulled,
                    )
                )
            self.track(record, state, detail=detail)
            fleet.settle({job.handle: Verdict(verdict=state.verdict, exit_code=state.exit_code)})
            self.cache.report(self.cache.run(record.handle, record.target), state.verdict)
        return MonitorReport(
            running=running + len(waiting),
            resumed=resumed,
            held=waiting,
            finished=finished,
            failed=failed,
            unreachable_hosts=[
                DownHost(host=host, reason=why) for host, why in resolved.down.items()
            ],
        )

    def pull(self, job: Run) -> str | None:
        """Bring a job's recorded results home whatever its verdict, answering where they landed.

        A crashed run keeps everything it wrote, the store staging and renaming each fragment so
        those readings survive the trial that killed it, so a failed job is the one most worth
        carrying home. None when no results path was recorded at dispatch or the pull failed (a
        directory never written, a host dropped mid-transfer, a rented disk that dies with the
        rental): one missing artifact is a warning, never a sweep that dies holding every other
        job's outcome.
        """
        path = job.handle.fetch_path
        if not path:
            return None
        try:
            job.pull()
        except (HostUnreachable, ProcessExecutionError, MissionError, OSError) as fault:
            if isinstance(fault, HostUnreachable):
                self.quiet[job.handle.host] = str(fault)
            logger.warning("could not pull %s from %s: %s", path, job.handle.host, fault)
            return None
        return path

    def answering(self, job: Run) -> None:
        """Refuse to contact a host that already went quiet in this pass.

        A dead host costs a full connect timeout per contact, and a settled run is contacted for
        its results and again for its log, so a handful of runs on one dead host held every pass
        for minutes. Later runs on that host are left pending for the next pass unasked.
        """
        if (fault := self.quiet.get(job.handle.host)) is not None:
            raise MissionError(f"{job.handle.host} went quiet earlier in this pass: {fault}")

    def release(self, job: Run) -> bool:
        """Let a settled run go, so nothing keeps billing for work that already ended.

        A scheduler job releases nothing; a provider run is cancelled here, the only thing that
        ends the rental. Asking twice is expected, since this pass may re-run one an earlier pass
        released, and a refused cancel is a warning, never the end of the sweep.
        """
        try:
            job.release()
        except (MissionError, OSError) as fault:
            logger.warning("could not release %s on %s: %s", job.handle.id, job.handle.host, fault)
            return False
        return True

    def track(self, record: RunRecord, state: JobState, *, detail: str) -> None:
        """Publish what this pass learned about one run into that run's own receipts stream.

        This tracks a plain submit or a study trial like a batch job. A batched run is skipped,
        since its batch's watch already publishes every line and a second publisher would double
        every row. Only a move is published, so a cron pass that finds nothing new writes nothing.

        detail: where its results landed or why it failed, empty while it is still in flight.
        """
        label = record.name or ""
        if is_batched(label) or not self.board.manifest.tracking.on:
            return
        stream, job = streamed(label, handle=record.handle)
        bus = self.streams.setdefault(stream, self.board.receipts(stream))
        seen = latest(bus.replay(), Topic.STATE).get(job)
        ours = _identity(record)
        moved = {**ours, "state": state.state or "", "verdict": state.verdict}
        if seen is not None and seen.data == moved:
            return
        publish(bus, stream, Topic.STATE, job=job, data=moved)
        if state.verdict in vocabulary.TERMINAL:
            settled = {
                **ours,
                "verdict": state.verdict,
                "exit_code": state.exit_code,
                "detail": detail,
            }
            publish(bus, stream, Topic.SETTLED, job=job, data=settled)

    def watch(self, interval: float) -> Iterator[MonitorReport]:
        """Repeat `once` every `interval` seconds, yielding each pass's report as it lands.

        The foreground loop a person watches; nothing durable depends on it.
        """
        while True:
            yield self.once()
            sleep(interval)


def _identity(record: RunRecord) -> dict[str, JsonValue]:
    """The fields that tie a published line to this one submission of `record`."""
    return {"handle": record.handle, "target": record.target, "submitted_at": record.submitted_at}


def _failed(record: RunRecord, reason: str) -> Failed:
    """`record`'s row in a report's failed list."""
    return Failed(handle=record.handle, target=record.target, reason=reason)


def _synced(path: Path, text: str, mode: str) -> None:
    """Write `text` to `path` and fsync it, so it outlives a rental destroyed right after."""
    with path.open(mode, encoding="utf-8") as opened:
        opened.write(text)
        opened.flush()
        os.fsync(opened.fileno())


def _receipt(line: str) -> JsonValue:
    """The `trial_receipt` payload of one captured line, None when it is not a receipt envelope."""
    envelope = json.loads(line)
    return envelope.get("trial_receipt") if isinstance(envelope, dict) else None


def _cases(receipts: tuple[str, ...]) -> list[JsonValue]:
    """Associate valid envelopes only; raw malformed lines stay in the captured log."""
    cases: list[JsonValue] = []
    for line in receipts:
        try:
            payload = _receipt(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            case = payload.get("case_id") or payload.get("run_id", "")
            cases.append([str(payload.get("run", "")), str(case)])
    return cases
