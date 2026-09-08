# The durable sweep behind `mainboard monitor`: one pass over every dispatched job the shared
# cache still owes an outcome for. Everything it reads is durable state, so a periodic cron
# closes out jobs this process never submitted and a remote job's result never depends on the
# agent that dispatched it staying alive to see it end.

import json
import os
import shlex
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout
from plumbum.commands.processes import ProcessExecutionError

from .batch.receipts import Topic, latest, publish
from .batch.runner import directory
from .core.errors import MissionError
from .dispatch import vocabulary
from .dispatch.backends.base import route
from .dispatch.dispatcher import Verdict
from .dispatch.evidence import receipts_in
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
_QUEUED = "ssh-family"

# What a resubmission is allowed to fail with before it becomes this run's row rather than the
# end of the sweep, the same set a batch dispatch absorbs: one target refusing says nothing about
# the next run on it.
_REFUSALS = (MissionError, HostUnreachable, OSError, LookupError, SystemExit)

# How many meaningful lines of a settled run's output are kept beside its receipts. Enough that a
# traceback and the work around it survive whole, bounded so a chatty training loop cannot fill
# the workspace with a run nobody will read.


class Sweep:
    """Every tracked run rebuilt and resolved, one query per target rather than one per run.

    The pass this feeds used to ask each run's own host about that one run, which a dispatch
    cache holding a thousand runs on a single box turns into a thousand round trips nobody waits
    for. Here each target is asked once for every handle it still owes an answer on, so the same
    thousand runs cost one query. A run whose verdict is already terminal costs nothing at all,
    since a terminal verdict can never change and is read straight from the cache.

    A target that cannot be resolved is recorded once with why, not once per run on it: a host
    with no declared root, a host that will not answer, a provider API that refuses. Its runs
    simply have no state here and are left for the next pass. That is also what a target which
    goes quiet halfway through means, so whatever it did answer for stays answered.

    runs: each record's rebuilt run.
    states: each record's current state, absent where its target could not be resolved.
    down: why a target could not be resolved, one entry per target.
    """

    def __init__(self, board: Board, records: Sequence[RunRecord]) -> None:
        """board: the workspace whose dispatch cache and host profiles the records belong to.

        records: every run the sweep still owes an outcome for, in the order it reports them.
        """
        self.board = board
        self.runs: dict[RunRecord, Run] = {}
        self.states: dict[RunRecord, JobState] = {}
        self.down: dict[str, str] = {}
        # A dispatch a quota is holding has no handle and no target that has heard of it, so it
        # is not a thing to probe. `Monitor.held` is what asks its target for room again.
        waiting = [record for record in records if record.verdict != vocabulary.HELD]
        for (target, kind), owned in self.grouped(waiting).items():
            if target in self.down:
                continue
            try:
                self.settle(kind, owned)
            except (HostUnreachable, MissionError, OSError) as fault:
                self.down[target] = str(fault)

    def grouped(self, records: Sequence[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
        """`records` bucketed by the target and kind that answer for them, first seen first.

        The kind rides in the key beside the target because it is what picks the scheduler and
        it was recorded at dispatch, so a host whose declared kind changed under an old run still
        has that run asked about the way it was submitted.
        """
        groups: dict[tuple[str, str], list[RunRecord]] = {}
        for record in records:
            groups.setdefault((record.target, record.kind), []).append(record)
        return groups

    def memoized(self, record: RunRecord) -> JobState | None:
        """`record`'s cached terminal state, None when it still owes its target a probe.

        A terminal verdict can never change, so it is read straight from the cache and the target
        is never touched for it, which is also what keeps a finished job the queue has already
        forgotten from reading back as vanished.
        """
        verdict = record.verdict
        if verdict is not None and verdict in vocabulary.TERMINAL:
            return JobState(
                handle=record.handle,
                state=record.state,
                exit_code=record.exit_code,
                verdict=verdict,
            )
        return None

    def settle(self, kind: str, records: Sequence[RunRecord]) -> None:
        """Rebuild every run in `records` and resolve the ones still owed a probe.

        Rebuilding covers every record, the ones the cache already settled included, since a
        target with no declared root refuses at the rebuild and that refusal is about the target
        rather than about any one run on it. What is left is one query for the whole host, unless
        a provider owns these runs, which has no listing to query and answers one run at a time.
        """
        pending: list[RunRecord] = []
        for record in records:
            self.runs[record] = self.board.job(record.handle, host=record.target)
            memoized = self.memoized(record)
            if memoized is None:
                pending.append(record)
            else:
                self.states[record] = memoized
        if not pending:
            return
        if route(kind) != _QUEUED:
            for record in pending:
                self.states[record] = self.runs[record].poll()
            return
        found = self.board.dispatcher.states([self.runs[record].handle for record in pending])
        self.states.update({record: found[record.handle] for record in pending})


class Monitor:
    """The durable pass over every dispatched job still owed an outcome.

    One `once` probes each unsettled run in the dispatch cache, pulls back the results of
    whatever just finished, releases what a settled run still holds, records each fresh verdict
    in the study ledger that owns it, and advances that run's reported cursor so the next pass
    over the same jobs says nothing at all. A host that does not answer is reported once with why
    and its jobs are left for the next pass, so a dead host never fails the sweep.

    Releasing is where a rented run differs from a queued one and why this sweep is worth running
    unattended at all. A queue stops charging when the job ends, but a provider keeps the
    instance up and billing after its command exits, so a terminal verdict here is followed by a
    cancel that the scheduler path deliberately does not make.
    """

    def __init__(self, board: Board) -> None:
        """board: the workspace board whose dispatch cache and study ledgers the sweep settles."""
        self.board = board
        self.cache = board.dispatcher.cache
        self.streams: dict[str, Bus] = {}

    @staticmethod
    def unpulled(path: str, *, host: str, fault: Exception) -> None:
        """Log one failed pull as a warning and stand for its absent results path."""
        logger.warning("could not pull %s from %s: %s", path, host, fault)

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
        with log.open("w", encoding="utf-8") as opened:
            opened.write(transcript)
            opened.flush()
            os.fsync(opened.fileno())
        harvested = receipts_in(transcript)
        if harvested:
            path = under / "receipts.ndjson"
            seen = set(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else set()
            fresh = [line for line in harvested if line not in seen]
            with path.open("a", encoding="utf-8") as opened:
                if fresh:
                    opened.write("\n".join(fresh) + "\n")
                    opened.flush()
                    os.fsync(opened.fileno())
        lines = transcript.count("\n")
        logger.info("captured %d log lines for %s (%s)", lines, record.handle, name)
        return harvested

    def evidence(
        self, record: RunRecord, receipts: tuple[str, ...], *, status: str, detail: str = ""
    ) -> None:
        """Append transfer status without rewriting the computational receipt or exit code."""
        self.cache.delivery(record, status)
        stream, job = streamed(record.name or "", handle=record.handle)
        bus = self.streams.setdefault(stream, self.board.receipts(stream))
        data: dict[str, JsonValue] = {
            "handle": record.handle,
            "target": record.target,
            "submitted_at": record.submitted_at,
            "status": status,
            "detail": detail,
            "trials": self._cases(receipts),
        }
        seen = latest(bus.replay(), Topic.EVIDENCE).get(job)
        if seen is None or seen.data != data:
            try:
                publish(bus, stream, Topic.EVIDENCE, job=job, data=data)
            except OSError as fault:
                logger.error(
                    "evidence status is cached but its event could not be saved: %s", fault
                )

    @staticmethod
    def _cases(receipts: tuple[str, ...]) -> list[JsonValue]:
        """Associate valid envelopes only; raw malformed lines stay in the captured log."""
        cases: list[JsonValue] = []
        for line in receipts:
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = envelope.get("trial_receipt") if isinstance(envelope, dict) else None
            if isinstance(payload, dict):
                cases.append(
                    [
                        str(payload.get("run", "")),
                        str(payload.get("case_id") or payload.get("run_id", "")),
                    ]
                )
        return cases

    def verify(
        self, record: RunRecord, job: Run, pulled: str | None, receipts: tuple[str, ...]
    ) -> None:
        """A successful transport is not proof that its declared artifacts arrived."""
        # The trials reader loads its dataframe dependency only when evidence needs checking,
        # matching the lazy reader used by `verdict` rather than taxing every CLI startup.
        from .trials.artifacts import Artifacts

        for line in receipts:
            envelope = json.loads(line)
            payload = envelope.get("trial_receipt") if isinstance(envelope, dict) else None
            if not isinstance(payload, dict) or not isinstance(payload.get("artifacts", {}), dict):
                raise ValueError("malformed trial receipt; raw line preserved in captured log")
        if job.handle.fetch_path and pulled is None:
            raise MissionError("result transfer failed; remote evidence retained")
        native = any(
            Path(token.partition("::")[0]).name.startswith("test_")
            for token in shlex.split(record.script)
        )
        if native and job.handle.fetch_path and not receipts:
            raise MissionError("native trial has no captured receipt; evidence is unverified")
        if receipts:
            if pulled is None:
                referenced = any(
                    isinstance(value, dict)
                    for line in receipts
                    for value in json.loads(line)["trial_receipt"].get("artifacts", {}).values()
                )
                if referenced:
                    raise MissionError("receipt references artifacts but no fetch was declared")
            else:
                Artifacts.verify(
                    receipts,
                    directory=self.board.dispatcher.local(pulled),
                    boundary=self.board.root,
                )

    def asked(self, record: RunRecord) -> Run | None:
        """Ask `record`'s target for its held dispatch again, None while the quota is still full.

        A quota refusal is not a verdict, so a target that still has no room leaves the row
        exactly as it was and the next sweep asks again. Any other refusal is: a queue that does
        not exist and an account without permission answer the same way every twenty minutes
        forever, so the row settles as failed with what the target said and stops asking.

        record: the held run, whose `request` is the dispatch to make.
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
            return None

    def held(self) -> tuple[list[Resumed], list[Held]]:
        """Ask every quota-held dispatch's target for room again, in the order they were held.

        This is what makes a hold a delay rather than a loss. A wave that meets a group's job
        limit used to drop the jobs the queue would not take, and the four missing rows were
        found by counting logs hours later (miyabi-g njobs-g, 2026-09-04); here the requests are
        durable and every pass of the sweep the cron already runs offers them again.

        A request that goes through replaces its own placeholder row: the run is recorded under
        the handle the target gave it, the held row is dropped, and the batch that asked for it
        is told through its own receipts, since the batch's watch reads submissions from there
        and would otherwise never learn the job had gone.
        """
        resumed: list[Resumed] = []
        waiting: list[Held] = []
        for record in reversed(self.cache.live()):
            if record.verdict != vocabulary.HELD:
                continue
            run = self.asked(record)
            if run is None:
                waiting.append(
                    Held(handle=record.handle, target=record.target, reason=record.reason)
                )
                continue
            self.cache.forget(record)
            self.submitted(record, run)
            resumed.append(Resumed(handle=run.handle.id, target=record.target, name=record.name))
            logger.info("held dispatch went through as %s on %s", run.handle.id, record.target)
        return resumed, waiting

    def submitted(self, record: RunRecord, run: Run) -> None:
        """Tell the stream that asked for `record` that its job finally went out.

        A batch publishes every line about its own jobs, and this dispatch was made here rather
        than by that batch, so the line it would have written is written here. Without it a watch
        reading submissions out of the receipts would show the job as held forever while it ran.

        record: the held run this sweep got through.
        run: the dispatch it became.
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

    def once(self) -> MonitorReport:
        """Serialize settlement before reading its cursor, including across monitor processes."""
        path = self.cache.path.with_suffix(".settlement.lock")
        lock = FileLock(path, timeout=0)
        try:
            lock.acquire()
        except Timeout:
            logger.debug("another monitor owns settlement; leaving its cursor untouched")
            return MonitorReport()
        try:
            return self._once()
        finally:
            lock.release()

    def _once(self) -> MonitorReport:
        """Resolve every unsettled run once, harvest the newly terminal ones, report the changes.

        Resolving happens first and by target, so each host is asked once about every handle it
        still owes an answer on instead of once per handle, and a run whose terminal verdict the
        cache already holds is never asked about at all. The harvest then walks the tracked runs
        in the order the cache reports them, so what a run causes and when it is announced does
        not depend on which target answered first.

        A run still in flight is only counted. A run that ended has its results pulled back, its
        output and trial receipts captured beside its own receipts stream, whatever it still
        holds released, its verdict recorded in the study ledger that owns it, and only then its
        reported cursor advanced, so a sweep killed halfway repeats work on the next pass rather
        than losing an outcome. Capturing before releasing is the one ordering that cannot be
        swapped: releasing destroys a rented instance, and its log goes with it. Releasing
        before the cursor moves is what makes that repetition worth wanting, since a pass dying
        between the two leaves the run tracked and the next one cancels the rental again. A run
        whose target could not be resolved has no state, which is the one reason to skip it here,
        and that target is named once in the report rather than once per run on it.

        The pass ends by dropping every pinned source tree no run still owed an outcome runs
        from. A dispatch freezes the code it ships so a later sync cannot rewrite it under a
        running job, and this is the other half of that bargain: without a sweep that lets the
        old trees go, a host under an inode quota fills up with them.

        The pull happens whatever the exit code said, and only the verdict the run reports is
        decided by it. A sweep of 500 trials that dies at 400 leaves 399 immutable receipt
        fragments on the host, which is exactly the crash safety the staged-and-renamed store
        exists to give, and a settle that pulled only a clean run threw those away on the runs
        that most needed them. A partial sweep is also the ordinary end of a metered rental that
        hit its cap, so this is the common case rather than the sad one.
        """
        running = 0
        finished: list[Finished] = []
        failed: list[Failed] = []
        resumed, waiting = self.held()
        fleet = self.board.fleet()
        records = self.cache.tracked()
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
            evidence = current.evidence
            discarded = state.state == vocabulary.CANCELLED and evidence == "unverified"
            if evidence == "not_started" or discarded:
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
                failed.append(Failed(handle=record.handle, target=record.target, reason=detail))
                continue
            stream, name = streamed(record.name or "", handle=record.handle)
            bus = self.streams.setdefault(stream, self.board.receipts(stream))
            history = (
                event
                for event in bus.replay()
                if event.data.get("handle") == record.handle
                and event.data.get("target") == record.target
                and event.data.get("submitted_at") == record.submitted_at
            )
            previous = latest(history, Topic.EVIDENCE).get(name)
            copied = current.evidence in {"copied", "verified"} or (
                previous is not None
                and previous.data.get("handle") == record.handle
                and previous.data.get("target") == record.target
                and previous.data.get("submitted_at") == record.submitted_at
                and previous.data.get("status") in {"copied", "verified"}
            )
            harvested: tuple[str, ...] = ()
            pulled = None
            try:
                if copied:
                    log = directory(self.board, stream) / f"{record.handle}.log"
                    if previous is None and not log.is_file():
                        raise MissionError("copied evidence has no recoverable local receipt log")
                    harvested = (
                        receipts_in(log.read_text(encoding="utf-8")) if log.is_file() else ()
                    )
                    if previous is not None and previous.data.get("trials") and not harvested:
                        raise MissionError("copied trial receipts are missing from the local log")
                    pulled = job.handle.fetch_path
                else:
                    pulled = self.pull(job)
                    harvested = self.capture(record, job)
                self.verify(record, job, pulled, harvested)
            except (MissionError, OSError, ValueError) as fault:
                detail = f"settlement pending; remote evidence retained: {fault}"
                self.evidence(record, harvested, status="pending", detail=detail)
                logger.error("%s on %s: %s", record.handle, record.target, detail)
                failed.append(Failed(handle=record.handle, target=record.target, reason=detail))
                continue
            self.evidence(record, harvested, status="copied")
            if not self.release(job):
                detail = "settlement pending; release failed and will be retried"
                self.evidence(record, harvested, status="copied", detail=detail)
                failed.append(Failed(handle=record.handle, target=record.target, reason=detail))
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
            verdict = Verdict(verdict=state.verdict, exit_code=state.exit_code)
            fleet.settle({job.handle: verdict})
            self.cache.report(self.cache.run(record.handle, record.target), state.verdict)
        self.board.dispatcher.prune_sources()
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
        """Bring a running or settled job's recorded results back, returning where they landed.

        Whatever its verdict was. A crashed run has written everything it wrote before it
        crashed, and the receipt store stages and renames each fragment precisely so that
        those readings survive the trial that killed the process, so a failed job is the one
        whose artifacts are least reproducible and most worth carrying home.

        None when the run recorded no results path at dispatch, or when the pull itself failed
        (a directory the job never wrote, a host that dropped mid-transfer, a provider whose disk
        dies with the rental and never had a delivery to make), so one missing artifact is a
        warning in the log rather than a sweep that dies holding every other job's outcome.
        """
        path = job.handle.fetch_path
        if not path:
            return None
        try:
            job.pull()
        except (ProcessExecutionError, HostUnreachable, MissionError, OSError) as fault:
            return self.unpulled(path, host=job.handle.host, fault=fault)
        return path

    def release(self, job: Run) -> bool:
        """Let a settled run go, so nothing keeps billing for work that already ended.

        A scheduler job releases nothing, since a queue stops charging when the job stops. A
        provider run is cancelled here, which is the only thing that ends the rental, and asking
        twice is expected rather than exceptional, since this pass may be re-running one an
        earlier pass already released. A provider that refuses the cancel is a warning naming the
        run, never the end of a sweep that still owes every other job an outcome.
        """
        try:
            job.release()
        except (MissionError, OSError) as fault:
            logger.warning("could not release %s on %s: %s", job.handle.id, job.handle.host, fault)
            return False
        return True

    def track(self, record: RunRecord, state: JobState, *, detail: str) -> None:
        """Publish what this pass learned about one run into that run's own receipts stream.

        This is what makes a plain submit and a study trial as tracked as a batch job. A run a
        batch owns is skipped, since that batch's own watch already publishes every line about
        it and a second publisher here would double every row.

        Only a move is published. The last state in the stream is what this pass compares
        against, so a sweep that finds nothing new writes nothing at all, which matters because
        this runs on a cron and would otherwise write a line every few minutes forever.

        record: the run as the dispatch cache holds it.
        state: what this pass found it in.
        detail: where its results landed or why it failed, empty while it is still in flight.
        """
        label = record.name or ""
        if is_batched(label) or not self.board.manifest.tracking.on:
            return
        stream, job = streamed(label, handle=record.handle)
        bus = self.streams.setdefault(stream, self.board.receipts(stream))
        seen = latest(bus.replay(), Topic.STATE).get(job)
        moved = {"handle": record.handle, "state": state.state or "", "verdict": state.verdict}
        if seen is not None and seen.data == moved:
            return
        publish(bus, stream, Topic.STATE, job=job, data=moved)
        if state.verdict in vocabulary.TERMINAL:
            publish(
                bus,
                stream,
                Topic.SETTLED,
                job=job,
                data={
                    "handle": record.handle,
                    "verdict": state.verdict,
                    "exit_code": state.exit_code,
                    "detail": detail,
                },
            )

    def watch(self, interval: float) -> Iterator[MonitorReport]:
        """Repeat `once` every `interval` seconds, yielding each pass's report as it lands.

        The foreground loop a person watches. Nothing durable depends on it, since each pass is
        the same self-contained `once` a cron calls.

        interval: seconds to wait between passes.
        """
        while True:
            yield self.once()
            sleep(interval)
