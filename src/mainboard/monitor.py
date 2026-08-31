# The durable sweep behind `mainboard monitor`: one pass over every dispatched job the shared
# cache still owes an outcome for. Everything it reads is durable state, so a periodic cron
# closes out jobs this process never submitted and a remote job's result never depends on the
# agent that dispatched it staying alive to see it end.

from time import sleep
from typing import TYPE_CHECKING

from plumbum.commands.processes import ProcessExecutionError

from .batch.receipts import Topic, latest, publish
from .batch.runner import directory
from .core.errors import MissionError
from .dispatch import vocabulary
from .dispatch.backends.base import route
from .dispatch.dispatcher import Verdict
from .dispatch.evidence import receipts_in
from .dispatch.schedulers import HostUnreachable, short_reason
from .dispatch.shared import logger
from .dispatch.state import DownHost, Failed, Finished, MonitorReport
from .dispatch.vocabulary import JobState
from .tracking import is_batched, streamed

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .batch.receipts import Bus
    from .board import Board, Run
    from .dispatch.state import RunRecord

# The routing answer for the schedulers reached over ssh, the one family whose whole host can be
# asked about in a single query. A provider has no such listing and is asked run by run.
_QUEUED = "ssh-family"

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
        for (target, kind), owned in self.grouped(records).items():
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

    def capture(self, record: RunRecord, job: Run) -> None:
        """Bring a settled run's output and its trial receipts home, beside that run's receipts.

        Only the exit code used to survive a run. The output lived on the host under the state
        dir, or on a rented disk that dies with the rental, and no verb ever brought either back,
        so losing the terminal that dispatched a job lost everything it printed. Here the tail is
        written next to the stream's own `events.ndjson`, which is the directory every reader
        already knows how to find, and the trial receipts the output carries are lifted into that
        stream so `verdict` can settle on them.

        This runs before the release, and that ordering is load-bearing rather than tidy: a
        release cancels a rental, and a cancelled rental's instance takes its log with it, so a
        read afterwards would come back empty on exactly the runs that most need it.

        Nothing here can fail the sweep. A host that went quiet between the probe and the read
        costs its own transcript and no other job's outcome, which is the same bargain `pull`
        already makes.

        record: the run as the dispatch cache holds it.
        job: that run rebuilt, whichever of the two worlds took it.
        """
        if not (transcript := job.transcript()):
            return
        stream, name = streamed(record.name or "", handle=record.handle)
        under = directory(self.board, stream)
        under.mkdir(parents=True, exist_ok=True)
        (under / f"{record.handle}.log").write_text(transcript, encoding="utf-8")
        harvested = receipts_in(transcript)
        if harvested:
            path = under / "receipts.ndjson"
            with path.open("a", encoding="utf-8") as opened:
                opened.write("\n".join(harvested) + "\n")
        lines = transcript.count("\n")
        logger.info("captured %d log lines for %s (%s)", lines, record.handle, name)

    def once(self) -> MonitorReport:
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
        fleet = self.board.fleet()
        records = self.cache.tracked()
        resolved = Sweep(self.board, records)
        for record in records:
            state = resolved.states.get(record)
            if state is None:
                continue
            job = resolved.runs[record]
            stored = self.cache.resolve(record, state.state, state.exit_code, state.verdict)
            if state.verdict not in vocabulary.TERMINAL:
                self.track(record, state, detail="")
                running += 1
                continue
            pulled = self.pull(job)
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
            self.capture(record, job)
            self.release(job)
            verdict = Verdict(verdict=state.verdict, exit_code=state.exit_code)
            fleet.settle({job.handle: verdict})
            self.cache.report(stored, state.verdict)
        return MonitorReport(
            running=running,
            finished=finished,
            failed=failed,
            unreachable_hosts=[
                DownHost(host=host, reason=why) for host, why in resolved.down.items()
            ],
        )

    def pull(self, job: Run) -> str | None:
        """Bring a settled job's recorded results back, returning where they landed.

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

    def release(self, job: Run) -> None:
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
