# The durable sweep behind `mainboard monitor`: one pass over every dispatched job the shared
# cache still owes an outcome for. Everything it reads is durable state, so a periodic cron
# closes out jobs this process never submitted and a remote job's result never depends on the
# agent that dispatched it staying alive to see it end.

from time import sleep
from typing import TYPE_CHECKING

from plumbum.commands.processes import ProcessExecutionError

from .core.errors import MissionError
from .dispatch import verdicts as vocabulary
from .dispatch.dispatcher import Verdict
from .dispatch.schedulers import HostUnreachable, JobState, short_reason
from .dispatch.shared import logger
from .dispatch.state import DownHost, Failed, Finished, MonitorReport

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .board import Board, Run
    from .dispatch.state import RunRecord


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

    def once(self) -> MonitorReport:
        """Resolve every unsettled run once, harvest the newly terminal ones, report the changes.

        Each tracked run is rebuilt as the kind of run it was dispatched as and asked for its
        current state, which is one probe unless the cache already holds a terminal verdict; a
        run still in flight is only counted. A run that ended has its results pulled back (or its
        cause read off the exit code), whatever it still holds released, its verdict recorded in
        the study ledger that owns it, and only then its reported cursor advanced, so a sweep
        killed halfway repeats work on the next pass rather than losing an outcome. Releasing
        before the cursor moves is what makes that repetition worth wanting, since a pass dying
        between the two leaves the run tracked and the next one cancels the rental again.
        Rebuilding and probing share one `try`, since a host with no declared root, a host that
        will not answer and a provider API that refuses are the same thing here, one target this
        pass cannot resolve and reports instead of raising.
        """
        running = 0
        finished: list[Finished] = []
        failed: list[Failed] = []
        down: dict[str, str] = {}
        fleet = self.board.fleet()
        for record in self.cache.tracked():
            if record.target in down:  # one down host is reported once, not once per job on it
                continue
            try:
                job = self.board.job(record.handle, host=record.target)
                state = self.current(record, job)
            except (HostUnreachable, MissionError, OSError) as fault:
                down[record.target] = str(fault)
                continue
            stored = self.cache.resolve(record, state.state, state.exit_code, state.verdict)
            if state.verdict not in vocabulary.TERMINAL:
                running += 1
                continue
            if state.verdict == vocabulary.OK:
                pulled = self.pull(job)
                finished.append(
                    Finished(handle=record.handle, target=record.target, pulled_path=pulled)
                )
            else:
                reason = short_reason(state.verdict, state.exit_code)
                failed.append(Failed(handle=record.handle, target=record.target, reason=reason))
            self.release(job)
            verdict = Verdict(verdict=state.verdict, exit_code=state.exit_code)
            fleet.settle({job.handle: verdict})
            self.cache.report(stored, state.verdict)
        return MonitorReport(
            running=running,
            finished=finished,
            failed=failed,
            unreachable_hosts=[DownHost(host=host, reason=why) for host, why in down.items()],
        )

    def current(self, record: RunRecord, job: Run) -> JobState:
        """`record`'s state now: its cached terminal verdict, else one fresh probe of its target.

        A terminal verdict can never change, so it is read straight from the cache and the target
        is never touched for it, which is also what keeps a finished job the queue has already
        forgotten from reading back as vanished. Anything else costs one probe, which raises
        `HostUnreachable` when a host does not answer and asks the backend when a provider owns
        the run.
        """
        verdict = record.verdict
        if verdict is not None and verdict in vocabulary.TERMINAL:
            return JobState(
                handle=record.handle,
                state=record.state,
                exit_code=record.exit_code,
                verdict=verdict,
            )
        return job.poll()

    def pull(self, job: Run) -> str | None:
        """Bring a finished job's recorded results back, returning where they landed.

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
            logger.warning("could not pull %s from %s: %s", path, job.handle.host, fault)
            return None
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

    def watch(self, interval: float) -> Iterator[MonitorReport]:
        """Repeat `once` every `interval` seconds, yielding each pass's report as it lands.

        The foreground loop a person watches. Nothing durable depends on it, since each pass is
        the same self-contained `once` a cron calls.

        interval: seconds to wait between passes.
        """
        while True:
            yield self.once()
            sleep(interval)
