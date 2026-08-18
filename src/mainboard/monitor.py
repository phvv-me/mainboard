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

    from .board import Board, Job
    from .dispatch.state import RunRecord


class Monitor:
    """The durable pass over every dispatched job still owed an outcome.

    One `once` probes each unsettled run in the dispatch cache, pulls back the results of
    whatever just finished, records each fresh verdict in the study ledger that owns it, and
    advances that run's reported cursor so the next pass over the same jobs says nothing at all.
    A host that does not answer is reported once with why and its jobs are left for the next
    pass, so a dead host never fails the sweep.
    """

    def __init__(self, board: Board) -> None:
        """board: the workspace board whose dispatch cache and study ledgers the sweep settles."""
        self.board = board
        self.cache = board.dispatcher.cache

    def once(self) -> MonitorReport:
        """Resolve every unsettled run once, harvest the newly terminal ones, report the changes.

        Each tracked run is rebuilt as a `Job` and asked for its current state, which is one
        probe unless the cache already holds a terminal verdict; a run still in flight is only
        counted. A run that ended has its results pulled back (or its cause read off the exit
        code), its verdict recorded in the study ledger that owns it, and only then its reported
        cursor advanced, so a sweep killed halfway repeats work on the next pass rather than
        losing an outcome. Rebuilding and probing share one `try`, since a host with no declared
        root and a host that will not answer are the same thing here, one target this pass cannot
        resolve and reports instead of raising.
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
            except (HostUnreachable, MissionError) as fault:
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
            verdict = Verdict(verdict=state.verdict, exit_code=state.exit_code)
            fleet.settle({job.handle: verdict})
            self.cache.report(stored, state.verdict)
        return MonitorReport(
            running=running,
            finished=finished,
            failed=failed,
            unreachable_hosts=[DownHost(host=host, reason=why) for host, why in down.items()],
        )

    def current(self, record: RunRecord, job: Job) -> JobState:
        """`record`'s state now: its cached terminal verdict, else one fresh scheduler probe.

        A terminal verdict can never change, so it is read straight from the cache and the host
        is never touched for it, which is also what keeps a finished job the queue has already
        forgotten from reading back as vanished. Anything else costs one probe, which raises
        `HostUnreachable` when the host does not answer.
        """
        verdict = record.verdict
        if verdict is not None and verdict in vocabulary.TERMINAL:
            return JobState(
                handle=record.handle,
                state=record.state,
                exit_code=record.exit_code,
                verdict=verdict,
            )
        return self.board.dispatcher.state(job.handle)

    def pull(self, job: Job) -> str | None:
        """Bring a finished job's recorded results back, returning where they landed.

        None when the run recorded no results path at dispatch, or when the pull itself failed
        (a directory the job never wrote, a host that dropped mid-transfer), so one missing
        artifact is a warning in the log rather than a sweep that dies holding every other
        job's outcome.
        """
        path = job.handle.fetch_path
        if not path:
            return None
        try:
            job.pull()
        except (ProcessExecutionError, HostUnreachable) as fault:
            logger.warning("could not pull %s from %s: %s", path, job.handle.host, fault)
            return None
        return path

    def watch(self, interval: float) -> Iterator[MonitorReport]:
        """Repeat `once` every `interval` seconds, yielding each pass's report as it lands.

        The foreground loop a person watches. Nothing durable depends on it, since each pass is
        the same self-contained `once` a cron calls.

        interval: seconds to wait between passes.
        """
        while True:
            yield self.once()
            sleep(interval)
