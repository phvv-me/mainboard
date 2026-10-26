import sys
import time
from contextlib import suppress
from functools import partial
from json import dumps
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ...core.errors import MissionError
from ...core.project import Project
from ...dispatch import constants
from ...dispatch.jobs import vocabulary
from ...dispatch.jobs.evidence import printed
from ...dispatch.schedulers import HostUnreachable, standing
from ...fleet.board import Board
from ...log import logger
from ...render import diverted, plain, progress
from ...settling.digest import Digester
from ...settling.durable import schedule
from ...settling.events import Events
from ...settling.listing import Listing
from ...waiting.vigil import Patience
from ..base import Verbs
from ..options import Output, said

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ...batch.watch import BatchStatus
    from ...dispatch.state import MonitorReport
    from ...waiting.verdicts import StreamVerdict


@Parameter(name="*")
class Waiting(Patience):
    """Whether `show` blocks until its target settles, and how long it may."""

    wait: bool = False
    """block until the target settles."""
    timeout: float = constants.WAIT_SECONDS
    """with `--wait`, give up after this many seconds (exit 2); 0 waits forever."""
    quiet: Annotated[bool, Parameter(show=False)] = False
    """leave out the line naming what is waited on."""


@Parameter(name="*")
class Scope(FrozenModel):
    """Which runs `job list` shows."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    limit: int = 20
    """how many settled runs to show behind the live ones, newest first."""
    project: str = ""
    """only the runs dispatched from this project (its directory under `research/` or
    `packages/`, or `MB_PROJECT`)."""
    batch: str = ""
    """one batch's jobs instead, as `submit --batch` printed its id."""


# The verbs' defaults: compact output, one look rather than a wait, the newest runs of every
# project.
_COMPACT = Output()
_WAITING = Waiting()
_SCOPE = Scope()


def _settled(settled: StreamVerdict, output: Output) -> int:
    """Print one stream's settled rows under its name, and answer its exit code.

    A stream with no rows says why on stderr, since an empty table looks identical whether the
    run has not started, the evidence landed elsewhere, or the harness wrote a shape this verb
    was never taught; stderr keeps a machine-readable stdout exactly what it was.
    """
    output.print_rows(
        [trial.model_dump() for trial in settled.trials],
        title=f"verdict: {settled.stream}",
        columns=_VERDICT_COLUMNS,
    )
    if settled.note:
        logger.info(settled.note)
    if settled.stalled:
        logger.info(f"stalled: {settled.stalled}")
    return settled.code


def _status(status: BatchStatus, output: Output) -> None:
    """Print one pass over a batch, its still-running count in the heading."""
    output.print_rows(
        [job.model_dump() for job in status.jobs],
        title=f"{status.batch}: {status.running} running",
        columns=_STATUS_COLUMNS,
    )
    if status.note:
        logger.info(status.note)


_CHANGE_COLUMNS = ("host", "handle", "outcome", "detail")


_STATUS_COLUMNS = ("job", "target", "handle", "state", "verdict", "detail")


_VERDICT_COLUMNS = (
    "job",
    "handle",
    "target",
    "node",
    "verdict",
    "settled",
    "exit_code",
    "detail",
    "cause",
    "gates",
    "contended",
    "commit",
    "digest",
)


_JOB_COLUMNS = (
    "state",
    "host",
    "project",
    "name",
    "handle",
    "cells",
    "quiet_s",
    "gpu_pct",
    "since",
    "starts",
    "submitted_at",
    "cause",
)


class JobVerbs(Verbs):
    """The `jobs` verbs."""

    @staticmethod
    def _changes(report: MonitorReport) -> list[dict[str, str]]:
        """What moved this pass, one row each.

        That is every job that settled or was re-dispatched and every dispatch a quota is still
        holding. A host that did not answer moved nothing, and the listing's own note names it with
        what to do about its runs.

        A held dispatch is here because nothing else reports it: it has no handle a scheduler knows
        and no verdict to settle, so a sweep silent about it is a job waiting in silence.
        """
        moved = [
            *((run.target, run.handle, "dispatched", run.name) for run in report.resumed),
            *((run.target, run.handle, "held", run.reason) for run in report.held),
            *((job.target, job.handle, "ok", job.pulled_path or "") for job in report.finished),
            *((job.target, job.handle, "failed", job.reason) for job in report.failed),
        ]
        return [dict(zip(_CHANGE_COLUMNS, row, strict=True)) for row in moved]

    @staticmethod
    def _followed[T](passes: Iterator[T], label: str, show: Callable[[T], None]) -> None:
        """Show each pass as it lands until the passes end or the reader interrupts.

        Each pass is taken inside its own progress block rather than iterated over, so the sweep's
        own noise is diverted the way a single pass's is and each report prints as its own
        document.
        """
        with suppress(KeyboardInterrupt, StopIteration):
            while True:
                with progress(label):
                    passed = next(passes)
                show(passed)

    @staticmethod
    def _unprinted(workspace: Board, handle: str, *, host: str) -> int:
        """Say where a run that has printed nothing stands, and exit on whether it still might.

        The durable registry row, always there for a real handle, is what this stands on; the
        backend is asked on top of it for the state and start time only a live scheduler knows, so
        a host that will not answer still gets a line built from what was last recorded.

        host: the alias narrowing a handle recorded on several hosts.
        """
        absent = f"no output on file for {handle}"
        try:
            record = workspace.verdicts().record(handle, host=host)
        except MissionError:
            record = None
        if record is None:
            logger.info(absent)
            return 1
        state = vocabulary.JobState(
            handle=record.handle,
            state=record.state,
            exit_code=record.exit_code,
            verdict=record.verdict or vocabulary.Word.RUNNING,
        )
        with suppress(HostUnreachable, MissionError, OSError):
            state = workspace.job(record.handle, host=record.target).poll()
        if state.verdict in vocabulary.Word.settled():
            logger.info(absent)
            return 1
        logger.info(standing(state, submitted_at=record.submitted_at, host=record.target))
        return 2

    def _listed(self, scope: Scope, *, watch: float, output: Output) -> None:
        """Settle what ended, then print the live runs and the newest settled ones, once."""
        workspace = self.session.board("local")
        with progress("reading job status"):
            report = workspace.monitor().once()
            quiet = {down.host: down.reason for down in report.unreachable_hosts}
            taken = Listing(
                workspace, limit=scope.limit, project=scope.project, quiet=quiet, busy=report.busy
            ).taken()
        if output.json_ and not watch:
            # The periodic pass reads this document: the rows and what the sweep moved.
            sweep = {**report.model_dump(), "changed": report.changed}
            rows = [row.model_dump() for row in taken.rows]
            logger.result(dumps({"jobs": rows, "sweep": sweep, "note": taken.note}, default=str))
            return
        for change in JobVerbs._changes(report):
            logger.info("settled " + " ".join(value for value in change.values() if value))
        output.print_rows(
            [row.model_dump() for row in taken.rows], title="jobs", columns=_JOB_COLUMNS
        )
        if taken.note:
            logger.info(taken.note)

    def _settling(self, every: str) -> None:
        """Install the settling pass into this machine's service manager at `every`."""
        settling = schedule(self.session.workspace_root(), every)
        logger.result(f"{Project().name}: {settling.detail}")
        if settling.fix:
            logger.result(f"{Project().name}: run `{settling.fix}`")

    def _swept(self, batch: str, *, watch: float, output: Output) -> None:
        """Print one pass over `batch`, or one every `watch` seconds until interrupted."""
        watcher = self.session.board("local").watch(batch)
        show_status = partial(_status, output=output)
        if watch:
            JobVerbs._followed(watcher.follow(watch), f"sweeping {batch}", show_status)
            return
        with progress(f"sweeping {batch}"):
            status = watcher.once()
        show_status(status)

    def cancel(self, handle: str, *, on: str = "", output: Output = _COMPACT) -> int:
        """Stop a dispatched job on whatever took it and settle its record in the same pass.

        The verb a provably doomed run needs. Without it a job could only die at its own
        walltime, and killing it over ssh by hand would stop the job while leaving the dispatch
        record claiming it still ran, so a cancellation lost its receipt trail. This kills
        through the backend the run was dispatched under, whether that is pueue, PBS or a
        provider API, writes the terminal verdict, publishes the settled receipt, and ends the
        rental, which is the only thing that stops a provider charging. A run on a host the
        manifest no longer declares, a released rental say, has nothing left to ask, so it
        settles cancelled with that cause and its evidence marked unverified. A run on a declared
        host that does not answer settles cancelled too, and a warning says nothing was stopped
        there.

        Exits the settled code, so a cancelled run exits 1: the stop was deliberate, and a
        completion check must still never call a stopped run complete.

        Args:
            handle: the job to cancel, as `submit` printed it or by the name `jobs` prints.
            on: the host alias narrowing a handle recorded on several hosts.
        """
        with progress(f"cancelling {handle}"):
            settled = self.session.board("local").verdicts().cancel(handle, host=on)
        return _settled(settled, output)

    def collect(self, path: str, *, on: str, json: bool = False) -> None:
        """Collect remote evidence for queries, including runs started directly on that node.

        Complete files are immutable. Conflicts preserve the local copy and fail collection.
        Live event snapshots exclude incomplete records; queries deduplicate overlapping frames.
        A new query sees published files. Collection is not one transaction across servers.

        Args:
            path: workspace-relative results file or directory, using forward slashes on every OS.
            on: declared SSH host; its profile gives the root, a `~` root expanded there when no
                setup has placed it yet. The host runs the exporter on its workspace
                environment's Python, else on the uv-managed CPython its setup put there, never
                on a system interpreter.
            json: print a machine-readable collection summary.
        """
        workspace = self.session.board(on)
        root = workspace.plan(container="none").profile.root
        published = workspace.dispatcher.fetch_path(on, root=root, path=path)
        Output(json_=json).print_record(
            {"host": on, "path": path, "new_files": published}, title="collection"
        )

    def digest(
        self,
        *,
        node: str = "",
        project: str = "",
        limit: int = 400,
        tail: int = 12,
        output: Output = _COMPACT,
    ) -> int:
        """Fold one experiment's runs: counts by state, each failure with its log tail, live runs.

        Settles first, as `list` does, and appends what that pass moved to the event stream, so
        one call says whether a wave is healthy without a `show` per handle. Exits 1 when any
        run in scope failed, so a script can branch on it.

        Args:
            node: the ledger slug the runs serve (a batch spec's `node`); every run when empty.
            project: only the runs dispatched from this project.
            limit: how many settled runs to read behind the live ones.
            tail: how many closing lines of each failed run's log to print.
        """
        with progress("folding the runs"):
            folded = Digester(self.session.board("local")).taken(
                node=node, project=project, limit=limit, tail_lines=tail
            )
        if output.json_:
            logger.result(dumps(folded.model_dump(), default=str))
            return int(bool(folded.failed))
        logger.result(folded.headline())
        for failure in folded.failed:
            logger.result(
                f"\nFAILED {failure.handle} {failure.name} on {failure.host}: {failure.cause}"
            )
            logger.result("\n".join(f"  | {line}" for line in failure.tail))
        if folded.running:
            output.print_rows(
                [row.model_dump() for row in folded.running], title="running", columns=_JOB_COLUMNS
            )
        if folded.note:
            logger.info(folded.note)
        return int(bool(folded.failed))

    def events(self, *, since: int = 0) -> None:
        """Print the job events after byte offset `since`, one JSON line each.

        The next offset is on stderr.

        The stream is what every settling pass appended: a job that settled ok or failed, a
        dispatch a quota held, one that finally got through. A follower keeps the offset and
        reads each change once.

        Args:
            since: the byte offset the last call printed on stderr, 0 for the whole stream.
        """
        fresh, cursor = Events(self.session.workspace_root()).since(since)
        for event in fresh:
            logger.result(event.model_dump_json())
        logger.info(f"offset {cursor}")

    def jobs(
        self,
        *,
        scope: Scope = _SCOPE,
        watch: float = 0.0,
        every: str = "",
        output: Output = _COMPACT,
    ) -> None:
        """Settle every job that ended, then list the live ones and the newest settled ones.

        Each host is asked once about every run it still owes (one `qstat`, `squeue` or `pueue
        status`), so a wave of thirty says which run and which queue, since when, and when the
        scheduler expects a start. A run that ended is settled on the way: its results pulled
        back, its verdict recorded, its rental released. Live runs are never cut by `--limit`;
        what the listing left out is said on stderr, beside what settled this pass. While another
        process holds settlement nothing is asked or settled: the cached states are listed and
        stderr names the holder.

        Args:
            watch: repeat every this many seconds until interrupted.
            every: install the settling pass into this machine's service manager at this period
                (`20m`), so jobs settle with no session open; `0` removes it.
        """
        if every:
            self._settling(every)
            return
        if scope.batch:
            self._swept(scope.batch, watch=watch, output=output)
            return
        self._listed(scope, watch=watch, output=output)
        with suppress(KeyboardInterrupt):
            while watch:
                time.sleep(watch)
                self._listed(scope, watch=watch, output=output)

    def logs(self, handle: str, *, on: str = "") -> int:
        """Print only what a dispatched job printed, whether or not its host still exists.

        Only the exit code used to survive a run: the output lived on the host or on a rented
        disk that dies with the rental, so a lost terminal lost everything the job said. The
        durable sweep now keeps each settled run's tail beside that run's receipts, and this
        reads that copy first, falling back to the backend for a run still in flight. The frame a
        rented run carries its receipts home in is this tool's and not the job's, so it is left
        out; `verdict` reads the receipts themselves.

        An empty log has two entirely different causes, so a run that printed nothing answers
        with where it stands instead: its verdict, the scheduler's own state word, how long it
        has been waiting, and what the backend says about when it will run, which on PBS is the
        estimated start time the server reports. That line is the difference between a job that
        is queued behind a full cluster and one that started and said nothing.

        Exits 0 having printed output, 2 for a run still in flight that has printed none, and 1
        when nothing was captured and nothing is coming, so a script can tell the three apart.

        Args:
            handle: the job to read, as `submit` printed it or by the name `jobs` prints.
            on: the host alias narrowing a handle recorded on several hosts.
        """
        workspace = self.session.board("local")
        captured = printed(workspace.verdicts().captured(handle, host=on))
        if not captured.strip():
            return JobVerbs._unprinted(workspace, handle, host=on)
        # A job that coloured its output for a terminal it never had leaves escape codes in
        # the capture; a pipe or a file gets the plain text, a terminal gets the colours.
        shown = captured if sys.stdout.isatty() else plain(captured)
        logger.result(shown, end="" if shown.endswith("\n") else "\n")
        return 0

    def monitor(self, *, json: bool = False) -> None:
        """The settling pass periodic runners installed before `job list --every` call.

        Args:
            json: print the listing and the sweep as JSON, what those runners read.
        """
        self.jobs(output=Output(json_=json))

    def show(
        self,
        target: str,
        *,
        on: str = "",
        run: str = "",
        waiting: Waiting = _WAITING,
        output: Output = _COMPACT,
    ) -> int:
        """Refresh one job or batch, print its recorded outcome, and exit with it.

        One row per trial: its outcome, its gate sweep and the ledger node it serves, never a
        scheduler's or a session's memory. Exit 0 when every row settled clean, 1 on a failure,
        2 while anything is in flight (or `--wait` timed out), 3 when the receipts prove
        nothing, 4 when `--wait` saw a job stall.

        Only the requested dispatch or batch is refreshed; receipt files and directories are
        read offline. Rental deadlines remain enforced across the workspace. `--wait` repeats
        that settlement until it finishes, with each test cell and a heartbeat on stderr.

        Args:
            target: a handle or name as `submit`/`list` print it, a batch id, a receipts stream id,
                directory or file.
            on: the host alias narrowing a handle recorded on several hosts.
            run: which run of a receipts store to score, its newest when unset.
        """
        verdicts = self.session.board("local").verdicts()
        if not waiting.wait:
            with progress(f"reading {target}"):
                settled = verdicts.refresh(target, host=on, run=run)
            return _settled(settled, output)
        if not waiting.quiet:
            logger.info(f"waiting on {target}")
        with diverted():
            settled = verdicts.wait(target, host=on, run=run, patience=waiting, say=said)
        return _settled(settled, output)
