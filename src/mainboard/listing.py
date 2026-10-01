# What `mainboard job list` shows: every run still in flight with its scheduler's word on it now,
# then the most recently settled ones.
#
# The verb used to print the cache's twenty newest rows with their memoized state, so a wave of
# thirty five jobs showed fifteen invisible runs and a blank state on every live one, and the
# operator ran `qstat` over ssh to learn the debug queue was starting two at a time. So live runs
# are never truncated, carry their backend's live state, and any truncation left says so.

from collections import Counter
from typing import TYPE_CHECKING

from patos import FrozenModel

from .core.project import Project
from .diagnosis import reason
from .dispatch import vocabulary
from .jobs.beacon import Progress
from .monitor import Sweep
from .pulse import Pulses

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .board import Board
    from .dispatch.state import RunRecord
    from .dispatch.vocabulary import JobState
    from .pulse import Pulse

# The verdicts whose row is worth a reason. A cancel is somebody's decision and a skip was never
# dispatched, so neither has output to explain itself with.
_FAILURES = frozenset({vocabulary.FAILED, vocabulary.TIMEOUT})


class JobRow(FrozenModel):
    """One dispatched run as the listing prints it.

    state: the backend's live word while in flight, the settled verdict once ended.
    name: the run's label, else the job script it was submitted as.
    since: when the run entered that state per the backend, else the dispatch time; empty once
        settled.
    starts: when the backend expects a queued run to start, empty where it estimates none.
    cause: a settled failure's last meaningful output line; empty for a live run, which has
        printed nothing home yet, and for a clean one.
    cells: a running test job's cells landed out of its total, `3/12`, plus how many failed
        when any did; empty where its log reports none.
    quiet_s: seconds since a running job's output last grew, empty until a look has seen it
        twice, since one look cannot tell silence from a job that just printed.
    gpu_pct: the busiest card on a running job's host, empty where that is not one cheap
        command away (a cluster's login node, a rented machine).
    """

    state: str
    host: str
    project: str = ""
    name: str
    handle: str
    since: str = ""
    starts: str = ""
    submitted_at: str
    cause: str = ""
    cells: str = ""
    quiet_s: int | None = None
    gpu_pct: int | None = None


class Listed(FrozenModel):
    """One listing: the rows it prints and the one line saying what it could not.

    rows: every live run, then the settled tail, newest first inside each.
    note: what the table leaves out and which hosts went quiet, empty when nothing is missing.
    """

    rows: tuple[JobRow, ...]
    note: str = ""


class Listing:
    """Every live dispatched run, resolved against its own host, and the settled tail behind it.

    Live rows are never cut to a limit nor answered from memory: a job last seen queued may be
    running now, which is why somebody types this verb. Resolution is the sweep's batched probe,
    one query per host (a thirty five job wave on one cluster is one `qstat`). A quiet host costs
    only its own runs their live state: they fall back to the cache and the note names the host.
    """

    def __init__(
        self,
        board: Board,
        *,
        limit: int,
        project: str = "",
        pulses: Pulses | None = None,
        quiet: Mapping[str, str] | None = None,
    ) -> None:
        """limit: how many settled runs to show behind the live ones.

        project: only the runs dispatched from this project; every run when empty.

        pulses: the look at running jobs' output and cards, the workspace's own when None.

        quiet: the hosts the settling pass just found unanswering, each with why, which this
            listing reports without knocking on them a second time.
        """
        self.board = board
        self.limit = limit
        self.project = project
        self.cache = board.dispatcher.cache
        self.pulses = pulses or Pulses(board)
        self.quiet = quiet or {}

    def taken(self) -> Listed:
        """The listing as it stands: every live run resolved now, then the settled tail."""
        live = self.cache.live(self.project)
        resolved = Sweep(self.board, live, down=self.quiet)
        running = [
            record
            for record in live
            if (state := resolved.states.get(record))
            and state.verdict == vocabulary.RUNNING
            and state.stage != vocabulary.QUEUED
        ]
        pulses = self.pulses.taken(running)
        unasked = {record for record in live if record.target in resolved.down}
        rows = [
            *(
                self.flying(
                    record,
                    resolved.states.get(record),
                    pulses.get(record),
                    unanswered=record in unasked,
                )
                for record in live
            ),
            *(self.landed(record) for record in self.cache.settled(self.limit, self.project)),
        ]
        owed = Counter(record.target for record in unasked)
        return Listed(
            rows=tuple(rows), note=self.note(shown=len(rows), quiet=resolved.down, owed=owed)
        )

    @staticmethod
    def flying(
        record: RunRecord, state: JobState | None, pulse: Pulse | None, *, unanswered: bool = False
    ) -> JobRow:
        """One live run's row, from what its host just said or from the cache when it went quiet.

        The state is the lifecycle's live word, `queued`, `running`, or `finished` for a job its
        queue is done with but the sweep has not settled, the distinctions a reader wants. A
        backend mapping neither stage leaves its raw word (`Queued` in a pueue status). A running
        job carries its pulse: cells landed, output quiet time, and its host's busiest card.

        unanswered: the run's host was not asked or did not answer, so the row says what was
            last recorded instead of reading as a run that stopped moving.
        """
        live = state.phase if state else ""
        progress = pulse.progress if pulse else Progress()
        failed = f" ({progress.failed} failed)" if progress.failed else ""
        recorded = record.state or record.verdict
        last = f"last recorded {recorded}" if recorded else "no state recorded since dispatch"
        return JobRow(
            state=live or record.state or vocabulary.UNKNOWN,
            host=record.target,
            project=record.project,
            name=record.name or record.script,
            handle=record.handle,
            since=(state.since if state else "") or record.submitted_at,
            starts=state.estimated_start if state else "",
            submitted_at=record.submitted_at,
            cause=f"{record.target} not heard from; {last}" if unanswered else "",
            cells=f"{progress.counted}{failed}",
            quiet_s=pulse.quiet_s if pulse else None,
            gpu_pct=pulse.gpu_pct if pulse else None,
        )

    def landed(self, record: RunRecord) -> JobRow:
        """One settled run's row from the cache, since a terminal verdict cannot move.

        A failure carries its `diagnosis.reason`, off the log the sweep already brought home.
        """
        verdict = record.verdict or vocabulary.UNKNOWN
        return JobRow(
            state=verdict,
            host=record.target,
            project=record.project,
            name=record.name or record.script,
            handle=record.handle,
            submitted_at=record.submitted_at,
            cause=reason(self.board, record) if verdict in _FAILURES else "",
        )

    def silence(self, host: str, why: str, live: int) -> str:
        """One quiet host's line: why it is quiet, and what to do about the runs it holds.

        A host the manifest dropped already says which commands settle its runs.

        live: how many live runs the host holds; with none it holds only ended ones.
        """
        if not self.board.declares(host):
            return why
        waiting = (
            f"its {live} live run(s) keep what was last recorded and settle once it answers; "
            f"`{Project().name} job cancel <handle>` gives one up now without stopping it there"
            if live
            else "the ended runs it still holds settle once it answers"
        )
        return f"{host} did not answer ({why}): {waiting}"

    def note(self, *, shown: int, quiet: Mapping[str, str], owed: Mapping[str, int]) -> str:
        """What this listing leaves out and which hosts went quiet, empty when it leaves nothing.

        A table silently stopping at its limit is the fault this verb was fixed for, and a quiet
        host is named rather than left to read as runs that stopped moving, with what to do
        about the runs it still owes.

        owed: how many live runs each quiet host holds.
        """
        total = self.cache.total()
        said = [self.silence(host, why, owed.get(host, 0)) for host, why in quiet.items()]
        if shown < total:
            said.insert(
                0,
                f"showing {shown} of {total} runs: every live one, and the {self.limit} most "
                f"recently settled behind them",
            )
        return "; ".join(said)
