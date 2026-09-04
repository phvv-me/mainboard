# What `mainboard jobs` shows: every dispatched run still in flight, with what its own scheduler
# says about it right now, and the most recently settled ones behind them.
#
# The verb used to print the twenty newest rows of the dispatch cache with whatever state had
# last been memoized in them, which on a wave of thirty five jobs meant fifteen invisible runs
# and a blank state column for every live one. The operator went to ssh and `qstat` by hand to
# learn that the debug queue was starting two jobs at a time. So the live runs are never
# truncated, they carry their backend's own live state rather than a remembered one, and the
# truncation that is left says so out loud.

from typing import TYPE_CHECKING

from patos import FrozenModel

from .dispatch import vocabulary
from .monitor import Sweep

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .board import Board
    from .dispatch.state import RunRecord
    from .dispatch.vocabulary import JobState


class JobRow(FrozenModel):
    """One dispatched run as the listing prints it.

    state: what the run is doing now, its backend's live word for a job still in flight and its
        settled verdict once it has ended.
    host: the target it was dispatched to.
    name: the run's label, the job script it was submitted as when the dispatch named none.
    handle: the scheduler or provider handle.
    since: when the run entered that state, as the backend reports it, falling back to the
        dispatch time while the backend says nothing about when; empty once it has settled.
    starts: when the backend expects a queued run to start, empty where it estimates none.
    submitted_at: when the run was dispatched.
    """

    state: str
    host: str
    name: str
    handle: str
    since: str = ""
    starts: str = ""
    submitted_at: str


class Listed(FrozenModel):
    """One listing: the rows it prints and the one line saying what it could not.

    rows: every live run, then the settled tail, newest first inside each.
    note: what the table leaves out and which hosts went quiet, empty when it holds everything
        and every host answered.
    """

    rows: tuple[JobRow, ...]
    note: str = ""


class Listing:
    """Every live dispatched run, resolved against its own host, and the settled tail behind it.

    The live rows are the point, so they are never cut to a limit and never answered from memory:
    a job the cache last saw queued may be running now, and that difference is the whole reason
    somebody types this verb. Resolving them costs one query per host rather than one per job,
    the same batched probe the durable sweep already makes, so a wave of thirty five jobs on one
    cluster is one `qstat`. A host that will not answer costs its own runs their live state and
    nothing else: those rows fall back to what the cache remembers and the note names the host.
    """

    def __init__(self, board: Board, *, limit: int) -> None:
        """board: the workspace whose cache holds the runs and whose hosts answer for them.

        limit: how many settled runs to show behind the live ones.
        """
        self.board = board
        self.limit = limit
        self.cache = board.dispatcher.cache

    def taken(self) -> Listed:
        """The listing as it stands: every live run resolved now, then the settled tail."""
        live = self.cache.live()
        resolved = Sweep(self.board, live)
        rows = [
            *(self.flying(record, resolved.states.get(record)) for record in live),
            *(self.landed(record) for record in self.cache.settled(self.limit)),
        ]
        return Listed(rows=tuple(rows), note=self.note(shown=len(rows), quiet=resolved.down))

    @staticmethod
    def flying(record: RunRecord, state: JobState | None) -> JobRow:
        """One live run's row, from what its host just said or from the cache when it went quiet.

        The state column is the lifecycle's own live word where the backend maps onto it, since
        `queued` and `running` are the distinction a person is reading this table for; the raw
        state the backend spells it with stands in until then (`Q` on PBS, `Queued` in a pueue
        status), and a host that answered nothing leaves the cache's memory of it.
        """
        live = (state.stage or state.state or "") if state else ""
        return JobRow(
            state=live or record.state or vocabulary.UNKNOWN,
            host=record.target,
            name=record.name or record.script,
            handle=record.handle,
            since=(state.since if state else "") or record.submitted_at,
            starts=state.estimated_start if state else "",
            submitted_at=record.submitted_at,
        )

    @staticmethod
    def landed(record: RunRecord) -> JobRow:
        """One settled run's row, read from the cache, since a terminal verdict cannot move."""
        return JobRow(
            state=record.verdict or vocabulary.UNKNOWN,
            host=record.target,
            name=record.name or record.script,
            handle=record.handle,
            submitted_at=record.submitted_at,
        )

    def note(self, *, shown: int, quiet: Mapping[str, str]) -> str:
        """What this listing leaves out and which hosts went quiet, empty when it leaves nothing.

        A table that silently stops at its limit is the fault this verb was fixed for, so the
        count is stated whenever anything was cut, and a host whose live state is missing is
        named rather than left to read as a run that stopped moving.
        """
        total = self.cache.total()
        said = [f"{host} did not answer: {why}" for host, why in quiet.items()]
        if shown < total:
            said.insert(
                0,
                f"showing {shown} of {total} runs: every live one, and the {self.limit} most "
                f"recently settled behind them",
            )
        return "; ".join(said)
