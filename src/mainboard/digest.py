# WHAT A WAVE OF JOBS IS DOING, IN ONE READ.
#
# Following forty five jobs meant `job show` on one handle after another, each dumping its row,
# and the failure among them was found by reading them all. The digest is the listing narrowed to
# one experiment and folded: how many runs are in each state, every failure with its cause and
# the last lines of its log, and the runs in flight with their progress. It settles first, as the
# listing does, so it is always the state of the hosts now, and the monitor appends what that
# pass moved to the event stream, so a follower of the stream sees what the digest saw.
# Settling is scoped to the experiment asked about, so a backlog of other runs' pending
# transfers never slows it.

from collections import Counter

from patos import FrozenModel

from .board import Board
from .diagnosis import tail
from .dispatch import vocabulary
from .listing import JobRow, Listing

# The states a settled run can end in that were not a success nor somebody's decision.
_FAILURES = frozenset({vocabulary.FAILED, vocabulary.TIMEOUT, vocabulary.VANISHED})


class Failure(FrozenModel):
    """One run that ended badly: where, why in one line, and what it printed last."""

    handle: str
    name: str
    host: str
    state: str
    cause: str
    tail: tuple[str, ...]


class Digest(FrozenModel):
    """One experiment's runs folded: the counts by state, the failures, the runs in flight.

    scope: the ledger slug the digest covers, empty for every run.
    counts: runs by state, the backend's live word or the settled verdict.
    running: the runs executing now with their cells landed, quiet time and card load.
    note: what the listing could not say (a quiet host, a truncated tail).
    """

    scope: str
    counts: dict[str, int]
    failed: tuple[Failure, ...]
    running: tuple[JobRow, ...]
    note: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def headline(self) -> str:
        """`probe_zoo: 45 runs: 12 ok, 1 failed, 1 running, 31 queued`."""
        states = ", ".join(f"{count} {state}" for state, count in sorted(self.counts.items()))
        return f"{self.scope or 'all runs'}: {self.total} runs: {states}"


class Digester:
    """Settles, records the events that pass moved, and folds one experiment's runs."""

    def __init__(self, board: Board) -> None:
        self.board = board

    def taken(
        self, *, node: str = "", project: str = "", limit: int = 400, tail_lines: int = 12
    ) -> Digest:
        """The digest of runs serving ledger slug `node` (every run when empty).

        limit: how many settled runs the listing reads behind the live ones.
        tail_lines: how many closing lines of a failed run's log to keep.
        """
        scope = (lambda run: run.node == node) if node else None
        report = self.board.monitor().once(scope=scope)
        quiet = {down.host: down.reason for down in report.unreachable_hosts}
        listed = Listing(
            self.board, limit=limit, project=project, quiet=quiet, busy=report.busy
        ).taken()
        cache = self.board.dispatcher.cache
        rows = [
            row for row in listed.rows if not node or cache.run(row.handle, row.host).node == node
        ]
        failed = tuple(
            Failure(
                handle=row.handle,
                name=row.name,
                host=row.host,
                state=row.state,
                cause=row.cause,
                tail=tuple(tail(self.board, cache.run(row.handle, row.host), tail_lines)),
            )
            for row in rows
            if row.state in _FAILURES
        )
        return Digest(
            scope=node,
            counts=dict(Counter(row.state for row in rows)),
            failed=failed,
            running=tuple(row for row in rows if row.state == vocabulary.RUNNING),
            note=listed.note,
        )
