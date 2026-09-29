# How often each host or provider delivered a machine that ran the job, from the run registry.
#
# A job failing on its own (an exit code) says nothing about the machine; a rental that never got
# the job started (a landing that failed, an image that would not pull), a machine that vanished
# under it (taken back, destroyed, lost) or a run that ended with no exit code at all says
# everything. So the share counted is the runs whose command ran to an exit, against every run
# that settled, per target, over a window. The registry is the log: every dispatch leaves a
# record whatever broke, so nothing new is written to measure this.
#
# A setup failure is charged to the target even when mb caused it (a lock input missing from the
# mirror, a bid Vast refuses), which is the point: the number says what a dispatch there costs
# in practice, and a jump after a release is a regression to look for.

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from patos import FrozenModel

from .dispatch import vocabulary

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .dispatch.state.cache import RunRecord

# Verdicts that say how the command ended, whatever its exit code.
_RAN = frozenset({vocabulary.OK, vocabulary.TIMEOUT})
# Verdicts that settle a run without judging the machine: asked to stop, or never dispatched.
_UNJUDGED = frozenset({vocabulary.CANCELLED, vocabulary.HELD, vocabulary.SKIPPED})


class Reliability(FrozenModel):
    """One target's record over the window.

    ran: runs whose command ran to an exit, its own failures included.
    unstarted: rentals or dispatches whose job never started (landing or setup failed).
    lost: runs whose machine vanished under them, or that ended with no exit code.
    unsettled: runs still in flight, or cancelled, held or skipped, which judge nothing.
    delivered: `ran` over every judged run, None when none was.
    """

    target: str
    kind: str
    runs: int
    ran: int
    unstarted: int
    lost: int
    unsettled: int
    delivered: float | None


def judged(run: RunRecord) -> str:
    """Which of `ran`, `unstarted`, `lost` or `unsettled` a run counts toward."""
    verdict = run.verdict or ""
    if verdict in _RAN or (verdict == vocabulary.FAILED and run.exit_code is not None):
        return "ran"
    if verdict == vocabulary.FAILED and run.evidence == "not_started":
        return "unstarted"
    if verdict in {vocabulary.FAILED, vocabulary.VANISHED, vocabulary.UNKNOWN}:
        return "lost"
    return "unsettled"


def reliability(runs: Iterable[RunRecord]) -> list[Reliability]:
    """Each target's counts, least reliable first so the one to avoid leads."""
    counted: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for run in runs:
        counted[run.target, run.kind][judged(run)] += 1
    rows = []
    for (target, kind), counts in counted.items():
        judged_runs = counts["ran"] + counts["unstarted"] + counts["lost"]
        rows.append(
            Reliability(
                target=target,
                kind=kind,
                runs=judged_runs + counts["unsettled"],
                ran=counts["ran"],
                unstarted=counts["unstarted"],
                lost=counts["lost"],
                unsettled=counts["unsettled"],
                delivered=round(counts["ran"] / judged_runs, 3) if judged_runs else None,
            )
        )
    return sorted(rows, key=lambda row: (row.delivered is None, row.delivered or 0.0, row.target))


def window(days: int) -> str:
    """The ISO stamp `days` ago, the lower bound a registry query takes."""
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def by_kind(rows: Iterable[Reliability]) -> dict[str, float]:
    """Each provider kind's delivered share over all its targets, for kinds with a judged run."""
    ran: dict[str, int] = defaultdict(int)
    judged_runs: dict[str, int] = defaultdict(int)
    for row in rows:
        ran[row.kind] += row.ran
        judged_runs[row.kind] += row.ran + row.unstarted + row.lost
    return {kind: round(ran[kind] / total, 3) for kind, total in judged_runs.items() if total}
