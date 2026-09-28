# The study-to-dispatch join, the one place that reads both sides: `StudyLedger` knows which
# handles a study dispatched, `Cache` what dispatch resolved for each on its own cadence. The key
# is `RunRecord.name`, the `study:<study_id>[/<trial>]` label `Fleet.submit_all` stamps. Both
# sides stamp `datetime.now(UTC).isoformat()` (`dispatch.state.cache.now`, `study._now`), so
# timestamps compare as plain strings.

from collections import Counter
from typing import TYPE_CHECKING

from patos import FrozenModel

from .identity import labelled_study
from .study import Progress, StudyLedger

if TYPE_CHECKING:
    from ..dispatch.state.cache import Cache, RunRecord
    from .study import Study


class StudySummary(FrozenModel):
    """One study's read-only shape, folded from its ledger file and joined dispatch verdicts.

    study_id: the study the ledger's rows are recorded under.
    name: the human label from its `created` event, `None` when none was recorded.
    counts: how many handles hold each resolved state word (`submitted`, `ok`, `vanished`, ...),
        merged per handle exactly as `study_progress` resolves them.
    oldest_at: the earliest event timestamp, `None` for an empty ledger (so is `newest_at`).
    """

    study_id: str
    name: str | None = None
    counts: dict[str, int]
    oldest_at: str | None = None
    newest_at: str | None = None


def study_runs(cache: Cache, study_id: str, *, limit: int = 100_000) -> list[RunRecord]:
    """Every dispatch run labelled for this study (with or without a trial), newest first.

    limit: how many of the newest dispatch rows `Cache.recent` scans; raise it for a study that
        outlives that window.
    """
    return [run for run in cache.recent(limit) if labelled_study(run.name) == study_id]


def _merge_states(cache: Cache, ledger: StudyLedger, study_id: str) -> dict[str, str]:
    """Each handle's resolved state: the ledger's own fold, dispatch's terminal verdict winning.

    A durable dispatch monitor can terminalize a run without writing back to the ledger, so a
    resolved `RunRecord.verdict` always wins. A handle only dispatch recorded (a crash between
    the two writes) still counts as `submitted`, since the record proves it was dispatched.
    """
    states = dict(ledger.statuses())
    for run in study_runs(cache, study_id):
        if run.verdict is not None:
            states[run.handle] = run.verdict
        else:
            states.setdefault(run.handle, "submitted")
    return states


def study_progress(cache: Cache, ledger: StudyLedger, study: Study) -> Progress:
    """`study`'s live trial counts, dispatch's resolved verdicts merged over its `ledger`."""
    return Progress.fold(_merge_states(cache, ledger, study.study_id))


def overview(cache: Cache) -> list[StudySummary]:
    """One summary per study the workspace lake records, in study id order.

    Counts are joined as `study_progress` joins, but per distinct state word rather than its four
    fixed buckets.
    """
    rows = cache.session.rows("SELECT DISTINCT study FROM lake.studies ORDER BY study")
    return [_summarize(cache, study) for (study,) in rows]


def _summarize(cache: Cache, study: str) -> StudySummary:
    ledger = StudyLedger(cache.session, study)
    events = ledger.events()
    timestamps = [event.at for event in events]
    return StudySummary(
        study_id=study,
        name=next((event.name for event in events if event.kind == "created"), None),
        counts=dict(Counter(_merge_states(cache, ledger, study).values())),
        oldest_at=min(timestamps, default=None),
        newest_at=max(timestamps, default=None),
    )
