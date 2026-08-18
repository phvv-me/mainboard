# The study-to-dispatch join. `Study`/`StudyLedger` know a study's own shape (which handles it
# dispatched, what the ledger itself folds them to); `Cache` knows what dispatch resolved for
# each handle, independently, on its own polling cadence. Neither side reaches for the other, so
# this module is the one place that reads both and reconciles them into one picture.
#
# The join key is `RunRecord.name`: `Fleet.submit_all` stamps every trial `study:<study_id>`
# (or, for a future caller that wants to distinguish trials within one study by name,
# `study:<study_id>/<trial>`), and dispatch never parses that string itself. Every timestamp on
# either side of the join is `datetime.now(UTC).isoformat()` (`RunRecord.submitted_at` via
# `dispatch.state.cache.now`, `StudyEvent.at` via this package's own `study._now`), so oldest and
# newest activity compare as plain strings without parsing a single one.

from collections import Counter
from typing import TYPE_CHECKING

from patos import FrozenModel

from .identity import labelled_study
from .study import Progress, StudyLedger

if TYPE_CHECKING:
    from pathlib import Path

    from ..dispatch.state.cache import Cache, RunRecord
    from .study import Study


class StudySummary(FrozenModel):
    """One study's read-only shape, folded from its ledger file and joined dispatch verdicts.

    study_id: parsed from the ledger's `<study_id>.jsonl` filename.
    name: the study's human label, read from its `created` event; `None` when that event was
        never recorded (an older ledger, or one a caller never called `StudyLedger.created` on).
    counts: each resolved state word (`submitted`, `ok`, `failed`, `vanished`, ...) mapped to
        how many handles currently hold it, dispatch's terminal verdict outranking the ledger's
        own fold per handle exactly as `study_progress` resolves it.
    oldest_at: the ledger's earliest event timestamp, `None` for an empty ledger file.
    newest_at: the ledger's latest event timestamp, `None` for an empty ledger file.
    """

    study_id: str
    name: str | None = None
    counts: dict[str, int]
    oldest_at: str | None = None
    newest_at: str | None = None


def study_runs(cache: Cache, study_id: str, *, limit: int = 100_000) -> list[RunRecord]:
    """Every dispatch run recorded for this study, newest first.

    Reads `cache` through its own public `recent` surface, the only read path `Cache` exposes,
    so this join never touches its SQLite file directly. Keeps the rows whose `name` carries
    this study's dispatch label, either the bare `study:<study_id>` `Fleet.submit_all` stamps on
    every trial or a `study:<study_id>/...` suffix a future caller might append to distinguish
    trials within one study by name.

    cache: the dispatch run registry to read.
    study_id: the study whose label a `RunRecord.name` is matched against.
    limit: how many of the newest dispatch rows `recent` scans; raise it past the default for a
        study that outlives that window.
    """
    return [run for run in cache.recent(limit) if labelled_study(run.name) == study_id]


def _merge_states(cache: Cache, ledger: StudyLedger, study_id: str) -> dict[str, str]:
    """Each handle's resolved state: the ledger's own fold, dispatch's terminal verdict winning.

    Starts from `ledger.statuses`, each dispatched handle already `submitted` or resolved to
    whatever verdict word the ledger itself recorded. `study_runs`'s dispatch rows then override
    a handle with their own `verdict` whenever dispatch has resolved one, even for a handle the
    ledger never got a `verdict` event for, since a durable dispatch monitor can terminalize a
    run without ever writing back to this study's ledger. A handle dispatch knows about but the
    ledger never recorded (a crash between the two writes) still counts, as `submitted`, since a
    recorded `RunRecord` is itself proof the trial was dispatched. Dispatch wins for terminal
    states; the ledger's own fold (or the bare `submitted` placeholder) wins whenever dispatch
    has nothing resolved yet.
    """
    states = dict(ledger.statuses())
    for run in study_runs(cache, study_id):
        if run.verdict is not None:
            states[run.handle] = run.verdict
        else:
            states.setdefault(run.handle, "submitted")
    return states


def study_progress(cache: Cache, ledger: StudyLedger, study: Study) -> Progress:
    """`study`'s live trial counts, dispatch's resolved verdicts merged over the ledger's fold.

    cache: the dispatch run registry `study_runs` joins against.
    ledger: `study`'s own event ledger.
    study: the study these counts belong to.
    """
    return Progress.fold(_merge_states(cache, ledger, study.study_id))


def overview(cache: Cache, ledgers_root: Path) -> list[StudySummary]:
    """Every study's summary, one per `<study_id>.jsonl` file found under `ledgers_root`.

    Each summary's `counts` are joined against `cache` exactly as `study_progress` resolves a
    single study, generalized from that call's four fixed buckets to a count per distinct state
    word actually seen. A `ledgers_root` that does not exist yet reads as no studies at all.

    cache: the dispatch run registry each summary's counts are joined against.
    ledgers_root: the directory holding every study's `.jsonl` ledger file.
    """
    return [_summarize(cache, path) for path in sorted(ledgers_root.glob("*.jsonl"))]


def _summarize(cache: Cache, path: Path) -> StudySummary:
    """One ledger file's `StudySummary`, its study id read off `path`'s own basename."""
    study_id = path.stem
    ledger = StudyLedger.at(path)
    events = ledger.events()
    timestamps = [event.at for event in events]
    name = next((event.name for event in events if event.kind == "created"), None)
    states = _merge_states(cache, ledger, study_id)
    return StudySummary(
        study_id=study_id,
        name=name,
        counts=dict(Counter(states.values())),
        oldest_at=min(timestamps, default=None),
        newest_at=max(timestamps, default=None),
    )
