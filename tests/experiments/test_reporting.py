from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard.experiments import Progress, Study, StudyLedger
from mainboard.experiments.identity import study_label
from mainboard.experiments.reporting import overview, study_progress, study_runs

from .support import make_run

if TYPE_CHECKING:
    from pathlib import Path

    from mainboard.dispatch.state import Cache


def test_study_runs_keeps_the_bare_and_slash_suffixed_labels_newest_first(cache: Cache) -> None:
    cache.record(make_run("study:sid", handle="H1", submitted_at="2024-01-01T00:00:00"))
    cache.record(make_run("study:other", handle="H2", submitted_at="2024-01-02T00:00:00"))
    cache.record(make_run("", handle="H3", submitted_at="2024-01-03T00:00:00"))
    cache.record(make_run("study:sid/trial-a", handle="H4", submitted_at="2024-01-04T00:00:00"))
    assert [run.handle for run in study_runs(cache, "sid")] == ["H4", "H1"]
    assert [run.handle for run in study_runs(cache, "sid", limit=1)] == ["H4"]


@pytest.mark.parametrize(
    ("recorded", "dispatched", "verdict", "progress"),
    [
        pytest.param(
            ("submitted",), False, None, Progress(submitted=1, running=1), id="unresolved-so-far"
        ),
        pytest.param(
            ("submitted",),
            True,
            "failed",
            Progress(submitted=1, failed=1),
            id="dispatchs-terminal-verdict-outranks-the-ledger",
        ),
        pytest.param(
            ("submitted", "ok"),
            True,
            None,
            Progress(submitted=1, ok=1),
            id="the-ledgers-own-verdict-stands-while-dispatch-has-none",
        ),
        pytest.param(
            (),
            True,
            None,
            Progress(submitted=1, running=1),
            id="a-handle-only-dispatch-ever-recorded-still-counts",
        ),
    ],
)
def test_study_progress_merges_dispatchs_resolved_verdicts_over_the_ledgers_own_fold(
    cache: Cache,
    tmp_path: Path,
    study: Study,
    recorded: Sequence[str],
    dispatched: bool,
    verdict: str | None,
    progress: Progress,
) -> None:
    ledger = StudyLedger(cache.session, study.study_id)
    for state in recorded:
        if state == "submitted":
            ledger.submitted("H1", host="gold")
        else:
            ledger.verdict("H1", state=state)
    if dispatched:
        cache.record(make_run(study_label(study.study_id), handle="H1", verdict=verdict))
    assert study_progress(cache, ledger, study) == progress


def test_overview_reads_a_lake_recording_no_study_as_nothing_to_summarize(cache: Cache) -> None:
    assert overview(cache) == []


def test_overview_summarizes_every_ledger_file_with_its_name_counts_and_timestamp_span(
    cache: Cache,
) -> None:
    named = Study.create("e", config_space={"x": 1}, source_digest="s", name="alpha")
    anonymous = Study.create("e", config_space={"x": 2}, source_digest="s", name="beta")
    ledger = StudyLedger(cache.session, named.study_id)
    ledger.created(named)
    for handle in ("H1", "H2"):
        ledger.submitted(handle, host="gold")
    cache.record(make_run(study_label(named.study_id), handle="H1", verdict="ok"))
    bare = StudyLedger(cache.session, anonymous.study_id)
    bare.submitted("H3", host="gold")
    bare.verdict("H3", state="vanished")

    summaries = {summary.study_id: summary for summary in overview(cache)}
    assert [summary.study_id for summary in overview(cache)] == sorted(summaries)
    assert summaries[named.study_id].name == "alpha"
    assert summaries[named.study_id].counts == {"ok": 1, "submitted": 1}
    oldest, newest = summaries[named.study_id].oldest_at, summaries[named.study_id].newest_at
    assert oldest is not None and newest is not None and oldest <= newest
    assert summaries[anonymous.study_id].name is None
    assert summaries[anonymous.study_id].counts == {"vanished": 1}
