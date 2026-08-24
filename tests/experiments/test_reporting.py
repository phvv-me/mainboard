from typing import TYPE_CHECKING

import pytest

from mainboard.experiments import Progress, Study, StudyLedger
from mainboard.experiments.identity import study_label
from mainboard.experiments.reporting import StudySummary, overview, study_progress, study_runs

from .conftest import make_run

if TYPE_CHECKING:
    from pathlib import Path

    from mainboard.dispatch.state import Cache


def test_study_runs_keeps_the_bare_and_slash_suffixed_labels_newest_first(cache: Cache) -> None:
    """`study:<id>` and `study:<id>/<trial>` both name the same study, and a run labelled for
    another study or for nothing at all is not this study's."""
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
    recorded: tuple[str, ...],
    dispatched: bool,
    verdict: str | None,
    progress: Progress,
) -> None:
    ledger = StudyLedger(tmp_path, study.study_id)
    for state in recorded:
        if state == "submitted":
            ledger.submitted("H1", host="gold")
        else:
            ledger.verdict("H1", state=state)
    if dispatched:
        cache.record(make_run(study_label(study.study_id), handle="H1", verdict=verdict))
    assert study_progress(cache, ledger, study) == progress


def test_overview_reads_an_absent_root_or_an_empty_ledger_file_as_nothing_to_summarize(
    cache: Cache, tmp_path: Path
) -> None:
    studies = tmp_path / "studies"
    assert overview(cache, studies) == []
    studies.mkdir()
    (studies / "sid.jsonl").touch()
    assert overview(cache, studies) == [StudySummary(study_id="sid", counts={})]


def test_overview_summarizes_every_ledger_file_with_its_name_counts_and_timestamp_span(
    cache: Cache, tmp_path: Path
) -> None:
    studies = tmp_path / "studies"
    named = Study.create("e", config_space={"x": 1}, git_sha="s", name="alpha")
    anonymous = Study.create("e", config_space={"x": 2}, git_sha="s", name="beta")
    ledger = StudyLedger.at(studies / f"{named.study_id}.jsonl")
    ledger.created(named)
    for handle in ("H1", "H2"):
        ledger.submitted(handle, host="gold")
    cache.record(make_run(study_label(named.study_id), handle="H1", verdict="ok"))
    bare = StudyLedger.at(studies / f"{anonymous.study_id}.jsonl")
    bare.submitted("H3", host="gold")
    bare.verdict("H3", state="vanished")

    summaries = {summary.study_id: summary for summary in overview(cache, studies)}
    assert [summary.study_id for summary in overview(cache, studies)] == sorted(summaries)
    assert summaries[named.study_id].name == "alpha"
    assert summaries[named.study_id].counts == {"ok": 1, "submitted": 1}
    oldest, newest = summaries[named.study_id].oldest_at, summaries[named.study_id].newest_at
    assert oldest is not None and newest is not None and oldest <= newest
    assert summaries[anonymous.study_id].name is None
    assert summaries[anonymous.study_id].counts == {"vanished": 1}
