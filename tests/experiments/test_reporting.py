from typing import TYPE_CHECKING

from mainboard.dispatch.state import Cache, RunRecord
from mainboard.experiments import Progress, Study, StudyLedger
from mainboard.experiments.reporting import StudySummary, overview, study_progress, study_runs

if TYPE_CHECKING:
    from pathlib import Path


def make_run(
    name: str, *, handle: str, submitted_at: str = "t0", verdict: str | None = None
) -> RunRecord:
    """A dispatch `RunRecord` labeled `name`, resolved to `verdict` when given."""
    return RunRecord(
        handle=handle,
        target="gold",
        kind="pbs",
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at=submitted_at,
        name=name,
        verdict=verdict,
    )


def test_study_runs_keeps_the_bare_and_slash_suffixed_labels(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("study:sid", handle="H1"))
    cache.record(make_run("study:sid/trial-a", handle="H2"))
    cache.record(make_run("study:other", handle="H3"))
    cache.record(make_run("", handle="H4"))
    runs = study_runs(cache, "sid")
    assert {run.handle for run in runs} == {"H1", "H2"}


def test_study_runs_orders_newest_first_and_respects_limit(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("study:sid", handle="H1", submitted_at="2024-01-01T00:00:00"))
    cache.record(make_run("study:sid", handle="H2", submitted_at="2024-01-02T00:00:00"))
    assert [run.handle for run in study_runs(cache, "sid")] == ["H2", "H1"]
    assert [run.handle for run in study_runs(cache, "sid", limit=1)] == ["H2"]


def test_study_progress_defaults_an_unresolved_handle_to_submitted(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    study = Study.create("e", config_space={}, git_sha="s")
    ledger = StudyLedger(tmp_path, study.study_id)
    ledger.submitted("H1", host="gold")
    assert study_progress(cache, ledger, study) == Progress(submitted=1, running=1)


def test_study_progress_lets_a_resolved_dispatch_verdict_outrank_the_ledger(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    study = Study.create("e", config_space={}, git_sha="s")
    ledger = StudyLedger(tmp_path, study.study_id)
    ledger.submitted("H1", host="gold")
    cache.record(make_run(f"study:{study.study_id}", handle="H1", verdict="failed"))
    assert study_progress(cache, ledger, study) == Progress(submitted=1, running=0, failed=1)


def test_study_progress_keeps_the_ledgers_own_verdict_when_dispatch_has_none_yet(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    study = Study.create("e", config_space={}, git_sha="s")
    ledger = StudyLedger(tmp_path, study.study_id)
    ledger.submitted("H1", host="gold")
    ledger.verdict("H1", state="ok")
    cache.record(make_run(f"study:{study.study_id}", handle="H1"))
    assert study_progress(cache, ledger, study) == Progress(submitted=1, ok=1)


def test_study_progress_counts_a_handle_dispatch_knows_but_the_ledger_never_recorded(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    study = Study.create("e", config_space={}, git_sha="s")
    ledger = StudyLedger(tmp_path, study.study_id)
    cache.record(make_run(f"study:{study.study_id}", handle="H1"))
    assert study_progress(cache, ledger, study) == Progress(submitted=1, running=1)


def test_overview_is_empty_for_a_ledgers_root_that_does_not_exist(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    assert overview(cache, tmp_path / "studies") == []


def test_overview_skips_an_empty_ledger_file_gracefully(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    studies = tmp_path / "studies"
    studies.mkdir()
    (studies / "sid.jsonl").touch()
    [summary] = overview(cache, studies)
    assert summary == StudySummary(study_id="sid", counts={})


def _two_ledgered_studies(cache: Cache, studies: Path) -> tuple[Study, Study]:
    """Two studies on disk: `first` created and fully resolved, `second` bare and vanished."""
    first = Study.create("e", config_space={"x": 1}, git_sha="s", name="alpha")
    second = Study.create("e", config_space={"x": 2}, git_sha="s", name="beta")
    first_ledger = StudyLedger.at(studies / f"{first.study_id}.jsonl")
    first_ledger.created(first)
    for handle in ("H1", "H2"):
        first_ledger.submitted(handle, host="gold")
    cache.record(make_run(f"study:{first.study_id}", handle="H1", verdict="ok"))
    second_ledger = StudyLedger.at(studies / f"{second.study_id}.jsonl")
    second_ledger.submitted("H3", host="gold")
    second_ledger.verdict("H3", state="vanished")
    return first, second


def test_overview_summarizes_every_ledger_file_joined_against_dispatch(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    studies = tmp_path / "studies"
    first, second = _two_ledgered_studies(cache, studies)
    summaries = overview(cache, studies)
    assert [summary.study_id for summary in summaries] == sorted([first.study_id, second.study_id])


def test_overview_summary_carries_a_studys_name_counts_and_timestamp_span(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    studies = tmp_path / "studies"
    first, _second = _two_ledgered_studies(cache, studies)
    by_id = {summary.study_id: summary for summary in overview(cache, studies)}
    assert by_id[first.study_id].name == "alpha"
    assert by_id[first.study_id].counts == {"ok": 1, "submitted": 1}
    assert by_id[first.study_id].oldest_at is not None
    assert by_id[first.study_id].newest_at is not None
    assert by_id[first.study_id].oldest_at <= by_id[first.study_id].newest_at


def test_overview_summary_has_no_name_when_the_ledger_never_recorded_a_created_event(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    studies = tmp_path / "studies"
    _first, second = _two_ledgered_studies(cache, studies)
    by_id = {summary.study_id: summary for summary in overview(cache, studies)}
    assert by_id[second.study_id].name is None
    assert by_id[second.study_id].counts == {"vanished": 1}
