from typing import TYPE_CHECKING

from mainboard.experiments import Progress, Study, StudyEvent, StudyLedger

if TYPE_CHECKING:
    from pathlib import Path


def test_create_derives_the_study_id_and_default_slug() -> None:
    study = Study.create("joint-search", config_space={"bits": [1, 2]}, git_sha="abc123")
    assert study.experiment == "joint-search"
    assert len(study.study_id) == 12
    assert study.name == f"joint-search-{study.study_id[:6]}"
    assert study.dirty is False
    assert study.hosts == ()
    assert study.models == ()
    assert "T" in study.created_at


def test_create_is_stable_for_the_same_inputs() -> None:
    first = Study.create("e", config_space={"x": 1}, git_sha="s")
    second = Study.create("e", config_space={"x": 1}, git_sha="s")
    assert first.study_id == second.study_id


def test_create_honors_an_explicit_name_hosts_models_and_dirty() -> None:
    study = Study.create(
        "e",
        config_space={},
        git_sha="s",
        name="my-run",
        hosts=("gold", "miyabi-g"),
        models=("m1",),
        dirty=True,
    )
    assert study.name == "my-run"
    assert study.hosts == ("gold", "miyabi-g")
    assert study.models == ("m1",)
    assert study.dirty is True


def test_ledger_path_lives_under_dot_mainboard_studies(tmp_path: Path) -> None:
    ledger = StudyLedger(tmp_path, "abc123def456")
    assert ledger.path == tmp_path / ".mainboard" / "studies" / "abc123def456.jsonl"


def test_ledger_at_binds_directly_to_an_already_resolved_path(tmp_path: Path) -> None:
    path = tmp_path / ".mainboard" / "studies" / "sid.jsonl"
    StudyLedger(tmp_path, "sid").submitted("H1", host="gold")
    ledger = StudyLedger.at(path)
    assert ledger.path == path
    assert len(ledger.events()) == 1


def test_ledger_events_statuses_and_progress_are_empty_before_anything_is_appended(
    tmp_path: Path,
) -> None:
    ledger = StudyLedger(tmp_path, "sid")
    assert ledger.events() == []
    assert ledger.statuses() == {}
    assert ledger.progress() == Progress()


def test_ledger_records_created_submitted_and_verdict_events(tmp_path: Path) -> None:
    study = Study.create("e", config_space={}, git_sha="s", name="joint-search")
    ledger = StudyLedger(tmp_path, "sid")
    ledger.created(study)
    ledger.submitted("H1", host="gold")
    ledger.verdict("H1", state="ok")
    events = ledger.events()
    assert [event.kind for event in events] == ["created", "submitted", "verdict"]
    assert events[0].name == "joint-search"
    assert (events[1].handle, events[1].host) == ("H1", "gold")
    assert events[2].state == "ok"


def test_ledger_persists_across_instances(tmp_path: Path) -> None:
    StudyLedger(tmp_path, "sid").submitted("H1", host="gold")
    reopened = StudyLedger(tmp_path, "sid")
    assert len(reopened.events()) == 1


def test_ledger_statuses_folds_to_each_handles_latest_state(tmp_path: Path) -> None:
    study = Study.create("e", config_space={}, git_sha="s")
    ledger = StudyLedger(tmp_path, "sid")
    ledger.created(study)
    ledger.submitted("H1", host="gold")
    ledger.submitted("H2", host="gold")
    ledger.verdict("H1", state="ok")
    assert ledger.statuses() == {"H1": "ok", "H2": "submitted"}


def test_ledger_progress_counts_submitted_running_ok_and_failed(tmp_path: Path) -> None:
    ledger = StudyLedger(tmp_path, "sid")
    for handle in ("H1", "H2", "H3", "H4"):
        ledger.submitted(handle, host="gold")
    ledger.verdict("H1", state="ok")
    ledger.verdict("H2", state="failed")
    ledger.verdict("H3", state="vanished")
    assert ledger.progress() == Progress(submitted=4, running=1, ok=1, failed=2)


def test_ledger_statuses_ignores_an_unrecognized_event_kind(tmp_path: Path) -> None:
    ledger = StudyLedger(tmp_path, "sid")
    ledger.append(StudyEvent(at="t0", kind="other", handle="H1"))
    assert ledger.statuses() == {}


def test_ledger_ignores_blank_lines(tmp_path: Path) -> None:
    ledger = StudyLedger(tmp_path, "sid")
    ledger.submitted("H1", host="gold")
    with ledger.path.open("a") as opened:
        opened.write("\n")
    assert len(ledger.events()) == 1
