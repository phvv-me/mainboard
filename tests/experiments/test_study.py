from typing import TYPE_CHECKING

from mainboard.experiments import Progress, Study, StudyEvent, StudyLedger

if TYPE_CHECKING:
    from pathlib import Path


def test_creating_a_study_derives_a_stable_identity_and_a_slug_from_its_experiment() -> None:
    derived = Study.create("joint-search", config_space={"bits": [1, 2]}, git_sha="abc123")
    assert derived.experiment == "joint-search"
    assert len(derived.study_id) == 12
    assert derived.name == f"joint-search-{derived.study_id[:6]}"
    assert (derived.hosts, derived.models, derived.dirty) == ((), (), False)
    assert "T" in derived.created_at
    twin = Study.create("joint-search", config_space={"bits": [1, 2]}, git_sha="abc123")
    assert twin.study_id == derived.study_id
    declared = Study.create(
        "e",
        config_space={},
        git_sha="s",
        name="my-run",
        hosts=("gold", "miyabi-g"),
        models=("m1",),
        dirty=True,
    )
    assert (declared.name, declared.hosts, declared.models, declared.dirty) == (
        "my-run",
        ("gold", "miyabi-g"),
        ("m1",),
        True,
    )


def test_a_study_ledger_lives_under_the_generated_studies_dir_and_reopens_by_path(
    tmp_path: Path,
) -> None:
    ledger = StudyLedger(tmp_path, "abc123def456")
    assert ledger.path == tmp_path / ".mainboard" / "studies" / "abc123def456.jsonl"
    ledger.submitted("H1", host="gold")
    assert len(StudyLedger(tmp_path, "abc123def456").events()) == 1
    assert len(StudyLedger.at(ledger.path).events()) == 1


def test_a_ledger_folds_each_handles_latest_event_into_its_status_and_progress(
    tmp_path: Path, study: Study
) -> None:
    ledger = StudyLedger(tmp_path, study.study_id)
    assert (ledger.events(), ledger.statuses(), ledger.progress()) == ([], {}, Progress())
    ledger.created(study)
    for handle in ("H1", "H2", "H3", "H4"):
        ledger.submitted(handle, host="gold")
    ledger.verdict("H1", state="ok")
    ledger.verdict("H2", state="failed")
    ledger.verdict("H3", state="vanished")
    ledger.append(StudyEvent(at="t0", kind="other", handle="H5"))

    events = ledger.events()
    assert [event.kind for event in events[:2]] == ["created", "submitted"]
    assert events[0].name == study.name
    assert (events[1].handle, events[1].host) == ("H1", "gold")
    assert ledger.statuses() == {"H1": "ok", "H2": "failed", "H3": "vanished", "H4": "submitted"}
    assert ledger.progress() == Progress(submitted=4, running=1, ok=1, failed=2)


def test_a_ledger_ignores_a_blank_line_left_in_its_append_only_file(tmp_path: Path) -> None:
    """A stray newline in the NDJSON must not read back as an empty event."""
    ledger = StudyLedger(tmp_path, "sid")
    ledger.submitted("H1", host="gold")
    with ledger.path.open("a", encoding="utf-8") as opened:
        opened.write("\n")
    assert len(ledger.events()) == 1
