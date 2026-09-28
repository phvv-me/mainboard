from mainboard.dispatch.state import Cache
from mainboard.experiments import Progress, Study, StudyEvent, StudyLedger


def test_creating_a_study_derives_a_stable_identity_and_a_slug_from_its_experiment() -> None:
    derived = Study.create("joint-search", config_space={"bits": [1, 2]}, source_digest="abc123")
    assert derived.experiment == "joint-search"
    assert len(derived.study_id) == 12
    assert derived.name == f"joint-search-{derived.study_id[:6]}"
    assert (derived.hosts, derived.models) == ((), ())
    assert "T" in derived.created_at
    twin = Study.create("joint-search", config_space={"bits": [1, 2]}, source_digest="abc123")
    assert twin.study_id == derived.study_id
    declared = Study.create(
        "e", config_space={}, source_digest="s", name="my-run", hosts=("gold",), models=("m1",)
    )
    assert (declared.name, declared.hosts, declared.models) == ("my-run", ("gold",), ("m1",))


def test_a_study_ledger_reopens_by_its_id_and_holds_only_its_own_study() -> None:
    session = Cache.private().session
    StudyLedger(session, "abc123def456").submitted("H1", host="gold")
    assert len(StudyLedger(session, "abc123def456").events()) == 1
    assert StudyLedger(session, "another").events() == []


def test_a_ledger_folds_each_handles_latest_event_into_its_status_and_progress(
    study: Study,
) -> None:
    ledger = StudyLedger(Cache.private().session, study.study_id)
    assert (ledger.events(), ledger.statuses(), ledger.progress()) == ([], {}, Progress())
    ledger.created(study)
    for handle in ("H1", "H2", "H3", "H4"):
        ledger.submitted(handle, host="gold")
    ledger.verdict("H1", state="ok")
    ledger.verdict("H2", state="failed")
    ledger.verdict("H3", state="vanished")
    ledger.append(StudyEvent(at="2026-01-01T00:00:00+00:00", kind="other", handle="H5"))

    events = ledger.events()
    assert [event.kind for event in events[:2]] == ["created", "submitted"]
    assert events[0].name == study.name
    assert (events[1].handle, events[1].host) == ("H1", "gold")
    assert ledger.statuses() == {"H1": "ok", "H2": "failed", "H3": "vanished", "H4": "submitted"}
    assert ledger.progress() == Progress(submitted=4, running=1, ok=1, failed=2)
