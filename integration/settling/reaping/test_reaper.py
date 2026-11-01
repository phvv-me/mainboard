"""The reap pass: which listed machines are provably ours and orphaned, and what the pass does."""

from datetime import UTC, datetime, timedelta

import pytest
from filelock import FileLock
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.constants import REAP_DELETES
from mainboard.dispatch.backends import Rented
from mainboard.dispatch.state import RunRecord
from mainboard.serve.constants import SERVICES
from mainboard.settling.reaping.constants import GRACE
from mainboard.settling.reaping.reaper import Action, Kept, Reaper, Stage, decide

from ...dispatch.backends.builders import refusal
from ...dispatch.backends.fakes import Recorded
from .builders import kept, record, rented
from .constants import NOW, PROVIDER

_LATE = timedelta(minutes=11)
_SOON = timedelta(minutes=1)
_HANDLES = ["m-1", "m-2", "m-3", "mainboard-a"]
_LABELS = ["mainboard-a", "mainboard-b", "mainboard-c"]
_INSTANTS = st.datetimes(
    min_value=datetime(2026, 10, 28), max_value=datetime(2026, 10, 30), timezones=st.just(UTC)
)
_RENTED = st.builds(
    Rented,
    handle=st.sampled_from(_HANDLES),
    label=st.sampled_from(["", "someone-else", *_LABELS]),
    status=st.sampled_from(["running", "pending", "gone"]),
)
_KEPT = st.builds(
    Kept,
    label=st.sampled_from(_LABELS),
    target=st.just(PROVIDER),
    provider=st.sampled_from([PROVIDER, "vast"]),
    handle=st.sampled_from(_HANDLES),
    stage=st.sampled_from(list(Stage)),
    submitted=_INSTANTS,
    release_by=st.none() | _INSTANTS,
)
_HELD = st.dictionaries(st.sampled_from(_HANDLES), st.just("a-session"))


def _decided(listing: list[Rented], known: list[Kept], held: dict[str, str]) -> list:
    return decide(listing, provider=PROVIDER, kept=known, held=held, now=NOW)


@pytest.mark.parametrize(
    ("found", "known", "held", "action"),
    [
        pytest.param(rented(), [kept()], {}, Action.END, id="ours-and-its-run-was-reported"),
        pytest.param(rented(), [kept()], {"m-1": "box"}, Action.KEEP, id="ours-but-held-as-a-box"),
        pytest.param(
            rented(status="gone"), [kept()], {}, Action.KEEP, id="ours-and-already-ended"
        ),
        pytest.param(
            rented(), [kept(stage=Stage.LIVE)], {}, Action.KEEP, id="ours-and-its-run-is-live"
        ),
        pytest.param(
            rented(), [kept(stage=Stage.OWED)], {}, Action.KEEP, id="ours-and-its-report-is-owed"
        ),
        pytest.param(
            rented(),
            [kept(stage=Stage.LIVE, release_by=NOW - _LATE)],
            {},
            Action.END,
            id="ours-live-and-past-its-deadline-by-more-than-the-grace",
        ),
        pytest.param(
            rented(),
            [kept(stage=Stage.LIVE, release_by=NOW - _SOON)],
            {},
            Action.KEEP,
            id="ours-live-and-just-past-its-deadline",
        ),
        pytest.param(
            rented(),
            [kept(stage=Stage.SUBMITTING, handle="mainboard-a", age=_LATE)],
            {},
            Action.BIND,
            id="submitting-past-the-grace",
        ),
        pytest.param(
            rented(),
            [kept(stage=Stage.SUBMITTING, handle="mainboard-a", age=_SOON)],
            {},
            Action.KEEP,
            id="submitting-and-its-create-may-be-in-flight",
        ),
        pytest.param(
            rented(), [kept(stage=Stage.PREPARED)], {}, Action.REPORT, id="prepared-never-sent"
        ),
        pytest.param(rented(label=""), [kept()], {}, Action.REPORT, id="label-missing"),
        pytest.param(rented(label="by-hand"), [kept()], {}, Action.REPORT, id="label-not-minted"),
        pytest.param(
            rented(),
            [kept().model_copy(update={"provider": "vast"})],
            {},
            Action.REPORT,
            id="minted-for-another-provider",
        ),
        pytest.param(
            rented(handle="m-9"), [kept()], {}, Action.REPORT, id="label-recorded-against-another"
        ),
    ],
)
def test_a_listed_machine_is_ended_or_bound_only_when_its_label_and_record_prove_it_ours(
    found: Rented, known: list[Kept], held: dict[str, str], action: Action
) -> None:
    assert [each.action for each in _decided([found], known, held)] == [action]


@given(listing=st.lists(_RENTED, max_size=8), known=st.lists(_KEPT, max_size=6), held=_HELD)
@example(listing=[Rented(handle="m-1", label="")], known=[], held={})
@example(listing=[Rented(handle="m-1", label="someone-else")], known=[], held={})
def test_the_pass_never_ends_or_binds_a_machine_without_a_label_this_center_minted(
    listing: list[Rented], known: list[Kept], held: dict[str, str]
) -> None:
    chosen = [each for each in _decided(listing, known, held) if each.selected]
    minted = {each.label for each in known if each.provider == PROVIDER}
    assert all(each.rented.label in minted for each in chosen)
    assert all(each.rented.handle not in held and each.rented.status != "gone" for each in chosen)


@given(listing=st.lists(_RENTED, max_size=8), known=st.lists(_KEPT, max_size=6), held=_HELD)
def test_a_machine_is_ended_only_when_finished_or_overdue_and_bound_only_when_submitting(
    listing: list[Rented], known: list[Kept], held: dict[str, str]
) -> None:
    chosen = [each for each in _decided(listing, known, held) if each.selected]
    ended = [each.kept for each in chosen if each.action is Action.END and each.kept]
    bound = [each.kept for each in chosen if each.action is Action.BIND and each.kept]
    assert all(each.stage is Stage.DONE or each.expired(NOW) for each in ended)
    assert all(each.stage is Stage.SUBMITTING and NOW - each.submitted >= GRACE for each in bound)
    assert len(ended) + len(bound) == len(chosen)


def test_two_machines_under_one_label_are_both_reported_and_neither_is_bound() -> None:
    twins = [rented("m-1"), rented("m-2")]
    crashed = kept(stage=Stage.SUBMITTING, handle="mainboard-a", age=_LATE)
    decided = _decided([*twins, rented("m-3", label="mainboard-b")], [crashed], {})
    assert [(each.rented.handle, each.action) for each in decided] == [
        ("m-1", Action.REPORT),
        ("m-2", Action.REPORT),
        ("m-3", Action.REPORT),
    ]
    assert "listed 2 times" in decided[0].reason


@given(listing=st.lists(_RENTED, max_size=8), known=st.lists(_KEPT, max_size=6), held=_HELD)
def test_a_label_listed_twice_selects_none_of_its_machines(
    listing: list[Rented], known: list[Kept], held: dict[str, str]
) -> None:
    labels = [each.label for each in listing if each.label]
    chosen = [each for each in _decided(listing, known, held) if each.selected]
    assert all(labels.count(each.rented.label) == 1 for each in chosen)


@pytest.mark.parametrize(
    ("verdict", "reported", "stage"),
    [
        pytest.param("prepared", None, Stage.PREPARED, id="never-sent"),
        pytest.param("submitting", None, Stage.SUBMITTING, id="sent-with-no-handle"),
        pytest.param("running", None, Stage.LIVE, id="running"),
        pytest.param("held", None, Stage.LIVE, id="held-by-a-quota-and-not-settled"),
        pytest.param("ok", None, Stage.OWED, id="settled-and-not-reported"),
        pytest.param("failed", "ok", Stage.OWED, id="settled-and-reported-as-something-else"),
        pytest.param("timeout", "timeout", Stage.DONE, id="settled-and-reported"),
    ],
)
def test_a_run_record_becomes_the_creation_stage_its_verdict_and_report_say(
    verdict: str, reported: str | None, stage: Stage
) -> None:
    creation = Kept.of(record("mainboard-a", handle="m-1", verdict=verdict, reported=reported))
    assert creation is not None and creation.stage is stage


def test_a_run_no_provider_created_has_no_creation() -> None:
    run = RunRecord(handle="7", target="gold", kind="ssh", script="x", args="", submitted_at="")
    assert Kept.of(run) is None


def test_a_dry_run_lists_and_logs_but_ends_nothing_and_binds_nothing(
    reaper: Reaper, wire: Recorded
) -> None:
    reaped = reaper.once(delete=False)
    crash = reaper.cache.creations.named("mainboard-crash", target=PROVIDER)
    assert [(method, url.rsplit("/", 1)[-1]) for method, url, _ in wire.sent] == [
        ("GET", "instances")
    ]
    assert (reaped.acted, reaped.failed, crash.handle) == ([], {}, "mainboard-crash")
    assert [each.rented.handle for each in reaped.decisions if each.selected] == [
        "m-done",
        "m-crash",
    ]


def test_a_deleting_pass_ends_the_reported_machine_and_binds_the_crashed_create_alone(
    reaper: Reaper, wire: Recorded
) -> None:
    reaped = reaper.once(delete=True)
    crash = reaper.cache.creations.named("mainboard-crash", target=PROVIDER)
    deletes = [url for method, url, _ in wire.sent if method == "POST"]
    assert reaped.acted == ["m-done", "m-crash"] and reaped.failed == {}
    assert [url.rsplit("/", 2)[-2] for url in deletes] == ["m-done"] and crash.handle == "m-crash"


def test_a_machine_the_labels_cannot_prove_ours_is_reported_and_never_touched(
    reaper: Reaper, wire: Recorded
) -> None:
    reaped = reaper.once(delete=True)
    left = {each.rented.handle: each.action for each in reaped.decisions}
    assert {left[handle] for handle in ("m-foreign", "m-hand", "m-blank")} == {Action.REPORT}
    assert {left[handle] for handle in ("m-owed", "m-live", "m-held")} == {Action.KEEP}
    assert all("m-foreign" not in url and "m-held" not in url for _, url, _ in wire.sent)


def test_a_provider_that_cannot_list_is_left_alone_and_nothing_is_ended(
    reaper: Reaper, wire: Recorded
) -> None:
    wire.answers[("GET", "/instances")] = refusal(503, "unavailable")
    reaped = reaper.once(delete=True)
    assert [method for method, _, _ in wire.sent] == ["GET"] and reaped.decisions == []
    assert set(reaped.unreached) == {PROVIDER}


def test_a_refused_delete_is_recorded_and_the_other_machines_are_still_bound(
    reaper: Reaper, wire: Recorded
) -> None:
    wire.answers[("POST", "/instances/m-done/delete")] = refusal(500, "boom")
    reaped = reaper.once(delete=True)
    assert set(reaped.failed) == {"m-done"} and reaped.acted == ["m-crash"]


def test_a_pass_that_finds_settlement_busy_does_nothing_at_all(
    reaper: Reaper, wire: Recorded
) -> None:
    with FileLock(reaper.cache.settlement.lock_file):
        reaped = reaper.once(delete=True)
    assert reaped.busy and reaped.decisions == [] and wire.sent == []


def test_the_reap_service_is_declared_off_and_deletes_nothing_until_the_owner_says_so() -> None:
    declared = next(service for service in SERVICES if service.name == "reap")
    assert (REAP_DELETES, declared.off, declared.every) == (False, "waits for the canaries", "10m")
