"""Failure boundaries around provider creation, without allocating a machine."""

from pathlib import Path

import pytest

from mainboard import Board, MissionError
from mainboard.dispatch.allocation import Allocation
from mainboard.dispatch.state import Cache
from mainboard.listing import Listing

from .support import created_request


def test_a_lost_create_response_remains_durable_across_processes(tmp_path: Path) -> None:
    original = created_request().record
    first = Cache(tmp_path / "dispatch.sqlite")
    first.reserve(original)
    allocation = Allocation(cache=first, record=original)
    allocation.begin()
    allocation.interrupted()
    fresh = Cache(first.path)
    assert fresh.run(original.handle).verdict == "submitting"
    assert fresh.tracked() == [fresh.run(original.handle)]
    with pytest.raises(ValueError, match="unresolved creation"):
        fresh.reserve(
            original.model_copy(
                update={"handle": "second", "creation": "second", "name": "new-auto-label"}
            )
        )


def test_binding_keeps_one_row_its_original_time_and_the_provider_label() -> None:
    allocation = created_request()
    allocation.begin()
    allocation.created("provider-42")
    record = allocation.cache.run("provider-42")
    assert allocation.cache.total() == 1
    assert record.creation == allocation.label
    assert record.submitted_at == allocation.record.submitted_at
    assert record.verdict == "queued"
    assert allocation.cache.creation(allocation.label, record.target) == record


@pytest.mark.parametrize("cancelled", [True, False], ids=["cancelled", "entered-twice"])
def test_only_a_prepared_request_crosses_the_create_boundary(cancelled: bool) -> None:
    allocation = created_request()
    if cancelled:
        allocation.cache.resolve(allocation.record, "cancelled", None, "cancelled")
    else:
        allocation.begin()
    with pytest.raises(ValueError, match="no longer prepared"):
        allocation.begin()


def test_cancellation_cannot_erase_a_create_that_just_started() -> None:
    allocation = created_request()
    stale = allocation.record
    allocation.begin()
    with pytest.raises(ValueError, match="no longer prepared"):
        allocation.cache.leave_prepared(stale, "cancelled")
    assert allocation.cache.run(stale.handle).verdict == "submitting"


def test_interrupted_landing_does_not_relabel_an_already_cancelled_request() -> None:
    allocation = created_request()
    allocation.cache.delivery(allocation.record, "not_started")
    cancelled = allocation.cache.leave_prepared(allocation.record, "cancelled")
    allocation.interrupted()
    assert allocation.cache.run(cancelled.handle).verdict == "cancelled"


@pytest.mark.parametrize("verdict", ["cancelled", "failed"])
def test_a_never_created_terminal_request_needs_no_second_write(
    board: Board, monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    record = created_request().record
    board.dispatcher.cache.reserve(record)
    board.dispatcher.cache.leave_prepared(record, verdict)
    fresh = Cache(board.dispatcher.cache.path)
    assert fresh.run(record.handle).reported == verdict
    assert fresh.tracked() == []
    monkeypatch.setattr(board, "job", lambda *args, **kwargs: pytest.fail("provider touched"))
    board.monitor().once()


def test_unknown_creation_is_visible_but_never_polled_cancelled_or_retried(
    board: Board,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = created_request().record
    board.dispatcher.cache.reserve(original)
    allocation = Allocation(cache=board.dispatcher.cache, record=original)
    allocation.begin()
    monkeypatch.setattr(board, "job", lambda *args, **kwargs: pytest.fail("provider touched"))
    monitor = board.monitor().once()
    assert len(monitor.failed) == 1
    assert "no automatic resubmission" in monitor.failed[0].reason
    assert Listing(board, limit=10).taken().rows[0].state == "submitting"
    assert board.verdicts().handled(allocation.label).code == 2
    with pytest.raises(MissionError, match="reconcile"):
        board.verdicts().cancel(allocation.label)
    assert allocation.cache.run(allocation.label).verdict == "submitting"


def test_setup_failure_after_handle_recovery_stays_tracked_for_release() -> None:
    allocation = created_request()
    allocation.cache.delivery(allocation.record, "not_started")
    allocation.begin()
    allocation.created("provider-42")
    allocation.interrupted()
    record = allocation.cache.run("provider-42")
    assert record.verdict == "failed" and record.reported is None
    assert allocation.cache.tracked() == [record]


def test_a_precreate_refusal_does_not_leave_an_unknown_paid_instance() -> None:
    allocation = created_request()
    allocation.interrupted()
    record = allocation.cache.run(allocation.label)
    assert record.verdict == "failed" and record.reported == "failed"
    assert allocation.cache.tracked() == []


@pytest.mark.parametrize("handle", ["", "None"])
def test_a_create_answered_without_a_handle_stays_submitting_for_a_reconciliation(
    handle: str,
) -> None:
    """An answer naming no instance does not prove none was made, so only the label can tell."""
    allocation = created_request()
    allocation.begin()
    with pytest.raises(MissionError, match="reconcile its label"):
        allocation.created(handle)
    assert allocation.cache.run(allocation.label).verdict == "submitting"


def test_a_request_that_already_holds_its_handle_is_left_to_the_monitor() -> None:
    """A late refusal or interruption cannot relabel a creation the provider already confirmed."""
    allocation = created_request()
    allocation.begin()
    allocation.created("provider-42")
    allocation.refused()
    allocation.interrupted()
    record = allocation.cache.run("provider-42")
    assert (record.verdict, record.reported) == ("queued", None)


def test_each_creation_transition_refuses_a_request_not_standing_where_it_expects() -> None:
    """Every step names the one state it moves from, so a stale caller is told, never obeyed."""
    allocation = created_request()
    store, record = allocation.cache, allocation.record
    with pytest.raises(ValueError, match="invalid prepared transition to 'queued'"):
        store.leave_prepared(record, "queued")
    with pytest.raises(ValueError, match="is not submitting; nothing to reopen"):
        store.reopen(record)
    with pytest.raises(LookupError, match="no submitting creation"):
        store.bind(record, "provider-42")
    with pytest.raises(LookupError, match="no creation 'elsewhere'"):
        store.creation("elsewhere", record.target)
    assert store.run(record.handle).verdict == "prepared"
