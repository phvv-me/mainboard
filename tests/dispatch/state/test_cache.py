import gc
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.costs.catalog import Offer
from mainboard.dispatch import HostSetup, now
from mainboard.dispatch.lease import Lease
from mainboard.dispatch.state import Cache

from ..support import cache, run_record


def test_in_memory_state_cannot_create_a_phony_durable_settlement_lock() -> None:
    with pytest.raises(MissionError, match="file-backed"), cache().settlement:
        pytest.fail("an in-memory registry cannot coordinate durable settlement")


def test_unlimited_history_retains_old_output_declarations() -> None:
    store = cache()
    for index in range(25):
        store.record(
            run_record(str(index), submitted_at=f"{index:02}").model_copy(
                update={"fetch_path": f"measurements/{index}", "verdict": "ok"}
            )
        )
    assert len(store.recent()) == 20
    assert {run.fetch_path for run in store.recent(limit=None)} == {
        f"measurements/{index}" for index in range(25)
    }


def test_the_settlement_lock_follows_the_opened_database_not_a_later_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = Cache(Path("dispatch.sqlite"))
    monkeypatch.chdir(tmp_path.parent)
    assert Path(store.settlement.lock_file) == tmp_path / "dispatch.settlement.lock"


def test_a_held_dispatch_can_reenter_its_own_settlement_lock(tmp_path: Path) -> None:
    original = Cache(tmp_path / "dispatch.sqlite")
    reopened = Cache(original.path)
    with original.settlement, reopened.settlement.acquire(timeout=0):
        assert reopened.settlement.is_locked


def test_a_cache_nobody_holds_any_more_closes_the_database_it_opened() -> None:
    """A short-lived cache is the collector's to close, not a caller's to remember."""
    store = cache()
    connection = store.connection
    del store
    gc.collect()
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_a_run_round_trips_through_the_registry_and_upserts_by_its_identity() -> None:
    """`(target, handle, submitted_at)` is the identity, so a rewrite replaces the row."""
    store = cache()
    store.record(run_record("H1"))
    store.record(run_record("H1").model_copy(update={"name": "renamed"}))
    store.record(run_record("H2", submitted_at="t1"))
    assert [run.handle for run in store.recent(10)] == ["H2", "H1"]
    assert store.run("H1").name == "renamed"
    assert "T" in now()


def test_resolve_memoizes_the_outcome_and_report_builds_on_that_same_write() -> None:
    """A sweep advancing its cursor must not clobber the verdict `resolve` just stored."""
    store = cache()
    run = run_record("H1")
    store.record(run)
    stored = store.resolve(run, "F", 0, "ok")
    assert (stored.verdict, stored.state, stored.exit_code) == ("ok", "F", 0)
    store.report(stored, "ok")
    settled = store.run("H1")
    assert (settled.verdict, settled.exit_code, settled.reported) == ("ok", 0, "ok")


def test_a_stale_running_probe_cannot_erase_a_definite_setup_failure() -> None:
    store = cache()
    run = run_record("H1")
    store.record(run)
    store.resolve(run, "failed", None, "failed")
    store.delivery(run, "not_started")
    refreshed = store.resolve(run, "R", None, "running")
    assert (refreshed.verdict, refreshed.evidence) == ("failed", "not_started")


def test_a_stale_setup_failure_cannot_erase_a_completed_cancellation() -> None:
    store = cache()
    run = run_record("H1")
    store.record(run)
    cancelled = store.resolve(run, "cancelled", None, "cancelled")
    store.report(cancelled, "cancelled")
    refreshed = store.resolve(run, "failed", None, "failed")
    assert (refreshed.verdict, refreshed.state, refreshed.reported) == (
        "cancelled",
        "cancelled",
        "cancelled",
    )
    assert store.tracked() == []


def test_cancel_can_mark_release_without_relabeling_a_terminal_computation() -> None:
    store = cache()
    run = run_record("H1")
    store.record(run)
    complete = store.resolve(run, "F", 0, "ok")
    cancelled = store.resolve(complete, "cancelled", 0, "ok")
    assert (cancelled.verdict, cancelled.state, cancelled.exit_code) == ("ok", "cancelled", 0)


def test_tracked_holds_a_run_until_its_terminal_verdict_has_been_reported() -> None:
    """The job whose dispatching agent died is exactly the one no sweep may ever drop."""
    store = cache()
    older = run_record("H1", submitted_at="t0")
    store.record(older)
    store.record(run_record("H2", submitted_at="t1"))
    assert [run.handle for run in store.tracked()] == ["H2", "H1"]
    running = store.resolve(older, "R", None, "running")
    finished = store.resolve(running, "F", 0, "ok")
    assert "H1" in [run.handle for run in store.tracked()]
    store.report(finished, "ok")
    assert [run.handle for run in store.tracked()] == ["H2"]


def test_independent_updates_preserve_delivery_and_the_exact_run_identity() -> None:
    store = cache()
    older = run_record("H1", submitted_at="t0")
    newer = run_record("H1", submitted_at="t1")
    store.record(older)
    store.record(newer)
    store.delivery(older, "pending")
    store.resolve(older, "F", 0, "ok")
    store.delivery(older, "copied")
    store.report(older, "ok")
    current, previous = store.recent(2)
    assert current == newer
    assert (previous.verdict, previous.reported, previous.evidence) == ("ok", "ok", "copied")
    store.forget(older)
    with pytest.raises(LookupError, match="no registered run"):
        store.delivery(older, "verified")


def test_run_resolves_the_newest_row_and_refuses_a_handle_recorded_on_two_targets() -> None:
    store = cache()
    store.record(run_record("H1", target="gold", submitted_at="t0"))
    store.record(run_record("H1", target="gold", submitted_at="t1"))
    assert store.run("H1").submitted_at == "t1"
    store.record(run_record("H1", target="crimson", submitted_at="t2"))
    assert store.run("H1", target="gold").submitted_at == "t1"
    with pytest.raises(LookupError, match="recorded on crimson, gold"):
        store.run("H1")
    with pytest.raises(LookupError, match="no recorded run 'ghost'"):
        store.run("ghost")
    with pytest.raises(LookupError, match="on 'gold'"):
        store.run("ghost", target="gold")


def test_a_host_is_stamped_when_it_was_onboarded_and_upserts_by_alias() -> None:
    store = cache()
    stamped = store.save_host(HostSetup(host="gold", root="/repo", installer="uv"))
    assert stamped.onboarded_at
    store.save_host(HostSetup(host="gold", root="/elsewhere", installer="pip"))
    store.save_host(HostSetup(host="crimson", root="/repo"))
    assert [record.host for record in store.hosts()] == ["crimson", "gold"]
    assert (store.host("gold").root, store.host("gold").installer) == ("/elsewhere", "pip")
    with pytest.raises(LookupError, match="has never been set up"):
        store.host("ghost")


def test_a_mirror_moves_the_watermark_a_later_transfer_measures_against() -> None:
    """Until a mirror lands the onboarding is the last one, and a host nobody set up has none."""
    store = cache()
    onboarded = store.save_host(HostSetup(host="gold", root="/repo"))
    assert onboarded.mirrored_at == onboarded.onboarded_at
    store.mark_synced("gold")
    mirrored = store.host("gold")
    assert mirrored.synced_at > mirrored.onboarded_at
    assert mirrored.mirrored_at == mirrored.synced_at
    store.mark_synced("ghost")
    assert [record.host for record in store.hosts()] == ["gold"]


def test_a_lease_is_replaced_in_place_and_a_host_forgotten_by_alias() -> None:
    """A hold moves its deadline once the machine is ready, and forgets the host it released."""
    store = cache()
    run = run_record("7")
    store.record(run)
    offer = Offer(provider="vast", gpu="RTX 5090", rate_usd_hr=0.6)
    lease = Lease(offer=offer, release_by=datetime(2026, 9, 25, 18, tzinfo=UTC))
    assert store.relet(run, lease).lease == lease
    with pytest.raises(LookupError, match="no registered run '8'"):
        store.relet(run_record("8"), lease)
    store.save_host(HostSetup(host="box", root="/root/projects"))
    store.drop_host("box")
    store.drop_host("never-set-up")
    assert store.hosts() == []
