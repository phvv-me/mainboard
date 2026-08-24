import gc
import sqlite3

import pytest

from mainboard.dispatch import HostSetup, now

from ..conftest import cache, run_record


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
