from typing import TYPE_CHECKING

import pytest

from mainboard.dispatch import HostSetup
from mainboard.dispatch.shared import now
from mainboard.dispatch.state import Cache, RunRecord

if TYPE_CHECKING:
    from pathlib import Path


def make_run(handle: str, *, target: str = "gold", submitted_at: str = "t0") -> RunRecord:
    return RunRecord(
        handle=handle,
        target=target,
        kind="ssh",
        script="job.sh",
        args="",
        git_sha="abc1234",
        dirty=0,
        submitted_at=submitted_at,
    )


def test_now_returns_an_iso_string() -> None:
    stamp = now()
    assert "T" in stamp


def test_record_and_recent_round_trip(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("H1"))
    [run] = cache.recent(10)
    assert run.handle == "H1"


def test_record_upserts_by_target_handle_submitted_at(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("H1"))
    cache.record(make_run("H1").model_copy(update={"name": "renamed"}))
    [run] = cache.recent(10)
    assert run.name == "renamed"


def test_resolve_memoizes_the_scheduler_outcome(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    run = make_run("H1")
    cache.record(run)
    stored = cache.resolve(run, "F", 0, "ok")
    assert cache.run("H1").verdict == "ok"
    assert cache.run("H1").exit_code == 0
    assert (stored.verdict, stored.state, stored.exit_code) == ("ok", "F", 0)


def test_reporting_the_record_resolve_returned_keeps_both_writes(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    run = make_run("H1")
    cache.record(run)
    cache.report(cache.resolve(run, "F", 0, "ok"), "ok")
    assert (cache.run("H1").verdict, cache.run("H1").reported) == ("ok", "ok")


def test_tracked_holds_a_run_until_its_terminal_verdict_is_reported(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    run = make_run("H1")
    cache.record(run)
    assert [tracked.handle for tracked in cache.tracked()] == ["H1"]
    running = cache.resolve(run, "R", None, "running")
    assert [tracked.handle for tracked in cache.tracked()] == ["H1"]
    finished = cache.resolve(running, "F", 0, "ok")
    assert [tracked.handle for tracked in cache.tracked()] == ["H1"]
    cache.report(finished, "ok")
    assert cache.tracked() == []


def test_tracked_orders_newest_first(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("H1", submitted_at="2024-01-01T00:00:00"))
    cache.record(make_run("H2", submitted_at="2024-01-02T00:00:00"))
    assert [run.handle for run in cache.tracked()] == ["H2", "H1"]


def test_report_records_the_monitor_cursor(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    run = make_run("H1")
    cache.record(run)
    cache.report(run, "ok")
    assert cache.run("H1").reported == "ok"


def test_recent_orders_newest_first(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("H1", submitted_at="2024-01-01T00:00:00"))
    cache.record(make_run("H2", submitted_at="2024-01-02T00:00:00"))
    runs = cache.recent(10)
    assert [run.handle for run in runs] == ["H2", "H1"]


def test_run_narrows_to_a_target_when_given(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("H1", target="gold", submitted_at="t0"))
    cache.record(make_run("H1", target="crimson", submitted_at="t1"))
    assert cache.run("H1", target="gold").target == "gold"


def test_run_without_a_target_returns_the_newest_row(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("H1", target="gold", submitted_at="t0"))
    cache.record(make_run("H1", target="gold", submitted_at="t1"))
    assert cache.run("H1").submitted_at == "t1"


def test_run_raises_for_an_unrecorded_handle(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    with pytest.raises(LookupError, match="no recorded run"):
        cache.run("ghost")


def test_run_raises_for_an_unrecorded_handle_on_a_specific_target(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    with pytest.raises(LookupError, match="on 'gold'"):
        cache.run("ghost", target="gold")


def test_run_is_ambiguous_across_multiple_targets_without_one_given(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.record(make_run("H1", target="gold", submitted_at="t0"))
    cache.record(make_run("H1", target="crimson", submitted_at="t1"))
    with pytest.raises(LookupError, match="recorded on crimson, gold"):
        cache.run("H1")


def test_save_host_stamps_and_upserts_by_alias(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    stamped = cache.save_host(HostSetup(host="gold", root="/repo", installer="uv"))
    assert stamped.onboarded_at
    cache.save_host(HostSetup(host="gold", root="/elsewhere", installer="pip"))
    [record] = cache.hosts()
    assert record.root == "/elsewhere"
    assert record.installer == "pip"


def test_hosts_are_listed_most_recently_onboarded_first(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    cache.save_host(HostSetup(host="gold", root="/repo"))
    cache.save_host(HostSetup(host="crimson", root="/repo"))
    assert [record.host for record in cache.hosts()] == ["crimson", "gold"]


def test_host_raises_for_a_machine_never_set_up(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "db.sqlite")
    with pytest.raises(LookupError, match="has never been set up"):
        cache.host("ghost")
