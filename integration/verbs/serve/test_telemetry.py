"""`serve sessions`, `serve usage` and `serve pending` read the telemetry views for anyone;
`serve process collect` and `serve process drain` are what the supervisor runs."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from mainboard.core.events import Event, EventSpool
from mainboard.state import TelemetryLake

from ...state.telemetry.builders import claude_turn, codex_turn, lake_with

_NOW = datetime.now(UTC).replace(tzinfo=None)
_AGENT = {"MB_AGENT": "1"}


def test_serve_sessions_lists_the_sessions_newest_first(mb, workspace: Path) -> None:
    lake_with(workspace, [claude_turn("old", _NOW - timedelta(hours=2)), codex_turn("new", _NOW)])
    ran = mb("serve", "sessions", env=_AGENT)
    rows = [line.split("\t")[0] for line in ran.out.splitlines()[1:]]
    assert (ran.code, rows) == (0, ["codex:new", "claude:old"])


def test_serve_sessions_given_a_session_lists_its_turns(mb, workspace: Path) -> None:
    lake_with(workspace, [claude_turn("a", _NOW), claude_turn("a", _NOW), claude_turn("b", _NOW)])
    ran = mb("serve", "sessions", "claude:a", env=_AGENT)
    assert (ran.code, len(ran.out.splitlines())) == (0, 3)


def test_serve_usage_prints_the_standings_then_each_days_spend(mb, workspace: Path) -> None:
    lake_with(workspace, [claude_turn("a", _NOW, micros=4916)])
    ran = mb("serve", "usage", env=_AGENT)
    lines = ran.out.splitlines()
    assert (ran.code, lines.index("# standings") < lines.index("# spend")) == (0, True)
    assert sum("0.004916" in line for line in lines) == 3


def test_serve_usage_without_a_lake_says_the_sink_makes_it(mb, workspace: Path) -> None:
    ran = mb("serve", "usage", env=_AGENT)
    assert (ran.code, "the sink creates it" in ran.err) == (1, True)
    assert not (workspace / ".mb" / "telemetry").exists()


def test_serve_process_collect_with_an_empty_spool_delivers_nothing(mb, workspace: Path) -> None:
    ran = mb("serve", "process", "collect", env=_AGENT)
    assert (ran.code, "delivered 0 spooled events" in ran.err) == (0, True)


def test_serve_process_drain_with_the_sink_down_ends_clean_and_pending_shows_the_line(
    mb, workspace: Path
) -> None:
    with EventSpool(TelemetryLake(root=workspace).spool / "kept.jsonl") as spool:
        spool.append(Event(name="mb.test.kept", at=datetime.now(UTC)))
    drained = mb("serve", "process", "drain", env=_AGENT)
    pending = mb("serve", "pending", env=_AGENT)
    assert (drained.code, "1 events in 1 spool files wait" in drained.err) == (0, True)
    assert (pending.code, pending.out.splitlines()[1].split("\t")[:3]) == (
        0,
        ["kept.jsonl", "waiting", "1"],
    )
