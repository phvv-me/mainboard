"""The telemetry views: what each session, day and trailing window used, and never who used it."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.core.events import Event, EventSpool, JobFrame, JobReading
from mainboard.state import TelemetryLake

from .builders import claude_turn, codex_turn, job_frame, lake_with
from .constants import EMAIL

_NOW = datetime.now(UTC).replace(tzinfo=None)
_WAITING = Event(name="mb.test.waiting", at=datetime(2026, 10, 29, 12, 0, tzinfo=UTC))
_SESSIONS = st.sampled_from(["a", "b", "c"])
# A Claude turn as (session, tokens, micro-USD); a Codex one as (conversation, read, output),
# where an output of 0 is the warm-up Codex opens a turn with.
_CLAUDE = st.lists(st.tuples(_SESSIONS, st.integers(0, 10**6), st.integers(0, 10**6)), max_size=8)
_CODEX = st.lists(st.tuples(_SESSIONS, st.integers(1, 10**6), st.integers(0, 3)), max_size=8)

type Turn = tuple[str, int, int]


def _expected(
    claude: Sequence[Turn], *, codex: Sequence[Turn]
) -> dict[str, tuple[int, int, float | None]]:
    """Each session's turns, output and cost, as the views must sum them."""
    sums: dict[str, tuple[int, int, float | None]] = {}
    for session, tokens, micros in claude:
        turns, output, cost = sums.get(f"claude:{session}", (0, 0, 0.0))
        sums[f"claude:{session}"] = (turns + 1, output + tokens, (cost or 0) + micros / 10**6)
    for conversation, _, output in (turn for turn in codex if turn[2]):
        turns, written, _ = sums.get(f"codex:{conversation}", (0, 0, None))
        sums[f"codex:{conversation}"] = (turns + 1, written + output, None)
    return {key: (t, o, None if c is None else round(c, 6)) for key, (t, o, c) in sums.items()}


@given(claude=_CLAUDE, codex=_CODEX)
@example(claude=[("a", 3, 4916)], codex=[("b", 12438, 0), ("b", 15868, 49), ("b", 15944, 5)])
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_each_session_sums_its_turns_and_leaves_codex_warm_ups_out(
    workspace: Path, claude: list[Turn], codex: list[Turn]
) -> None:
    records = [
        *(claude_turn(s, _NOW, tokens=tokens, micros=micros) for s, tokens, micros in claude),
        *(codex_turn(s, _NOW, read=read, cached=read // 2, output=out) for s, read, out in codex),
    ]
    lake = lake_with(workspace, records)
    rows = lake.rows("SELECT session, turns, output, cost_usd FROM sessions")
    assert {row["session"]: (row["turns"], row["output"], row["cost_usd"]) for row in rows} == (
        _expected(claude, codex=codex)
    )


def test_codex_input_counts_its_uncached_tokens_as_claude_does(workspace: Path) -> None:
    lake = lake_with(workspace, [codex_turn("x", _NOW, read=31812, cached=28928, output=54)])
    (row,) = lake.rows("SELECT input, cache_read, output FROM turns")
    assert (row["input"], row["cache_read"], row["output"]) == (2884, 28928, 54)


@pytest.mark.parametrize("view", ["turns", "sessions", "spend", "standings"])
def test_no_view_shows_the_address_every_record_carries(workspace: Path, view: str) -> None:
    lake = lake_with(workspace, [claude_turn("a", _NOW), codex_turn("b", _NOW)])
    rows = lake.rows(f"SELECT * FROM {view}")
    assert rows and EMAIL not in repr(rows)


def test_standings_count_only_the_turns_inside_each_trailing_window(workspace: Path) -> None:
    ages = (timedelta(hours=1), timedelta(days=2), timedelta(days=10))
    lake = lake_with(workspace, [claude_turn("a", _NOW - age) for age in ages])
    rows = lake.rows("SELECT harness, span, turns FROM standings")
    assert {(row["harness"], row["span"]): row["turns"] for row in rows} == {
        ("claude", "5h"): 1,
        ("claude", "7d"): 2,
    }


def test_each_job_shows_its_start_beats_and_peak_memory_and_a_settle_only_when_it_got_one(
    workspace: Path,
) -> None:
    beat = JobFrame.heartbeat
    lake = lake_with(
        workspace,
        [
            job_frame("gold/1", JobFrame.started, _NOW),
            job_frame("gold/1", beat, _NOW, {JobReading.rss_bytes: 7}),
            job_frame("gold/1", beat, _NOW, {JobReading.rss_bytes: 9}),
            job_frame(
                "gold/1",
                JobFrame.settled,
                _NOW,
                {JobReading.exit_code: 3, JobReading.stop: "exited"},
            ),
            job_frame("gold/2", JobFrame.started, _NOW),
            job_frame("gold/2", beat, _NOW, {JobReading.rss_bytes: 5}),
        ],
    )
    rows = lake.rows("SELECT job, beats, peak_rss_bytes, exit_code, stop FROM jobs ORDER BY job")
    assert rows == [
        {"job": "gold/1", "beats": 2, "peak_rss_bytes": 9, "exit_code": 3, "stop": "exited"},
        {"job": "gold/2", "beats": 1, "peak_rss_bytes": 5, "exit_code": None, "stop": None},
    ]


def test_reading_a_lake_the_sink_never_made_is_refused_and_creates_nothing(
    workspace: Path,
) -> None:
    lake = TelemetryLake(root=workspace)
    with pytest.raises(MissionError, match="the sink creates it"):
        lake.rows("SELECT * FROM sessions")
    assert not lake.home.exists()


def test_pending_counts_each_spool_file_past_its_offset_and_a_refused_one_apart(
    workspace: Path,
) -> None:
    lake, sent = TelemetryLake(root=workspace), Event(name="mb.test.sent", at=_WAITING.at)
    with EventSpool(lake.spool / "a.jsonl") as spool:
        spool.append(sent)
        spool.append(_WAITING)
    lake.keep({"a.jsonl": len(sent.line()) + 1})
    (lake.spool / "b.refused").write_bytes(b"not a spool line\nnor this\n")
    rows = [file.row(at=_WAITING.at + timedelta(minutes=1)) for file in lake.pending()]
    oldest = _WAITING.at.astimezone().isoformat(timespec="seconds")
    assert rows == [
        {"file": "a.jsonl", "state": "waiting", "lines": 1, "oldest": oldest, "age_s": 60},
        {"file": "b.refused", "state": "refused", "lines": 2},
    ]
