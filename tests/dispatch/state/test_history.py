from pathlib import Path
from time import monotonic

import pytest
from mainboard.dispatch.state import History


def test_record_and_recent_round_trip(tmp_path: Path) -> None:
    history = History(tmp_path / "db.sqlite")
    history.record("submit", ("gold", "job.sh"), monotonic(), "ok", handle="H1")
    [event] = history.recent(10)
    assert event.command == "submit"
    assert event.target == "gold"
    assert event.handle == "H1"
    assert event.outcome == "ok"


def test_recent_orders_oldest_to_newest(tmp_path: Path) -> None:
    history = History(tmp_path / "db.sqlite")
    history.record("a", (), monotonic(), "ok")
    history.record("b", (), monotonic(), "ok")
    events = history.recent(10)
    assert [event.command for event in events] == ["a", "b"]


def test_record_captures_a_detail_on_error(tmp_path: Path) -> None:
    history = History(tmp_path / "db.sqlite")
    history.record("submit", (), monotonic(), "error", detail="boom")
    [event] = history.recent(10)
    assert event.detail == "boom"


def test_target_is_the_first_string_argument(tmp_path: Path) -> None:
    history = History(tmp_path / "db.sqlite")
    history.record("submit", (3, "gold", "job.sh"), monotonic(), "ok")
    [event] = history.recent(10)
    assert event.target == "gold"


def test_disabled_history_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAINBOARD_NO_HISTORY", "1")
    history = History(tmp_path / "db.sqlite")
    history.record("submit", (), monotonic(), "ok")
    assert history.recent(10) == []
    assert not (tmp_path / "db.sqlite").exists()
