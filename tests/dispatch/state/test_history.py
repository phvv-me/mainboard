from pathlib import Path
from time import monotonic

import pytest

from mainboard.dispatch.state import History


def test_history_replays_every_recorded_invocation_oldest_to_newest() -> None:
    history = History(Path(":memory:"))
    history.record("submit", (3, "gold", "job.sh"), monotonic(), "ok", handle="H1")
    history.record("ls", (), monotonic(), "error", detail="boom")
    first, second = history.recent(10)
    assert (first.command, first.target, first.handle, first.outcome) == (
        "submit",
        "gold",
        "H1",
        "ok",
    )
    assert first.args == ["3", "gold", "job.sh"]
    assert first.duration_ms is not None
    assert (second.command, second.target, second.detail) == ("ls", None, "boom")


def test_a_disabled_history_never_touches_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAINBOARD_NO_HISTORY", "1")
    history = History(tmp_path / "db.sqlite")
    history.record("submit", (), monotonic(), "ok")
    assert history.recent(10) == []
    assert not (tmp_path / "db.sqlite").exists()
