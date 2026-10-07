"""Every DuckDB connection takes the workspace's settings, a machine's environment winning."""

import pytest

from mainboard.core.errors import MissionError
from mainboard.state import schema
from mainboard.state.database import connect, settings


@pytest.fixture(autouse=True)
def fresh() -> None:
    settings.cache_clear()


def _setting(name: str) -> object:
    [(value,)] = connect().execute(f"SELECT current_setting('{name}')").fetchall()
    return value


def test_instants_read_in_utc_unless_a_setting_says_otherwise() -> None:
    assert _setting("TimeZone") == "UTC"


def test_the_environment_overrides_a_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MB_DUCKDB_THREADS", "3")
    monkeypatch.setenv("MB_DUCKDB_TIMEZONE", "Asia/Tokyo")
    assert (_setting("threads"), _setting("TimeZone")) == (3, "Asia/Tokyo")


def test_a_setting_name_that_is_not_a_plain_name_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_DUCKDB_THREADS;SELECT", "2")
    with pytest.raises(MissionError, match="plain names"):
        connect()


def test_the_latest_record_of_a_log_without_drops_is_kept() -> None:
    assert "dropped" not in str(schema.latest(schema.pulse, "key"))
