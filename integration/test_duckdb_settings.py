"""Every DuckDB connection takes the workspace's settings, a machine's environment winning,
the lake lives where the workspace says, and only an undeclared host waits on it."""

from pathlib import Path

import pytest

from mainboard.core.errors import MissionError
from mainboard.core.project import Project
from mainboard.manifest.loading import load
from mainboard.state import schema
from mainboard.state.database import connect, settings
from mainboard.state.lake import Lake


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


def test_the_lake_lives_where_the_workspace_says(workspace: Path, tmp_path_factory) -> None:
    assert Lake.at(workspace).catalog.parent == Project().out(workspace)
    elsewhere = tmp_path_factory.mktemp("portable")
    (workspace / "mb.toml").write_text(
        f'[workspace]\nname = "it"\nlake = "{elsewhere.as_posix()}"\n', encoding="utf-8"
    )
    # A dispatch mirror ships without `.git`, so the center's path does not reach it.
    assert Lake.at(workspace).catalog.parent == Project().out(workspace)
    (workspace / ".git").mkdir()
    lake = Lake.at(workspace)
    assert (lake.catalog, lake.data) == (elsewhere / "lake.sqlite", elsewhere / "lake")


def test_only_an_undeclared_host_reads_the_holds(workspace: Path) -> None:
    (workspace / "mb.toml").write_text(
        '[workspace]\nname = "it"\n[hosts.box]\nkind = "ssh"\n', encoding="utf-8"
    )

    def unreachable() -> dict:
        raise TimeoutError("the lake volume stopped answering")

    manifest = load(workspace / "mb.toml").holding(unreachable)
    assert manifest.profile("box").kind == "ssh"
    assert manifest.profile("local").kind != "ssh"
    with pytest.raises(TimeoutError):
        manifest.profile("rented")
