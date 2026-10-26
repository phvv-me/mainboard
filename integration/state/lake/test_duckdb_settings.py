import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import example, given
from hypothesis import settings as tuned
from hypothesis import strategies as st

from mainboard import MissionError, Project, load
from mainboard.state import Lake, schema
from mainboard.state.storage.database import connect, settings

_NAMES = st.text(st.characters(blacklist_characters="=\0", blacklist_categories=["Cs"]))


@pytest.fixture(autouse=True)
def fresh() -> Iterator[None]:
    """Start and end each test with no settings remembered."""
    settings.cache_clear()
    yield
    settings.cache_clear()


def _setting(name: str) -> str | int:
    [(value,)] = connect().execute(f"SELECT current_setting('{name}')").fetchall()
    return value


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        pytest.param({}, {"TimeZone": "UTC"}, id="instants-read-in-utc-by-default"),
        pytest.param(
            {"MB_DUCKDB_THREADS": "3", "MB_DUCKDB_TIMEZONE": "Asia/Tokyo"},
            {"threads": 3, "TimeZone": "Asia/Tokyo"},
            id="the-environment-overrides-a-setting",
        ),
    ],
)
def test_every_connection_takes_the_settings_with_the_environment_winning(
    environment: dict[str, str], expected: dict[str, str | int]
) -> None:
    with patch.dict(os.environ, environment):
        assert {name: _setting(name) for name in expected} == expected


@tuned(max_examples=100)
@example("THREADS;SELECT")
@given(_NAMES.filter(lambda name: not name.lower().isidentifier()))
def test_a_setting_name_that_is_not_a_plain_name_is_refused(name: str) -> None:
    with (
        patch.dict(os.environ, {f"MB_DUCKDB_{name}": "2"}),
        pytest.raises(MissionError, match="plain names"),
    ):
        settings()


def test_the_latest_record_of_a_log_without_drops_is_kept() -> None:
    assert "dropped" not in str(schema.latest(schema.pulse, "key"))


def test_the_lake_lives_where_the_workspace_says(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
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
