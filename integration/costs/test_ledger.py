from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import func, select

from mainboard.costs import Ledger, Observation
from mainboard.state import Lake, schema

_CLOUD_COLUMNS = ("tier", "datacenter", "verdict", "ready_s", "probe_mbps")
_TEXT = st.text(st.characters(codec="utf-8", exclude_characters="\x00"), max_size=20)
_NUMBER = st.floats(min_value=0, max_value=1e9, allow_nan=False)
_OBSERVATIONS = st.builds(
    Observation,
    provider=_TEXT,
    gpu=_TEXT,
    region=_TEXT,
    t_submit=_NUMBER,
    t_running=_NUMBER,
    t_ended=_NUMBER,
    billed_usd=_NUMBER,
    tier=_TEXT,
    datacenter=_TEXT,
    verdict=_TEXT,
    ready_s=_NUMBER,
    probe_mbps=_NUMBER,
)

_HELD_SQL = (
    "SELECT column_name FROM duckdb_columns() "
    "WHERE database_name = 'lake' AND table_name = 'costs'"
)


@pytest.fixture(scope="module")
def ledger(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Ledger]:
    """A ledger over a lake of its own, created at this release's schema."""
    folder = tmp_path_factory.mktemp("ledger")
    (folder / "mb.toml").write_text('[workspace]\nname = "it"\n', encoding="utf-8")
    session = Lake.at(folder).ready().session()
    yield Ledger(session)
    session.close()


@pytest.fixture
def before(workspace: Path) -> Lake:
    """A lake as the release before the cloud columns left it: one old row, no new columns."""
    lake = Lake.at(workspace).ready()
    with lake.open(write=True) as connection:
        connection.execute(
            "INSERT INTO lake.costs(provider, t_submit, billed_usd) VALUES ('vast', 1, 0.5)"
        )
        for column in _CLOUD_COLUMNS:
            connection.execute(f"ALTER TABLE lake.costs DROP COLUMN {column}")
        connection.execute("DELETE FROM lake.schema_log")
    return lake


def _columns(lake: Lake) -> list[str]:
    with lake.open() as connection:
        return [name for (name,) in connection.execute(_HELD_SQL).fetchall()]


def test_a_lake_before_the_cloud_columns_still_answers_a_read_only_attach(before: Lake) -> None:
    with before.open() as connection:
        rows = connection.execute("SELECT provider, billed_usd FROM lake.costs").fetchall()
    assert rows == [("vast", 0.5)]
    assert not set(_CLOUD_COLUMNS) & set(_columns(before))


def test_a_lake_gains_the_cloud_columns_once_however_often_it_is_opened(before: Lake) -> None:
    before.current().current()
    with before.open(write=True) as connection:
        before.evolve(connection)
    logged = before.query(select(schema.schema_log.c.version))
    assert logged == [(9,)]
    assert _columns(before)[-5:] == list(_CLOUD_COLUMNS)


def test_an_observation_recorded_before_the_cloud_columns_reads_their_defaults(
    before: Lake,
) -> None:
    (old,) = Ledger(before.current().session()).observations()
    assert old == Observation(provider="vast", t_submit=1.0, billed_usd=0.5)
    before.session().close()


@given(observation=_OBSERVATIONS)
def test_an_observation_with_its_cloud_columns_reads_back_as_recorded(
    ledger: Ledger, observation: Observation
) -> None:
    ledger.record(observation)
    count = ledger.session.rows(select(func.count()).select_from(schema.costs))[0][0]
    assert ledger.observations()[count - 1] == observation
