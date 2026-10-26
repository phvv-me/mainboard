import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from string import Template

import duckdb
import pytest
from dirty_equals import IsStr

from mainboard import MissionError, Results
from mainboard.state import Lake, schema
from mainboard.state.views import Views

from ..constants import MAPS
from ..footer import Footer
from ..rendering import Rendering

_READ_ONLY = 'on database "lake" which is attached in read-only mode'
_COUNTS = (
    "SELECT (SELECT count(*) FROM lake.runs) AS runs, "
    "(SELECT count(*) FROM lake.runs_log) AS runs_log, "
    "(SELECT count(*) FROM lake.schema_log) AS schema_log"
)
_ORDERED = (
    "SELECT i % 3 AS n, CASE WHEN i % 2 = 0 THEN NULL ELSE '' END AS text FROM range(30) t(i)"
)
_CURRENT_SNAPSHOT = "SELECT * FROM lake.current_snapshot()"


def _forbidden(*args, **kwargs) -> None:
    pytest.fail("a query that does not name the lake must not touch it or hash its result")


@pytest.fixture
def counted(workspace: Path, lake: Lake) -> Callable[[], dict[str, int]]:
    """The row counts of the lake's runs view and two logs, once one run is recorded."""
    lake.append(
        schema.runs_log,
        [{"ts": datetime.now(UTC), "target": "local", "handle": "read-only", "record": {}}],
    )
    results = Results(workspace)
    return lambda: results.rows(_COUNTS)[0]


@pytest.fixture
def unprepared_lake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test when a query prepares a lake or hashes a result."""
    monkeypatch.setattr(Lake, "current", _forbidden)
    monkeypatch.setattr("mainboard.results.ResultHash", _forbidden)


@pytest.fixture
def unattached_lake(unprepared_lake: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test when a query also attaches a lake."""
    monkeypatch.setattr(Lake, "attach", _forbidden)


@pytest.fixture
def bare_root(tmp_path_factory: pytest.TempPathFactory, read_only: Callable[[Path], None]) -> Path:
    """A folder nobody can write to, holding no workspace."""
    folder = tmp_path_factory.mktemp("bare")
    read_only(folder)
    return folder


@pytest.fixture
def lake_root(workspace: Path, lake: Lake) -> Path:
    """A workspace whose lake already exists."""
    return workspace


@pytest.fixture(
    params=[
        pytest.param("bare_root", id="a-read-only-folder-without-a-lake"),
        pytest.param("lake_root", id="a-workspace-holding-a-lake"),
    ]
)
def root(request: pytest.FixtureRequest) -> Path:
    """Where a plain query runs: a folder with no lake, or a workspace that holds one."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def plain_results(
    root: Path, unprepared_lake: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Results]:
    """Results over `root`, whose connection must carry no lake and whose folder stays as found.

    Evidence indexes and jobs may read their own lake sessions, so only the query's own
    connection is checked.
    """
    results = Results(root)
    load = results.views.load

    def unattached(connection: duckdb.DuckDBPyConnection, project: str, *, sql: str) -> None:
        databases = connection.execute("SELECT database_name FROM duckdb_databases()").fetchall()
        assert ("lake",) not in databases
        load(connection, project, sql=sql)

    monkeypatch.setattr(results.views, "load", unattached)
    before = sorted(root.iterdir())
    yield results
    assert sorted(root.iterdir()) == before


@pytest.fixture
def scratch_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session table every query finds already made, since no query can make one for the next."""
    load = Views.load

    def seeded(
        views: Views, connection: duckdb.DuckDBPyConnection, project: str, *, sql: str
    ) -> None:
        connection.execute("CREATE TABLE scratch AS SELECT 1 AS a")
        load(views, connection, project, sql=sql)

    monkeypatch.setattr(Views, "load", seeded)


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        # runs is a view: DuckDB rejects its DELETE before the read-only database check.
        pytest.param(
            "DELETE FROM lake.runs", "Can only delete from base table", id="delete-a-view"
        ),
        pytest.param("DELETE FROM lake.runs_log", _READ_ONLY, id="delete-a-log"),
        pytest.param("DELETE FROM lake.schema_log", _READ_ONLY, id="delete-the-schema-log"),
        pytest.param(
            "UPDATE lake.schema_log SET version = version + 1", _READ_ONLY, id="update-a-log"
        ),
        pytest.param(
            "INSERT INTO lake.schema_log SELECT * FROM lake.schema_log",
            _READ_ONLY,
            id="insert-into-a-log",
        ),
        pytest.param("DROP TABLE lake.schema_log", _READ_ONLY, id="drop-a-log"),
        pytest.param("TRUNCATE TABLE lake.schema_log", _READ_ONLY, id="truncate-a-log"),
        pytest.param(
            "CREATE TABLE lake.query_probe AS SELECT 1 AS n", _READ_ONLY, id="create-a-table"
        ),
        pytest.param(
            "WITH a AS (DELETE FROM lake.schema_log) SELECT 1", _READ_ONLY, id="delete-in-a-with"
        ),
    ],
)
def test_query_lake_writes_are_refused_by_the_engine(
    mb, counted: Callable[[], dict[str, int]], sql: str, message: str
) -> None:
    before = counted()
    assert before["runs"] == before["runs_log"] == 1 and before["schema_log"] > 0
    ran = mb("query", sql)
    assert ran.code == 1 and message in ran.err, ran.said
    assert not ran.out and "Traceback" not in ran.said
    assert counted() == before


@pytest.mark.parametrize(
    ("sql", "mode", "body"),
    [
        pytest.param(
            "SELECT count(*) AS n FROM lake.runs", "json", '[{"n":0}]\n', id="a-view-of-the-lake"
        ),
        pytest.param(
            "SELECT count(*) AS n FROM ducklake_snapshots('lake') "
            "WHERE EXISTS (SELECT 1 FROM lake.schema_log)",
            None,
            IsStr(regex=r"n\n[1-9]\d*\n"),
            id="the-snapshots-function-beside-a-table",
        ),
        pytest.param(
            _CURRENT_SNAPSHOT, None, IsStr(regex=r"\w+\n[1-9]\d*\n"), id="current-snapshot-binds"
        ),
        pytest.param(
            "SELECT * FROM lake.snapshots()",
            None,
            IsStr(regex=r"(?s)snapshot_id\tsnapshot_time\t.*"),
            id="snapshots-binds",
        ),
    ],
)
def test_a_query_naming_the_lake_creates_it_and_pins_its_current_snapshot(
    render: Callable[..., Rendering], workspace: Path, sql: str, mode: str | None, body: str
) -> None:
    shown = render(sql, mode=mode)
    assert shown.body == body and (workspace / ".mb").is_dir()
    assert shown.stamp.snapshot_id == Lake.at(workspace).query(_CURRENT_SNAPSHOT)[0][0] > 0


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            f"{_ORDERED} ORDER BY i", f"{_ORDERED} ORDER BY i DESC", id="the-same-rows-reordered"
        ),
        pytest.param(MAPS, MAPS, id="nested-maps"),
    ],
)
def test_a_stamp_ignores_row_order_caps_the_lakes_existence_and_the_surface(
    render: Callable[..., Rendering], workspace: Path, first: str, second: str
) -> None:
    stamps = [render(first, max_rows=4).stamp, render(second).stamp]
    results = Results(workspace)
    results.stamped(first)
    assert stamps[0] == stamps[1] == results.provenance[0] and stamps[0].snapshot_id == 0
    assert not (workspace / ".mb").exists()
    Lake.at(workspace).ready()
    assert render(first).stamp == stamps[0]


def test_query_snapshot_stays_pinned_beside_an_append(workspace: Path, lake: Lake) -> None:
    results = Results(workspace)
    sql = "SELECT count(*) AS n FROM lake.schema_log WHERE spec = 'after-query'"
    with results.stream(sql) as query:
        snapshot = query.snapshot_id
        lake.append(
            schema.schema_log,
            [{"ts": datetime.now(UTC), "version": 0, "spec": "after-query", "engine": "it"}],
        )
        assert query.cursor.fetchmany(1) == [(0,)]
    with results.stream(sql) as query:
        assert query.snapshot_id > snapshot
        assert query.cursor.fetchmany(1) == [(1,)]


def test_a_plain_query_never_attaches_prepares_or_hashes_a_lake(plain_results: Results) -> None:
    answers = {
        "SELECT 1": [(1,)],
        "SELECT 'lake.fake' AS text": [("lake.fake",)],
        "SELECT run, data FROM metrics": [],
        "SELECT * FROM artifacts": [],
        "SELECT run, trial, verdict, artifacts FROM trials": [],
        "SELECT * FROM runs": [],
        "SELECT * FROM jobs": [],
    }
    asked = {sql: plain_results.query(sql, project="it").fetchall() for sql in answers}
    counted = {sql: len(plain_results.rows(sql, project="it")) for sql in answers}
    assert asked == answers
    assert counted == {sql: len(rows) for sql, rows in answers.items()}
    assert plain_results.provenance == []


@pytest.mark.parametrize(
    ("where", "sql", "message"),
    [
        pytest.param(
            "scratch", 'SELECT * FROM "lake".runs', "not a workspace", id="not-a-workspace"
        ),
        pytest.param(
            ".", "SELECT * FROM lake.runs", "read-only workspace", id="read-only-workspace"
        ),
    ],
)
def test_a_missing_lake_is_created_only_in_a_writable_workspace(
    workspace: Path,
    read_only: Callable[[Path], None],
    unattached_lake: None,
    where: str,
    sql: str,
    message: str,
) -> None:
    root = workspace / where
    root.mkdir(exist_ok=True)
    before = sorted(root.iterdir())
    read_only(workspace)
    with pytest.raises(MissionError, match=message):
        Results(root).query(sql)
    assert Results(root).query("SELECT 1").fetchall() == [(1,)]
    assert sorted(root.iterdir()) == before


@pytest.mark.parametrize(
    ("sql", "files", "pinned"),
    [
        pytest.param(
            "WITH copied AS (COPY (SELECT 7 AS n) TO '$workspace/copied.csv' "
            "(FORMAT csv, HEADER false)) SELECT * FROM copied",
            {"copied.csv": "7\n"},
            False,
            id="a-copy-writes-its-file",
        ),
        pytest.param(
            "INSERT INTO scratch SELECT 2 WHERE EXISTS (SELECT 1 FROM lake.schema_log)",
            {},
            True,
            id="an-insert-fills-a-session-table-while-reading-the-lake",
        ),
    ],
)
def test_a_statement_that_writes_answers_its_count(
    in_process_mb,
    scratch_table: None,
    workspace: Path,
    sql: str,
    files: dict[str, str],
    pinned: bool,
) -> None:
    ran = in_process_mb("query", Template(sql).safe_substitute(workspace=workspace), "--json")
    assert ran.code == 0 and json.loads(ran.out) == [{"Count": 1}], ran.said
    assert (Footer.of(ran.err).snapshot_id > 0) is pinned
    assert {
        path.name: path.read_text(encoding="utf-8") for path in workspace.glob("*.csv")
    } == files


def test_the_jobs_view_names_a_jobs_host_as_job_list_does(workspace: Path) -> None:
    columns = Results(workspace).query("SELECT * FROM jobs").columns

    assert columns[0] == "host" and "server" not in columns
