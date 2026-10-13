"""Bounded SQL output, rendered multiset identities, and figure provenance."""

import json
import re
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest
from PIL import Image

from mainboard.core.errors import MissionError
from mainboard.render.query import Query, canonical, execution_timeout
from mainboard.results import Results
from mainboard.state import schema
from mainboard.state.lake import Lake

from .conftest import _in_process


def _footer(text: str) -> tuple[int, str, int]:
    match = re.search(r"(\d+) rows.*sha256=([0-9a-f]{64}).*snapshot_id=(\d+)", text)
    assert match is not None, text
    return int(match[1]), match[2], int(match[3])


def test_query_keeps_empty_columns_and_distinguishes_null_text(mb) -> None:
    ran = mb("query", "SELECT NULL AS absent, '' AS empty, 'NULL' AS literal")
    assert ran.code == 0, ran.said
    assert ran.out.splitlines()[:2] == ["absent\tempty\tliteral", 'NULL\t\t"NULL"']
    assert _footer(ran.out)[0] == 1
    empty = mb("query", "SELECT NULL AS absent, '' AS empty WHERE false")
    assert empty.code == 0, empty.said
    assert empty.out.splitlines()[0] == "absent\tempty"
    assert _footer(empty.out)[0] == 0
    human = mb("query", "SELECT NULL AS absent, '' AS empty", "--human")
    assert human.code == 0 and "absent" in human.out and "empty" in human.out


@pytest.mark.parametrize("verb", ["EXPLAIN", "EXPLAIN ANALYZE", "DESCRIBE", "SUMMARIZE"])
def test_query_renders_inspection_statements(mb, verb: str) -> None:
    ran = mb("query", f"{verb} SELECT 1 AS number")
    assert ran.code == 0, ran.said
    assert len(ran.out.splitlines()) >= 3
    assert _footer(ran.out)[0] > 0


def test_query_runs_writes_and_table_functions_with_rendered_results(
    mb, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = mb("query", "CREATE TABLE scratch AS SELECT 1 AS a")
    assert created.code == 0, created.said
    assert created.out.splitlines()[:2] == ["Count", "1"]
    assert _footer(created.out)[2] == 0 and not (workspace / ".mb").exists()
    target = workspace / "copied.csv"
    copied = mb(
        "query",
        f"WITH copied AS (COPY (SELECT 7 AS n) TO '{target}' (FORMAT csv, HEADER false)) "
        "SELECT * FROM copied",
    )
    assert copied.code == 0 and target.read_text(encoding="utf-8") == "7\n", copied.said
    dynamic = mb("query", "SELECT * FROM query('SELECT 7 AS n')", "--json")
    assert dynamic.code == 0 and json.loads(dynamic.out) == [{"n": 7}], dynamic.said
    ranged = mb("query", "SELECT * FROM range(2)", "--json")
    assert ranged.code == 0 and json.loads(ranged.out) == [{"range": 0}, {"range": 1}]
    version = mb("query", "PRAGMA version")
    assert version.code == 0 and "library_version" in version.out, version.said
    assert not (workspace / ".mb").exists()

    fresh = mb("query", "SELECT count(*) AS n FROM lake.runs", "--json")
    assert fresh.code == 0 and json.loads(fresh.out) == [{"n": 0}], fresh.said
    assert _footer(fresh.err)[2] > 0 and (workspace / ".mb").is_dir()

    original = Results._views

    def views(
        results: Results, connection: duckdb.DuckDBPyConnection, project: str, sql: str
    ) -> None:
        connection.execute("CREATE TABLE scratch AS SELECT 1 AS a")
        original(results, connection, project, sql)

    # Each CLI call owns a fresh session; seed the INSERT's scratch table in that session.
    with monkeypatch.context() as session:
        session.setattr(Results, "_views", views)
        inserted = _in_process(workspace)(
            "query",
            "INSERT INTO scratch SELECT 2 WHERE EXISTS (SELECT 1 FROM lake.schema_log)",
            "--json",
        )
    assert inserted.code == 0 and json.loads(inserted.out) == [{"Count": 1}], inserted.said
    assert _footer(inserted.err)[2] > 0
    inspected = mb(
        "query",
        "SELECT count(*) AS n FROM ducklake_snapshots('lake') "
        "WHERE EXISTS (SELECT 1 FROM lake.schema_log)",
    )
    assert inspected.code == 0 and _footer(inspected.out)[2] > 0, inspected.said


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        # runs is a view: DuckDB rejects its DELETE before the read-only database check.
        ("DELETE FROM lake.runs", "Can only delete from base table"),
        ("DELETE FROM lake.runs_log", "read-only"),
        ("DELETE FROM lake.schema_log", "read-only"),
        ("UPDATE lake.schema_log SET version = version + 1", "read-only"),
        ("INSERT INTO lake.schema_log SELECT * FROM lake.schema_log", "read-only"),
        ("DROP TABLE lake.schema_log", "read-only"),
        ("TRUNCATE TABLE lake.schema_log", "read-only"),
        ("CREATE TABLE lake.query_probe AS SELECT 1 AS n", "read-only"),
        ("WITH a AS (DELETE FROM lake.schema_log) SELECT 1", "read-only"),
    ],
)
def test_query_lake_writes_are_refused_by_the_engine(
    mb, workspace: Path, sql: str, message: str
) -> None:
    lake = Lake.at(workspace).ready()
    lake.append(
        schema.runs_log,
        [{"ts": datetime.now(UTC), "target": "local", "handle": "read-only", "record": {}}],
    )
    results = Results(workspace)
    counts = (
        "SELECT (SELECT count(*) FROM lake.runs) AS runs, "
        "(SELECT count(*) FROM lake.runs_log) AS runs_log, "
        "(SELECT count(*) FROM lake.schema_log) AS schema_log"
    )
    previous = results.rows(counts)
    assert previous[0]["runs"] == previous[0]["runs_log"] == 1
    assert previous[0]["schema_log"] > 0
    ran = mb("query", sql)
    assert ran.code == 1 and message in ran.err, ran.said
    if message == "read-only":
        assert "lake" in ran.err, ran.said
    assert not ran.out and "Traceback" not in ran.said
    assert results.rows(counts) == previous


def test_query_without_result_columns_reports_success(capsys: pytest.CaptureFixture[str]) -> None:
    for description in (None, []):
        for mode in (None, "json", "human"):
            for limit in (0, 1):
                cursor = Mock(spec=duckdb.DuckDBPyConnection, description=description)
                stamp = Query(cursor, snapshot_id=0).render(
                    mode=mode, max_rows=limit, max_bytes=limit
                )
                captured = capsys.readouterr()
                status = captured.err if mode == "json" else captured.out
                assert status.startswith("ok\n") and _footer(status) == (0, stamp.sha256, 0)
                assert "displayed=0; omitted=0" in status and stamp.complete
                if mode == "json":
                    assert json.loads(captured.out) == []


def test_query_caps_rows_and_bytes_with_a_counted_middle(mb) -> None:
    ran = mb("query", "SELECT * FROM range(20)", "--max-rows", "6", "--max-bytes", "0")
    assert ran.code == 0, ran.said
    assert ran.out.splitlines()[:-1] == [
        "range",
        "0",
        "1",
        "2",
        "… 14 rows omitted (--max-rows 6) …",
        "17",
        "18",
        "19",
    ]
    assert _footer(ran.out)[0] == 20
    wide = mb(
        "query",
        "SELECT i, repeat('x', 20) AS text FROM range(100) t(i)",
        "--max-rows",
        "0",
        "--max-bytes",
        "300",
    )
    assert wide.code == 0, wide.said
    lines = wide.out.splitlines()[1:-1]
    assert sum(len(line.encode()) + 1 for line in lines if "omitted" not in line) <= 300
    assert "rows omitted" in wide.out and _footer(wide.out)[0] == 100
    full = mb("query", "SELECT * FROM range(20)", "--max-rows", "0", "--max-bytes", "-1")
    assert full.code == 0 and len(full.out.splitlines()) == 22
    assert "rows omitted" not in full.out and _footer(full.out)[0] == 20


def test_query_stops_a_runaway_range_and_labels_the_partial_hash(mb) -> None:
    start = time.monotonic()
    ran = mb("query", "SELECT * FROM range(1000000000000)", timeout=15)
    assert ran.code == 0, ran.said
    assert time.monotonic() - start < 10
    assert "at least" in ran.out and "query stopped early" in ran.out
    assert "(partial)" in ran.out and "rows omitted" in ran.out
    counted, _, _ = _footer(ran.out)
    assert 1000 <= counted <= 11001
    assert len(ran.out.encode()) < 10500
    assert len(ran.out.splitlines()) <= 1003


def test_json_stream_caps_and_preserves_its_array(mb) -> None:
    ran = mb(
        "query",
        "SELECT * FROM range(20)",
        "--json",
        "--max-rows",
        "6",
        "--max-bytes",
        "0",
    )
    assert ran.code == 0, ran.said
    values = json.loads(ran.out)
    assert values[:3] == [{"range": i} for i in range(3)]
    marker = values[3]["_truncated"]
    assert marker["omitted"] == 14 and marker["rows"] == 20 and marker["complete"]
    assert values[-3:] == [{"range": i} for i in range(17, 20)]
    assert _footer(ran.err)[0] == 20
    full = mb(
        "query",
        "SELECT NULL AS absent, '' AS empty",
        "--json",
        "--max-rows",
        "-1",
        "--max-bytes",
        "0",
    )
    assert json.loads(full.out) == [{"absent": None, "empty": ""}]
    assert _footer(full.err)[0] == 1


def test_hash_is_a_stable_multiset_of_the_rendered_rows(mb, workspace) -> None:
    sql = (
        "SELECT i % 3 AS n, CASE WHEN i % 2 = 0 THEN NULL ELSE '' END AS text FROM range(30) t(i)"
    )
    stamps = []
    for order in ("i", "i DESC", "i"):
        ran = mb("query", sql + f" ORDER BY {order}", "--max-rows", "4")
        assert ran.code == 0, ran.said
        stamps.append(_footer(ran.out))
    assert stamps[0] == stamps[1] == stamps[2]
    assert stamps[0][0] == 30
    # Fresh workspaces no longer create a lake just to report a snapshot.
    assert stamps[0][2] == 0 and not (workspace / ".mb").exists()
    Lake.at(workspace).ready()
    ran = mb("query", sql)
    assert ran.code == 0 and _footer(ran.out) == stamps[0], ran.said
    ran = mb("query", "SELECT count(*) AS n FROM lake.runs")
    assert ran.code == 0, ran.said
    with Lake.at(workspace).open() as connection:
        current = connection.execute("SELECT * FROM lake.current_snapshot()").fetchone()
        assert current is not None and _footer(ran.out)[2] == current[0]
    hashes = []
    for expression in ("0.0::DOUBLE", "-0.0::DOUBLE", "1.00000000000001::DOUBLE"):
        ran = mb("query", f"SELECT {expression} AS n")
        hashes.append(_footer(ran.out)[1])
    assert hashes[0] == hashes[1] != hashes[2]
    one = mb("query", "SELECT 1.0::DOUBLE AS n")
    assert hashes[2] == _footer(one.out)[1]
    for text in ("NULL", "''", "'NULL'"):
        ran = mb("query", f"SELECT {text} AS n")
        hashes.append(_footer(ran.out)[1])
    assert len(set(hashes[-3:])) == 3
    once = mb("query", "SELECT 1 AS n")
    twice = mb("query", "SELECT 1 AS n FROM range(2)")
    assert _footer(once.out)[1] != _footer(twice.out)[1]
    nan_payloads = [
        struct.unpack(">d", bits.to_bytes(8))[0]
        for bits in (0x7FF8000000000001, 0x7FF8000000000002)
    ]
    assert canonical(nan_payloads[0]) == canonical(nan_payloads[1]) == "NaN"


def test_execution_timeout_interrupts_before_unbounded_work(mb) -> None:
    start = time.monotonic()
    ran = mb(
        "query",
        "SELECT sum(i) AS n FROM range(1000000000000) t(i)",
        "--timeout",
        "0.02",
        timeout=15,
    )
    assert ran.code != 0, ran.said
    assert ran.out.startswith("n\n") and "query timed out" in ran.out
    assert "timed out" in ran.err
    assert _footer(ran.out)[0] == 0 and "(partial)" in ran.out
    assert time.monotonic() - start < 10


def test_plot_records_the_same_result_identity(mb, workspace) -> None:
    sql = "SELECT i AS x, i * 2 AS y FROM range(5) t(i)"
    queried = mb("query", sql)
    expected = _footer(queried.out)
    target = workspace / "figure.png"
    plotted = mb("paper", "plot", sql, "--x", "x", "--y", "y", "--out", str(target))
    assert plotted.code == 0, plotted.said
    assert target.is_file() and _footer(plotted.out) == expected
    [stamp] = json.loads(target.with_name("figure.png.metadata.json").read_text(encoding="utf-8"))[
        "results"
    ]
    assert (stamp["rows"], stamp["sha256"], stamp["snapshot_id"]) == expected
    assert stamp["complete"]
    with Image.open(target) as image:
        assert json.loads(image.info["Description"])["results"] == [stamp]
    relation = Results(workspace).query(sql)
    assert relation.fetchall() == [(i, i * 2) for i in range(5)]


def test_query_snapshot_stays_pinned_beside_an_append(workspace) -> None:
    lake = Lake.at(workspace).ready()
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


def test_multiple_statements_keep_the_programmatic_contract_and_cli_error(mb, workspace) -> None:
    with pytest.raises(ValueError, match="one statement"):
        Results(workspace).query("SELECT 1; SELECT 2")
    ran = mb("query", "SELECT 1; SELECT 2")
    assert ran.code == 1 and "one statement" in ran.err
    assert not (workspace / ".mb").exists()


def test_plain_query_neither_attaches_nor_stamps_a_root_without_a_lake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs) -> None:
        pytest.fail("a plain file query must not attach or hash")

    monkeypatch.setattr(Lake, "attach", forbidden)
    monkeypatch.setattr("mainboard.results.ResultHash", forbidden)
    results = Results(tmp_path)
    tmp_path.chmod(0o555)
    try:
        assert results.query("SELECT 1").fetchall() == [(1,)]
        assert results.query("SELECT 'lake.fake' AS text").fetchall() == [("lake.fake",)]
        for sql in (
            "SELECT run, data FROM metrics",
            "SELECT * FROM artifacts",
            "SELECT run, trial, verdict, artifacts FROM trials",
            "SELECT * FROM runs",
            "SELECT * FROM jobs",
        ):
            assert results.query(sql, project="it").fetchall() == []
            assert results.rows(sql, project="it") == []
        assert results.provenance == [] and not (tmp_path / ".mb").exists()
    finally:
        tmp_path.chmod(0o755)


def test_plain_audit_queries_skip_an_existing_lake(workspace, monkeypatch) -> None:
    Lake.at(workspace).ready()

    def forbidden(*args, **kwargs) -> None:
        pytest.fail("a plain audit query must not attach, prepare a lake, or hash")

    monkeypatch.setattr(Lake, "current", forbidden)
    monkeypatch.setattr("mainboard.results.ResultHash", forbidden)
    results = Results(workspace)
    original = results._views

    def views(connection, project, sql) -> None:
        # Evidence indexes and jobs may read their own lake sessions. The query's connection
        # must not pay for an unrelated lake attach.
        assert ("lake",) not in connection.execute(
            "SELECT database_name FROM duckdb_databases()"
        ).fetchall()
        original(connection, project, sql)

    monkeypatch.setattr(results, "_views", views)
    for sql in (
        "SELECT run, data FROM metrics",
        "SELECT * FROM artifacts",
        "SELECT run, trial, verdict, artifacts FROM trials",
        "SELECT * FROM runs",
        "SELECT * FROM jobs",
    ):
        assert results.query(sql, project="it").fetchall() == []
        assert results.rows(sql, project="it") == []
    assert results.provenance == []


def test_missing_lake_requires_a_writable_workspace(tmp_path, workspace, monkeypatch) -> None:
    def forbidden(*args, **kwargs) -> None:
        pytest.fail("an unavailable lake must not be created or attached")

    monkeypatch.setattr(Lake, "attach", forbidden)
    monkeypatch.setattr(Lake, "current", forbidden)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(MissionError, match="not a workspace"):
        Results(scratch).query('SELECT * FROM "lake".runs')
    assert list(scratch.iterdir()) == []
    workspace.chmod(0o555)
    try:
        with pytest.raises(MissionError, match="read-only workspace"):
            Results(workspace).query("SELECT * FROM lake.runs")
        assert Results(workspace).query("SELECT 1").fetchall() == [(1,)]
        assert not (workspace / ".mb").exists()
    finally:
        workspace.chmod(0o755)


@pytest.mark.parametrize("mode", [[], ["--json"], ["--human"]])
def test_duplicate_column_names_are_rejected(mb, workspace: Path, mode: list[str]) -> None:
    sql = "SELECT 1 AS id, 2 AS id"
    ran = mb("query", sql, *mode)
    assert ran.code != 0 and "unique AS alias" in ran.err and not ran.out
    with pytest.raises(MissionError, match="duplicate column names"):
        Results(workspace).stamped(sql)
    target = workspace / "duplicates.json"
    exported = mb("query", sql, "--out", str(target))
    assert exported.code != 0 and "unique AS alias" in exported.err and not target.exists()


def test_nested_maps_and_structs_have_deterministic_display_and_hash(mb, workspace: Path) -> None:
    sql = (
        "SELECT map {DATE '2020-01-02': {'items': [1, NULL]}, "
        "DATE '2020-01-01': {'items': [2, 3]}} AS dates, "
        "map {TIMESTAMP '2020-01-01 12:00:00': [4, 5]} AS times"
    )
    ran = mb("query", sql, "--json")
    assert ran.code == 0, ran.said
    [row] = json.loads(ran.out)
    assert list(row["dates"]) == ["2020-01-01", "2020-01-02"]
    assert row["dates"]["2020-01-02"] == {"items": [1, None]}
    assert row["times"] == {"2020-01-01 12:00:00": [4, 5]}
    results = Results(workspace)
    assert results.stamped(sql).fetchall()
    assert results.provenance[0].sha256 == _footer(ran.err)[1]


def test_byte_marker_reports_empty_and_single_row_displays(mb) -> None:
    empty = mb("query", "SELECT repeat('x', 6000) AS text", "--max-bytes", "100")
    assert empty.code == 0 and empty.out.startswith("text\n")
    assert "1 rows omitted (--max-bytes 100; 0 rows displayed)" in empty.out
    assert _footer(empty.out)[0] == 1
    one = mb("query", "SELECT i, repeat('x', 6000) AS text FROM range(10) t(i)")
    assert one.code == 0 and one.out.splitlines()[1].startswith("0\t")
    assert "9 rows omitted (--max-bytes 10000; 1 row displayed)" in one.out
    assert _footer(one.out)[0] == 10
    row_cap = mb(
        "query",
        "SELECT repeat('x', 60) AS text FROM range(3)",
        "--max-rows",
        "1",
        "--max-bytes",
        "100",
    )
    assert row_cap.code == 0 and row_cap.out.splitlines()[1] == "x" * 60
    assert "2 rows omitted" in row_cap.out and "1 row displayed" in row_cap.out
    json_cap = mb("query", "SELECT repeat('x', 6000) AS text", "--json", "--max-bytes", "1")
    [marker] = json.loads(json_cap.out)
    assert marker["_truncated"]["columns"] == ["text"]
    assert marker["_truncated"]["displayed"] == 0 and marker["_truncated"]["omitted"] == 1


@pytest.mark.parametrize(
    "sql",
    [
        "/* 日本 */ EXPLAIN (FORMAT TEXT) SELECT 'ANALYZE' AS literal",
        "EXPLAIN (SELECT 'ANALYZE' AS literal)",
        "EXPLAIN ((SELECT 'ANALYZE' AS literal))",
        "EXPLAIN (FORMAT TEXT) (SELECT 'ANALYZE' AS literal)",
    ],
)
def test_explain_plan_is_multiline_and_plain_query_can_materialize_it(
    mb, workspace: Path, sql: str
) -> None:
    ran = mb("query", sql)
    assert ran.code == 0, ran.said
    assert "\\n" not in ran.out and len(ran.out.splitlines()) > 5
    assert "Projection" in ran.out
    assert Results(workspace).query(sql).fetchall()


def test_export_timeout_does_not_publish_an_incomplete_file(mb, workspace: Path) -> None:
    target = workspace / "timed-out.csv"
    ran = mb(
        "query",
        "SELECT sum(i) FROM range(1000000000000) t(i)",
        "--timeout",
        "0.02",
        "--out",
        str(target),
    )
    assert ran.code != 0 and "timed out" in ran.err and not target.exists()
    assert list(workspace.iterdir()) == [workspace / "mb.toml"]


@pytest.mark.parametrize("option", ["--json", "--human", "--max-rows", "--max-bytes"])
def test_export_rejects_display_options(mb, workspace: Path, option: str) -> None:
    args = [option, "1"] if option.startswith("--max-") else [option]
    target = workspace / "out.csv"
    ran = mb("query", "SELECT 1", "--out", str(target), *args)
    assert ran.code != 0 and "--out exports all rows" in ran.err and not target.exists()


def test_query_help_documents_snapshot_and_remote_timeout_scope(mb) -> None:
    ran = mb("query", "--help")
    help_text = " ".join(ran.out.split())
    assert "not snapshot-pinned" in help_text and "server limit" in help_text
    assert "attached read-only" in help_text and "session scratch tables" in help_text
    assert "AT (VERSION => n)" in help_text and "recovers prior snapshots" in help_text


def test_forwarded_timeout_errors_are_recognized() -> None:
    assert execution_timeout(duckdb.IOException("Timeout was reached error for HTTP POST"))
    assert execution_timeout(duckdb.IOException("remote query interrupted"))
    assert not execution_timeout(duckdb.CatalogException("table not found"))


def test_a_consumer_closing_the_pipe_stops_query_without_a_traceback(workspace: Path) -> None:
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from mainboard.cli import main; main()",
            "query",
            "SELECT * FROM range(1000000000000)",
            "--max-rows",
            "0",
            "--max-bytes",
            "0",
            "--timeout",
            "0",
        ],
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout is not None and child.stderr is not None
        assert child.stdout.readline() == "range\n"
        child.stdout.close()
        assert child.wait(timeout=15) == 0
        assert not child.stderr.read()


def test_json_timeout_finishes_a_parseable_partial_array(mb) -> None:
    ran = mb(
        "query",
        "SELECT sum(i) FROM range(1000000000000) t(i)",
        "--json",
        "--max-rows",
        "0",
        "--max-bytes",
        "0",
        "--timeout",
        "0.02",
    )
    assert ran.code != 0 and "timed out" in ran.err
    [marker] = json.loads(ran.out)
    assert marker["_truncated"]["timed_out"] and not marker["_truncated"]["complete"]


def test_uncapped_human_output_can_stream_multiple_batches(mb) -> None:
    ran = mb("query", "SELECT * FROM range(33)", "--human", "--max-rows", "0", "--max-bytes", "0")
    assert ran.code == 0, ran.said
    assert _footer(ran.out)[0] == 33 and "32" in ran.out and "omitted=0" in ran.out


def test_lake_table_functions_bind_after_attach(workspace: Path) -> None:
    Lake.at(workspace).ready()
    for sql in ("SELECT * FROM lake.current_snapshot()", "SELECT * FROM lake.snapshots()"):
        with Results(workspace).stream(sql) as query:
            assert query.cursor.fetchmany(1)
