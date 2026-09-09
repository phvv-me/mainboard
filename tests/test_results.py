from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mainboard import MissionError, Results
from mainboard.cli import build
from mainboard.dispatch.shared import db_file
from mainboard.dispatch.state import Cache, RunRecord

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".json", ".CSV"])
@pytest.mark.parametrize("from_file", [False, True])
def test_query_exports_the_same_rows_without_overwriting(
    tmp_path: Path, suffix: str, from_file: bool
) -> None:
    results = Results(tmp_path)
    sql = "SELECT 'GH200' AS card, 24::BIGINT AS readings, 0.25::DOUBLE AS seconds"
    source = tmp_path / "readings.sql"
    source.write_text(sql, encoding="utf-8")
    target = tmp_path / "exports" / f"readings{suffix}"
    assert results.export(source if from_file else sql, target) == target
    readers = {".csv": pl.read_csv, ".parquet": pl.read_parquet, ".json": pl.read_json}
    assert_frame_equal(readers[suffix.casefold()](target), results.query(sql))
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        results.export("SELECT 0 AS replaced", target)
    assert target.read_bytes() == before
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize(
    ("sql", "suffix", "message"),
    [
        ("SELECT 1", ".xlsx", "output suffix"),
        ("CREATE TABLE invented(x INTEGER)", ".csv", "one SELECT"),
        ("SELECT 1; SELECT 2", ".json", "one SELECT"),
    ],
)
def test_bad_exports_create_no_destination(
    tmp_path: Path, sql: str, suffix: str, message: str
) -> None:
    target = tmp_path / f"refused{suffix}"
    with pytest.raises(ValueError, match=message):
        Results(tmp_path).export(sql, target)
    assert not target.exists()


def test_writer_failure_leaves_no_partial_export(tmp_path: Path) -> None:
    target = tmp_path / "nested.csv"
    with pytest.raises(pl.exceptions.ComputeError, match="nested"):
        Results(tmp_path).export("SELECT [1, 2] AS nested", target)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("from_file", [False, True])
def test_cli_export_uses_project_scope_and_prints_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], from_file: bool
) -> None:
    for project in ["one", "two"]:
        directory = tmp_path / "research" / project / "datasets/experiments/node/evidence/receipts"
        directory = directory / "run=sample"
        directory.mkdir(parents=True)
        pl.DataFrame(
            {
                "run": [project],
                "trial": ["case"],
                "verdict": ["validated"],
                "artifacts": ["{}"],
                "host": ["miyabi-g"],
                "card_name": ["GH200"],
                "commit": ["source"],
            }
        ).write_parquet(directory / "part-0.parquet")
    target = tmp_path / "scoped.parquet"
    sql = "SELECT project, run FROM runs"
    source = tmp_path / "scoped.sql"
    source.write_text(sql, encoding="utf-8")
    query = ["--file", str(source)] if from_file else [sql]
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["query", *query, "--project", "one", "--out", str(target)])
    assert capsys.readouterr().out.strip() == str(target)
    assert pl.read_parquet(target).to_dicts() == [{"project": "one", "run": "one"}]


def test_query_json_stdout_stays_available(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["query", "SELECT 24 AS readings", "--json"])
    assert '"readings":24' in capsys.readouterr().out.replace(" ", "")


def test_sql_file_and_data_paths_use_caller_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    queries = tmp_path / "queries"
    queries.mkdir()
    source = queries / "data.sql"
    sql = "SELECT * FROM read_parquet('data.parquet')"
    source.write_text(sql, encoding="utf-8")
    pl.DataFrame({"value": [7]}).write_parquet(tmp_path / "data.parquet")
    pl.DataFrame({"value": [99]}).write_parquet(queries / "data.parquet")
    results = Results(tmp_path / "different-results-root")
    assert_frame_equal(results.query(source.relative_to(tmp_path)), results.query(sql))
    assert results.query(source).to_dicts() == [{"value": 7}]
    assert source.read_text(encoding="utf-8") == sql


def test_sql_text_ending_in_sql_suffix_remains_text(tmp_path: Path) -> None:
    assert Results(tmp_path).query("SELECT\n  7 AS value -- report.sql").to_dicts() == [
        {"value": 7}
    ]


def test_sql_file_errors_name_the_file(tmp_path: Path) -> None:
    results = Results(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing.sql"):
        results.query(tmp_path / "missing.sql")
    source = tmp_path / "invalid.sql"
    source.write_bytes(b"SELECT '\xff'")
    with pytest.raises(ValueError, match=r"invalid.sql.*UTF-8"):
        results.query(source)


@pytest.mark.parametrize("sql", ["", "SELECT 1; SELECT 2", "CREATE TABLE hidden(x INTEGER)"])
def test_sql_files_obey_the_single_select_guard(tmp_path: Path, sql: str) -> None:
    source = tmp_path / "refused.sql"
    source.write_text(sql, encoding="utf-8")
    target = tmp_path / "refused.parquet"
    with pytest.raises(ValueError, match="one SELECT"):
        Results(tmp_path).export(source, target)
    assert not target.exists()


def test_cli_query_rejects_two_sources_before_reading_a_file(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="mutually exclusive"):
        build(tmp_path)(["query", "SELECT 1", "--file", str(tmp_path / "missing.sql")])


def test_cli_query_preserves_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.sql"):
        build(tmp_path)(["query", "--file", str(tmp_path / "missing.sql")])


@pytest.mark.parametrize(
    ("verdict", "reported", "evidence", "settled"),
    [
        ("ok", "ok", "verified", True),
        ("ok", None, "copied", False),
        ("ok", "ok", "unverified", True),
        ("running", "running", "", False),
        (None, None, "", False),
    ],
)
def test_jobs_distinguish_backend_observations_from_settlement(
    tmp_path: Path, verdict: str | None, reported: str | None, evidence: str, settled: bool
) -> None:
    cache = Cache(db_file(tmp_path))
    record = RunRecord(
        handle="5080",
        target="vast",
        kind="vast",
        script="test_carry.py::test_carry",
        args="",
        git_sha="acquired-source",
        dirty=0,
        submitted_at="2026-09-08T10:40:00Z",
        state="running",
        verdict=verdict,
        reported=reported,
        evidence=evidence,
    )
    cache.record(record)
    assert Results(tmp_path).query(
        "SELECT backend_state, verdict, evidence, settled FROM jobs"
    ).to_dicts() == [
        dict(backend_state="running", verdict=verdict, evidence=evidence, settled=settled)
    ]
    assert cache.run("5080", "vast") == record
