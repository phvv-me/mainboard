from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mainboard import Results
from mainboard.cli import build
from mainboard.dispatch.shared import db_file
from mainboard.dispatch.state import Cache, RunRecord

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".json", ".CSV"])
def test_query_exports_the_same_rows_without_overwriting(tmp_path: Path, suffix: str) -> None:
    results = Results(tmp_path)
    sql = "SELECT 'GH200' AS card, 24::BIGINT AS readings, 0.25::DOUBLE AS seconds"
    target = tmp_path / "exports" / f"readings{suffix}"
    assert results.export(sql, target) == target
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


def test_cli_export_uses_project_scope_and_prints_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(
            ["query", "SELECT project, run FROM runs", "--project", "one", "--out", str(target)]
        )
    assert capsys.readouterr().out.strip() == str(target)
    assert pl.read_parquet(target).to_dicts() == [{"project": "one", "run": "one"}]


def test_query_json_stdout_stays_available(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="^0$"):
        build(tmp_path)(["query", "SELECT 24 AS readings", "--json"])
    assert '"readings":24' in capsys.readouterr().out.replace(" ", "")


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
