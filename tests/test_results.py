import json
from compression import zstd
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mainboard import MissionError, Results
from mainboard.cli import build
from mainboard.dispatch.shared import db_file
from mainboard.dispatch.state import Cache, RunRecord
from mainboard.observe import Frame, Kind, encode
from mainboard.trials.artifacts import Artifacts

if TYPE_CHECKING:
    from pathlib import Path


def test_experiment_scope_does_not_open_unrelated_evidence(tmp_path: Path) -> None:
    root = tmp_path / "research/example/datasets/experiments"
    receipt = root / "selected/evidence/receipts/run=sample/part-0.parquet"
    receipt.parent.mkdir(parents=True)
    pl.DataFrame({"run": ["selected"], "trial": ["case"], "artifacts": ["{}"]}).write_parquet(
        receipt
    )
    unrelated = root / "unrelated/evidence/receipts/run=sample/part-0.parquet"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"not a receipt")
    event = root / "unrelated/evidence/artifacts/run/case/events/live.ndjson"
    event.parent.mkdir(parents=True)
    event.write_text("not an event")
    results = Results(tmp_path, experiment="selected")
    assert results.query("SELECT run FROM trials")["run"].to_list() == ["selected"]
    assert results.query("SELECT * FROM events").is_empty()
    assert results.table("missing.v1").is_empty()


@pytest.mark.parametrize("experiment", ["..", ".", "../other", "a/b", "a*", "a?"])
def test_experiment_scope_requires_one_directory(tmp_path: Path, experiment: str) -> None:
    with pytest.raises(ValueError, match="one experiment directory"):
        Results(tmp_path, experiment=experiment)


def test_run_selection_reads_only_selected_artifacts_and_keeps_verification(
    tmp_path: Path,
) -> None:
    project = tmp_path / "research/example"
    evidence = project / "datasets/experiments/node/evidence"
    directory = evidence / "artifacts/complete/case"
    writer = Artifacts(project, directory)
    payload = BytesIO()
    pl.DataFrame({"value": [7]}).write_parquet(payload)
    reference = writer.write(
        payload.getvalue(), media_type="application/vnd.apache.parquet", schema_name="test.v1"
    )
    receipt = evidence / "receipts/run=complete/part-0.parquet"
    receipt.parent.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "run": "complete",
                "trial": "case",
                "artifacts": json.dumps({"table": reference.model_dump()}),
            }
        ]
    ).write_parquet(receipt)
    live = evidence / "artifacts/running/case/events/live.ndjson"
    live.parent.mkdir(parents=True)
    missing = {**reference.model_dump(), "path": "datasets/experiments/node/not-yet-collected"}
    live.write_text(
        "".join(
            encode(
                Frame(
                    job="running/case",
                    kind=Kind.sample,
                    offset=i,
                    at=datetime.now(UTC),
                    payload=payload,
                )
            )
            for i, payload in enumerate(
                [
                    {"topic": "started", "trial": "case", "data": {"run": "running"}},
                    {"topic": "artifact", "trial": "case", "data": {"name": "table", **missing}},
                ]
            )
        )
    )
    results = Results(tmp_path)
    assert results.table("test.v1", runs=["complete"])["value"].to_list() == [7]
    assert results.table("test.v1", runs=[]).is_empty()
    with pytest.raises(FileNotFoundError):
        results.table("test.v1")
    (project / reference.path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content changed"):
        results.table("test.v1", runs=["complete"])


@pytest.mark.parametrize(
    ("media_type", "columns", "refusal"),
    [
        ("text/csv", {"value": [7]}, "non-Parquet artifact"),
        ("application/vnd.apache.parquet", {"_trial": ["forged"]}, "reserves the _trial"),
    ],
    ids=["a table that is not Parquet", "a payload claiming the provenance column"],
)
def test_a_table_refuses_an_artifact_it_could_only_misread(
    tmp_path: Path, media_type: str, columns: dict[str, list[int] | list[str]], refusal: str
) -> None:
    """Guessing at another format, or letting a payload overwrite `_trial`, would forge a row."""
    project = tmp_path / "research/example"
    evidence = project / "datasets/experiments/node/evidence"
    payload = BytesIO()
    pl.DataFrame(columns).write_parquet(payload)
    reference = Artifacts(project, evidence / "artifacts/run/case").write(
        payload.getvalue(), media_type=media_type, schema_name="test.v1"
    )
    receipt = evidence / "receipts/run=run/part-0.parquet"
    receipt.parent.mkdir(parents=True)
    pl.DataFrame(
        [{"run": "run", "trial": "case", "artifacts": json.dumps({"t": reference.model_dump()})}]
    ).write_parquet(receipt)
    with pytest.raises(ValueError, match=refusal):
        Results(tmp_path).table("test.v1")


def test_a_join_the_catalog_cannot_bind_before_its_views_exist_still_answers(
    tmp_path: Path,
) -> None:
    """DuckDB cannot name a USING join's tables until their columns exist, so all are built.

    A second project that holds no receipts yet contributes no inventory rather than an error.
    """
    (tmp_path / "research/empty/datasets/experiments").mkdir(parents=True)
    receipt = (
        tmp_path
        / "research/example/datasets/experiments/node/evidence/receipts/run=r/part-0.parquet"
    )
    receipt.parent.mkdir(parents=True)
    pl.DataFrame(
        [{"run": "r", "trial": "case", "artifacts": "{}", "host": "h", "commit": "c"}]
    ).write_parquet(receipt)
    joined = Results(tmp_path).query(
        "SELECT run, trials.host FROM trials JOIN runs USING (project, run)"
    )
    assert joined.to_dicts() == [{"run": "r", "host": "h"}]


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


@pytest.mark.parametrize("source", ["read_parquet('input.parquet')", "jobs", "trials"])
def test_queries_do_not_read_unrelated_corrupt_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.chdir(tmp_path)
    pl.DataFrame({"value": [7]}).write_parquet("input.parquet")
    evidence = tmp_path / "datasets/experiments/node/evidence"
    events = evidence / "artifacts/run/case/events/live.ndjson"
    events.parent.mkdir(parents=True)
    events.write_text("not an event\n", encoding="utf-8")
    receipts = evidence / "receipts/run=run/part-0.parquet"
    receipts.parent.mkdir(parents=True)
    if source == "trials":
        pl.DataFrame({"run": ["run"]}).write_parquet(receipts)
    else:
        receipts.write_bytes(b"not parquet")
    frame = Results(tmp_path).query(
        f"WITH selected AS (SELECT * FROM {source}) SELECT * FROM selected"
    )
    assert frame.height == (0 if source == "jobs" else 1)


def test_events_are_utc_without_extension_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "datasets/experiments/node/evidence/artifacts/run/case/events"
    directory.mkdir(parents=True)
    frame = Frame(
        job="run/case",
        kind=Kind.sample,
        offset=0,
        at=datetime.fromisoformat("2026-09-17T09:00:00+09:00"),
        payload={"topic": "metrics", "trial": "case", "data": {}},
    )
    (directory / "live.ndjson").write_text(encode(frame), encoding="utf-8")
    connect = duckdb.connect

    def offline_connection(*, config: dict[str, bool]) -> duckdb.DuckDBPyConnection:
        assert config["autoinstall_known_extensions"] is False
        return connect(config={**config, "autoload_known_extensions": False})

    monkeypatch.setattr(duckdb, "connect", offline_connection)
    assert Results(tmp_path).query("SELECT recorded_at FROM events").item() == datetime(
        2026, 9, 17, 0, 0
    )


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


@pytest.mark.parametrize("workspace_relative", [False, True])
@pytest.mark.parametrize("from_project", [False, True])
@pytest.mark.parametrize("events", [False, True])
@pytest.mark.parametrize("receipt_alias", [False, True])
def test_transferred_tables_keep_source_context_but_read_collected_bytes(
    tmp_path: Path,
    workspace_relative: bool,
    from_project: bool,
    events: bool,
    receipt_alias: bool,
) -> None:
    project = tmp_path / "research" / "project's results"
    evidence = project / "datasets/experiments/node/evidence"
    directory = evidence / "artifacts/run/case"
    writer = Artifacts(tmp_path if workspace_relative else project, directory)
    buffer = BytesIO()
    pl.DataFrame({"value": [7]}).write_parquet(buffer)
    reference = writer.write(
        buffer.getvalue(), media_type="application/vnd.apache.parquet", schema_name="test.v1"
    )
    source = "C:\\Users\\researcher\\original workspace"
    context = {
        "run": "run",
        "trial": "case",
        "verdict": "known",
        "host": "windows-node",
        "card_name": "GPU",
        "commit": "source",
        "repository": source,
        "params": {"precision": "bf16"},
    }
    receipt = evidence / "receipts/run=run/part-0.parquet"
    receipt.parent.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                **context,
                "params": json.dumps(context["params"]),
                "artifacts": json.dumps(
                    {
                        "events": (directory / "events").relative_to(writer.root).as_posix(),
                        "table": reference.model_dump(),
                        **({"alias": reference.model_dump()} if receipt_alias else {}),
                    }
                ),
            }
        ]
    ).write_parquet(receipt)
    if events:
        stream = directory / "events/live.ndjson"
        stream.parent.mkdir(parents=True)
        payloads = [
            {"topic": "started", "trial": "case", "data": context},
            {
                "topic": "artifact",
                "trial": "case",
                "data": {"name": "table", **reference.model_dump()},
            },
        ]
        stream.write_text(
            "".join(
                encode(
                    Frame(
                        job="run/case",
                        kind=Kind.sample,
                        offset=i,
                        at=datetime.now(UTC),
                        payload=payload,
                    )
                )
                for i, payload in enumerate(payloads)
            ),
            encoding="utf-8",
        )
    results = Results(project if from_project else tmp_path)
    table = results.table("test.v1", project=project.name)
    count = 2 if receipt_alias else 1
    assert table["value"].to_list() == [7] * count
    assert json.loads(table["_trial"][0])["repository"] == source
    assert json.loads(table["_trial"][0])["params"] == context["params"]
    assert "artifacts" not in json.loads(table["_trial"][0])
    artifacts = results.query("SELECT project, root, reference FROM artifacts")
    assert artifacts["project"].to_list() == [project.name] * count
    assert artifacts["root"].to_list() == [str(project)] * count
    assert sorted(results.query("SELECT name FROM artifacts")["name"]) == (
        ["alias", "table"] if receipt_alias else ["table"]
    )
    assert json.loads(artifacts["reference"][0]) == reference.model_dump()
    assert results.table("missing.v1").is_empty()
    (writer.root / reference.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        results.table("test.v1")


def test_receipt_projects_use_local_ownership_and_union_different_schemas(tmp_path: Path) -> None:
    for project in ("one", "two"):
        directory = (
            tmp_path / "research" / project / "datasets/experiments/node/evidence/receipts/run=run"
        )
        directory.mkdir(parents=True)
        pl.DataFrame(
            [
                {
                    "run": project,
                    "trial": "case",
                    "verdict": "known",
                    "artifacts": "{}",
                    "host": "node",
                    "card_name": "GPU",
                    "commit": "source",
                    project: 1,
                }
            ]
        ).write_parquet(directory / "part-0.parquet")
    assert Results(tmp_path).query(
        "SELECT project, one, two FROM trials ORDER BY project"
    ).to_dicts() == [
        {"project": "one", "one": 1, "two": None},
        {"project": "two", "one": None, "two": 1},
    ]


def test_wide_trial_context_does_not_repeat_the_artifact_index(tmp_path: Path) -> None:
    directory = tmp_path / "datasets/experiments/node/evidence/receipts/run=wide"
    directory.mkdir(parents=True)
    references = {
        f"row-{index}": {
            "path": f"datasets/experiments/node/evidence/artifacts/wide/objects/{index}",
            "sha256": "0" * 64,
            "size": 1,
            "media_type": "application/octet-stream",
        }
        for index in range(2048)
    }
    pl.DataFrame(
        [
            {
                "run": "wide",
                "trial": "case",
                "verdict": "known",
                "params": '{"precision":"bf16"}',
                "artifacts": json.dumps(references),
                "host": "GH200",
                "extra_provenance": "preserved",
            }
        ]
    ).write_parquet(directory / "part-0.parquet")
    rows = Results(tmp_path).query("SELECT context FROM artifacts")
    assert rows.height == len(references)
    contexts = [json.loads(value) for value in rows["context"]]
    assert all("artifacts" not in context for context in contexts)
    assert all(context["extra_provenance"] == "preserved" for context in contexts)
    assert max(len(value) for value in rows["context"]) < 1024


def test_old_receipts_keep_missing_fields_null_without_inventing_artifacts(tmp_path: Path) -> None:
    directory = tmp_path / "datasets/experiments/node/evidence/receipts/run=old"
    directory.mkdir(parents=True)
    pl.DataFrame(
        [{"run": "old", "trial": "case", "verdict": "known", "measured": '{"rmse":0.25}'}]
    ).write_parquet(directory / "part-0.parquet")
    results = Results(tmp_path)
    assert results.query(
        "SELECT run, trial, artifacts, host, card_name, commit, measured FROM trials"
    ).to_dicts() == [
        {
            "run": "old",
            "trial": "case",
            "artifacts": None,
            "host": None,
            "card_name": None,
            "commit": None,
            "measured": '{"rmse":0.25}',
        }
    ]
    assert results.query("SELECT run, host, hardware, commit FROM runs").to_dicts() == [
        {"run": "old", "host": None, "hardware": None, "commit": None}
    ]
    assert results.query("SELECT * FROM artifacts").is_empty()
    assert results.query("SELECT * FROM metrics").is_empty()
    assert results.table("missing.v1").is_empty()


@pytest.mark.parametrize("conflicting", [False, True])
def test_overlapping_event_snapshots_only_deduplicate_identical_frames(
    tmp_path: Path, conflicting: bool
) -> None:
    directory = tmp_path / "datasets/experiments/node/evidence/artifacts/run/case/events"
    directory.mkdir(parents=True)
    frame = Frame(
        job="run/case",
        kind=Kind.sample,
        offset=12,
        at=datetime.now(UTC),
        payload={"topic": "metrics", "trial": "case", "data": {"rmse": 0.25}},
    )
    (directory / "live.ndjson").write_text(encode(frame), encoding="utf-8")
    (directory / "00000000000000000012.ndjson.zst").write_bytes(
        zstd.compress(encode(frame).encode())
    )
    collected = (
        frame.model_copy(
            update={"payload": {"topic": "metrics", "trial": "case", "data": {"rmse": 0.5}}}
        )
        if conflicting
        else frame
    )
    (directory / "collected-12-100.ndjson").write_text(encode(collected), encoding="utf-8")
    results = Results(tmp_path)
    if conflicting:
        with pytest.raises(ValueError, match="conflicting event snapshots.*run/case.*offset 12"):
            results.query("SELECT * FROM events")
    else:
        events = results.query("SELECT data FROM events")
        assert events.height == 1
        assert json.loads(events["data"][0]) == {"rmse": 0.25}
