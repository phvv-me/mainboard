"""The receipt store, artifact archives, row logs and plots, read and written through DuckDB."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from mainboard.experiments.rows import RowLog
from mainboard.plots.columns import Columns
from mainboard.plots.table import Plot
from mainboard.state.relations import Relations, parquet_bytes, records
from mainboard.trials.archive import ParquetArtifacts
from mainboard.trials.artifacts import Artifact
from mainboard.trials.coverage import Cell, Probed
from mainboard.trials.dataset import Ambiguous, Dataset
from mainboard.trials.provenance import Admissibility
from mainboard.trials.vocabulary import Outcome

OLDER, NEWER = "20260101T000000Z-a", "20260102T000000Z-b"
HERE = Cell(values={"card": "h100"}, probing={"card": Probed.FOUND})


def _receipt(key: str, ms: int, *, passed: bool = True, admissible: bool = True) -> dict:
    return {
        "lane": "speed",
        "key": key,
        "card": "h100",
        "card_probed": str(Probed.FOUND),
        "outcome": Outcome.PASSED if passed else Outcome.FAILED,
        "admissibility": Admissibility.ADMISSIBLE if admissible else Admissibility.DIRTY,
        "measured": {"ms": ms},
    }


def _measured(rows: list[dict]) -> list[tuple[str, int]]:
    return [(row["key"], json.loads(row["measured"])["ms"]) for row in rows]


@pytest.fixture
def store(tmp_path: Path) -> Dataset:
    """Two runs: the newer re-measures `a`, fails `b` and writes `c` from a dirty tree."""
    dataset = Dataset(tmp_path / "receipts", axes=("card",))
    older = dataset.writer(OLDER, {"opened_at_ns": 1})
    older.write(_receipt("a", 3))
    older.write(_receipt("b", 4))
    newer = dataset.writer(NEWER, {"opened_at_ns": 2})
    newer.write(_receipt("a", 2))
    newer.write(_receipt("b", 9, passed=False))
    newer.write(_receipt("c", 1, admissible=False))
    return dataset


def test_runs_order_by_their_creation_coordinate(store: Dataset) -> None:
    assert store.runs == (OLDER, NEWER)
    assert store.newest == NEWER
    assert store.stored == {OLDER, NEWER}
    assert store.lanes() == {"speed"}


def test_passing_takes_each_cell_from_its_newest_admissible_run(store: Dataset) -> None:
    assert _measured(records(store.passing())) == [("a", 2), ("b", 4)]
    assert _measured(records(store.passing(every=True))) == [("a", 3), ("a", 2), ("b", 4)]


def test_status_counts_admissible_passes_at_the_cell(store: Dataset) -> None:
    status = store.status("speed", ["a", "b", "c"], HERE)
    assert (status.want, status.have, status.missing, status.run) == (3, 2, ("c",), NEWER)
    elsewhere = Cell(values={"card": "a100"}, probing={"card": Probed.FOUND})
    assert store.status("speed", ["a"], elsewhere).have == 0


def test_rows_read_the_newest_run_in_the_order_written(store: Dataset) -> None:
    rows = store.rows()
    assert [row["key"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["measured"] == {"ms": 2}
    assert [row["key"] for row in store.rows(OLDER)] == ["a", "b"]


def test_a_compacted_run_reads_the_same(store: Dataset) -> None:
    before = records(store.scan())
    store.writer(NEWER, {}).compact()
    assert len(list((store.root / f"run={NEWER}").glob("part-*.parquet"))) == 1
    assert records(store.scan()) == before


def test_an_unwritten_store_reads_empty(tmp_path: Path) -> None:
    empty = Dataset(tmp_path / "none", axes=("card",))
    assert empty.runs == ()
    assert records(empty.passing()) == []
    assert empty.status("speed", ["a"], HERE).have == 0


def test_runs_opened_at_one_instant_are_refused(tmp_path: Path) -> None:
    tied = Dataset(tmp_path / "tied")
    for run in ("x", "y"):
        tied.writer(run, {"opened_at_ns": 5}).write(_receipt("a", 1))
    with pytest.raises(Ambiguous):
        _ = tied.runs


def test_tables_read_every_matching_artifact_with_its_provenance(tmp_path: Path) -> None:
    data = parquet_bytes(Relations().rows([{"x": 1}, {"x": 2}]))
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / digest).write_bytes(data)
    reference = Artifact(
        path=f"objects/{digest}",
        sha256=digest,
        size=len(data),
        media_type="application/vnd.apache.parquet",
        schema_name="demo",
    )
    dataset = Dataset(tmp_path / "receipts")
    receipt = {**_receipt("a", 1), "artifacts": {"t": reference.model_dump()}}
    dataset.writer(OLDER, {"opened_at_ns": 1}).write(receipt)
    read = records(dataset.tables(tmp_path, schema_name="demo"))
    assert [row["x"] for row in read] == [1, 2]
    assert json.loads(read[0]["_trial"])["artifact_name"] == "t"
    none = dataset.tables(tmp_path, schema_name="other")
    assert none.columns == ["_trial"] and none.fetchall() == []


def test_rows_round_trip_through_parquet() -> None:
    rows = [{"n": 1, "name": "a"}, {"n": 2.5, "nested": [1, 2]}]
    read = records(Relations().parquet(parquet_bytes(Relations().rows(rows))))
    assert read == [
        {"n": 1.0, "name": "a", "nested": None},
        {"n": 2.5, "name": None, "nested": [1, 2]},
    ]


def test_archives_keep_exact_bytes_under_every_path(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 600
    for name in ("one.bin", "two.bin"):
        (tmp_path / name).write_bytes(payload)
    ParquetArtifacts(tmp_path).pack([tmp_path / "one.bin", tmp_path / "two.bin"])
    digest = hashlib.sha256(payload).hexdigest()
    for name in ("one.bin", "two.bin"):
        (tmp_path / name).unlink()
        assert ParquetArtifacts.read(tmp_path / name, boundary=tmp_path, digest=digest) == payload


def test_a_row_log_resumes_from_every_part(tmp_path: Path) -> None:
    log = RowLog(tmp_path / "rows.parquet", id_fields=("model",))
    assert log.extend([{"model": "a", "score": 1.0}, {"model": "b", "score": 2.0}]) == 2
    log.flush()
    again = RowLog(tmp_path / "rows.parquet", id_fields=("model",))
    assert again.has(model="a") and not again.append({"model": "b", "score": 3.0})
    assert sorted(row["model"] for row in again.rows) == ["a", "b"]


def test_a_plot_draws_a_relation(tmp_path: Path) -> None:
    relation = Relations().kept(
        "SELECT i AS x, (i / 4)::DECIMAL(4, 2) AS y, 'run' AS hue FROM range(4) t(i)"
    )
    columns = Columns.of(relation)
    assert columns.numbers == {"x", "y"} and columns.plain()["y"][1] == 0.25
    assert columns["y"][1] == Decimal("0.25")
    [saved] = Plot(columns).save(tmp_path / "plot.png", x="x", y="y", kind="line")
    assert saved.stat().st_size > 0
    with pytest.raises(ValueError, match="numeric"):
        Plot(columns).save(tmp_path / "bad.png", x="x", y="hue")
