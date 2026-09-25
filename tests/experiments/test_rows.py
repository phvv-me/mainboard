from pathlib import Path

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from mainboard.experiments import ExperimentPaths, RowLog


def test_an_experiment_lays_out_raw_results_and_plots_under_its_name(tmp_path: Path) -> None:
    paths = ExperimentPaths(name="shootout", root=tmp_path)
    assert paths.raw_dir == tmp_path / "shootout" / "raw"
    assert paths.results_dir.is_dir()
    assert paths.plots_dir.is_dir()
    assert paths.plot("curve.png") == tmp_path / "shootout" / "plots" / "curve.png"
    assert paths.table("calls.csv").name == "calls.csv"
    assert paths.table("Qwen3-1.7B") == paths.raw_dir / "Qwen3-1.7B.parquet"
    assert (
        paths.device_table("RTX_4090_CC8.9", "calls")
        == paths.raw_dir / "RTX_4090_CC8.9" / "calls.parquet"
    )


def test_the_layout_derives_from_the_package_file(tmp_path: Path) -> None:
    package = tmp_path / "experiments" / "shootout"
    package.mkdir(parents=True)
    paths = ExperimentPaths.from_file(package / "__init__.py")
    assert (paths.name, paths.root) == ("shootout", tmp_path / "experiments")


@given(st.lists(st.integers(0, 5), min_size=1, max_size=20))
def test_a_row_is_taken_once_however_often_its_identity_is_appended(cells: list[int]) -> None:
    log = RowLog(Path("unused/calls"), id_fields=("cell",))
    taken = [log.append({"cell": cell, "value": index}) for index, cell in enumerate(cells)]
    assert sum(taken) == len(set(cells)) == len(log)


def test_a_second_writer_resumes_from_every_part_and_never_rewrites_another(
    tmp_path: Path,
) -> None:
    first = RowLog(tmp_path / "calls", id_fields=("model", "block"))
    first.append({"model": "gpt2", "block": 0, "ms": 1.0})
    first.flush()
    first.part.rename(tmp_path / "calls" / "part-otherhost-1.parquet")
    second = RowLog(tmp_path / "calls", id_fields=("model", "block"))
    assert second.has(model="gpt2", block=0)
    assert not second.append({"model": "gpt2", "block": 0, "ms": 2.0})
    assert second.append({"model": "gpt2", "block": 1, "ms": 3.0})
    second.flush()
    parts = sorted(p.name for p in (tmp_path / "calls").glob("part-*.parquet"))
    assert len(parts) == 2
    pooled = pl.concat([pl.read_parquet(p) for p in (tmp_path / "calls").glob("part-*.parquet")])
    assert sorted(pooled["block"].to_list()) == [0, 1]


def test_flushing_nothing_writes_nothing(tmp_path: Path) -> None:
    RowLog(tmp_path / "calls", id_fields=("cell",)).flush()
    assert not (tmp_path / "calls").exists()


def test_a_table_suffix_names_the_same_directory(tmp_path: Path) -> None:
    assert RowLog(tmp_path / "calls.parquet", id_fields=("a",)).dir == tmp_path / "calls"


def test_extend_counts_only_the_new_rows_and_drop_where_forgets_their_keys(tmp_path: Path) -> None:
    log = RowLog(tmp_path / "calls", id_fields=("model",))
    assert (
        log.extend([{"model": "a", "v": 1}, {"model": "b", "v": 2}, {"model": "a", "v": 3}]) == 2
    )
    log.drop_where(lambda row: row["model"] == "a")
    assert [row["model"] for row in log.rows] == ["b"]
    assert log.append({"model": "a", "v": 4}) is True
