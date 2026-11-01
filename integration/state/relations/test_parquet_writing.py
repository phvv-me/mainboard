from pathlib import Path

from mainboard.state import Relations, records, write_parquet


def test_a_relation_is_written_as_parquet_under_a_path_holding_quotes(tmp_path: Path) -> None:
    target = tmp_path / "it's ''quoted''" / "rows.parquet"
    target.parent.mkdir()

    write_parquet(Relations().rows([{"x": 1}, {"x": 2}]), target)

    assert records(Relations().parquet(target.read_bytes())) == [{"x": 1}, {"x": 2}]
