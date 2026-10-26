"""Large blob inserts split row groups even below DuckDB's vector size."""

import hashlib
import os

import pytest

from mainboard.state import Lake
from mainboard.state.lake import maintenance as lake_module
from mainboard.state.storage import blobs as blobs_module
from mainboard.state.storage.blobs import Blobs, _Batch
from mainboard.state.storage.database import connect


@pytest.mark.parametrize(
    ("chunk_bytes", "row_count", "legacy"),
    [(1000, 20, False), (256000, 20, False), (1000, 4300, True)],
)
def test_blob_row_groups_obey_the_payload_byte_budget(
    workspace, monkeypatch, chunk_bytes: int, row_count: int, legacy: bool
) -> None:
    monkeypatch.setattr(blobs_module, "_CHUNK_BYTES", chunk_bytes)
    monkeypatch.setattr(blobs_module, "_ROW_GROUP_BYTES", 4 * chunk_bytes)
    if legacy:
        monkeypatch.setattr(
            lake_module,
            "_BYTES_OPTIONS",
            tuple(
                option
                for option in lake_module._BYTES_OPTIONS
                if option[0] != "parquet_row_group_size_bytes"
            ),
        )
    payload = os.urandom(chunk_bytes * row_count)
    digest = hashlib.sha256(payload).hexdigest()
    lake = Lake.at(workspace).ready()
    with lake.open(write=True) as connection:
        connection.execute(
            "CALL lake.set_option('parquet_row_group_size', 4, table_name => 'blobs')"
        )
    blobs = Blobs(lake)
    lake.transact(lambda connection: blobs.stage(connection, {digest: payload}))
    files = [str(path) for path in lake.data.rglob("*.parquet") if "blobs" in path.parts]
    assert len(files) == 1
    with lake.open() as connection:
        groups = connection.execute(
            "SELECT row_group_num_rows FROM parquet_metadata(?) WHERE column_id = 0", [files]
        ).fetchall()
        assert len(groups) > 1 and sum(rows for (rows,) in groups) == row_count
        assert max(rows for (rows,) in groups) <= 4
        assert blobs.read(connection, [digest]) == {digest: payload}


def test_many_small_objects_use_one_scan_and_multiple_bounded_groups(
    workspace, monkeypatch
) -> None:
    sources: list[str] = []
    original = _Batch._source

    def recorded(batch: _Batch) -> str:
        source = original(batch)
        sources.append(source)
        return source

    monkeypatch.setattr(_Batch, "_source", recorded)
    objects = {}
    for number in range(10000):
        payload = number.to_bytes(8) + b"x" * 56
        objects[hashlib.sha256(payload).hexdigest()] = payload
    lake = Lake.at(workspace).ready()
    blobs = Blobs(lake)
    lake.transact(lambda connection: blobs.stage(connection, objects))
    assert len(sources) == 1 and "UNION ALL" not in sources[0]
    files = [str(path) for path in lake.data.rglob("*.parquet") if "blobs" in path.parts]
    with lake.open() as connection:
        groups = connection.execute(
            "SELECT row_group_num_rows FROM parquet_metadata(?) WHERE column_id = 0", [files]
        ).fetchall()
        assert len(groups) > 1 and sum(rows for (rows,) in groups) == len(objects)
        assert max(rows for (rows,) in groups) <= 2048
        assert blobs.read(connection, objects) == objects


def test_mixed_batch_bounds_union_arms_by_bytes_not_the_largest_chunks_row_limit() -> None:
    # A few large chunks must not turn thousands of tiny chunks into thousands of scans.

    with connect() as connection:
        batch = _Batch(connection)
        for payload in (b"x" * (8 << 20), *(b"x" * 64 for _ in range(10000)), b"y" * (8 << 20)):
            batch.keep(hashlib.sha256(payload).hexdigest(), payload)
        assert batch._source().count("UNION ALL") <= 2
