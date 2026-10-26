import hashlib
from collections.abc import Sequence

import duckdb
import pytest
from patos import FrozenModel

from mainboard.state import Lake
from mainboard.state.storage import blobs as blobs_module
from mainboard.state.storage.blobs import Blobs

_VECTOR_ROWS = 2048
_WITHOUT_BYTES_OPTION = "ALTER TABLE lake.blobs RESET (parquet_row_group_size_bytes)"
_INSERTS = (
    "SELECT message FROM duckdb_logs "
    "WHERE type = 'QueryLog' AND message LIKE 'INSERT INTO lake.blobs%'"
)
_GROUPS = "SELECT row_group_num_rows FROM parquet_metadata(?) WHERE column_id = 0"


class Staged(FrozenModel):
    """What staging objects left in the lake: the inserts sent, the files and row groups kept."""

    inserts: list[str]
    files: int
    groups: list[int]
    read: dict[str, bytes]


def _objects(sizes: Sequence[int]) -> dict[str, bytes]:
    """One object per size, each different from the others and without a pattern to compress."""
    payloads = [
        hashlib.shake_256(number.to_bytes(8)).digest(size) for number, size in enumerate(sizes)
    ]
    return {hashlib.sha256(payload).hexdigest(): payload for payload in payloads}


def _configured(lake: Lake, statements: Sequence[str]) -> None:
    with lake.open(write=True) as connection:
        for statement in statements:
            connection.execute(statement)


def _stage(lake: Lake, objects: dict[str, bytes]) -> Staged:
    """Stage `objects` in one transaction, watching the SQL it sends, then read them back."""
    blobs = Blobs(lake)

    def work(connection: duckdb.DuckDBPyConnection) -> list[str]:
        connection.execute("CALL enable_logging('QueryLog')")
        blobs.stage(connection, objects)
        return [message for (message,) in connection.execute(_INSERTS).fetchall()]

    inserts = lake.transact(work)
    files = [str(path) for path in lake.data.rglob("*.parquet") if "blobs" in path.parts]
    with lake.open() as connection:
        groups = connection.execute(_GROUPS, [files]).fetchall()
        read = blobs.read(connection, list(objects))
    return Staged(
        inserts=inserts, files=len(files), groups=[rows for (rows,) in groups], read=read
    )


@pytest.fixture
def chunk_bytes(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    """Chunks of the requested size, with a row group that holds four of them."""
    monkeypatch.setattr(blobs_module, "_CHUNK_BYTES", request.param)
    monkeypatch.setattr(blobs_module, "_ROW_GROUP_BYTES", 4 * request.param)
    return request.param


@pytest.mark.parametrize(
    ("chunk_bytes", "row_count", "statements"),
    [
        pytest.param(1000, 20, [], id="thousand-byte-chunks"),
        pytest.param(256000, 20, [], id="quarter-megabyte-chunks"),
        pytest.param(
            1000,
            4300,
            [_WITHOUT_BYTES_OPTION],
            id="a-lake-from-before-the-bytes-option",
        ),
    ],
    indirect=["chunk_bytes"],
)
def test_blob_row_groups_obey_the_payload_byte_budget(
    lake: Lake, chunk_bytes: int, row_count: int, statements: list[str]
) -> None:
    objects = _objects([chunk_bytes * row_count])
    _configured(lake, statements)
    staged = _stage(lake, objects)
    assert staged.files == 1 and len(staged.inserts) == 1
    assert len(staged.groups) > 1 and sum(staged.groups) == row_count
    assert max(staged.groups) <= 4 and staged.read == objects


@pytest.mark.parametrize(
    ("sizes", "unions"),
    [
        pytest.param([64] * 10000, 0, id="many-small-objects-share-one-scan"),
        pytest.param(
            [8 << 20, *[64] * 10000, 8 << 20], 1, id="a-mixed-batch-bounds-union-arms-by-bytes"
        ),
    ],
)
def test_staged_objects_land_in_one_insert_of_bounded_row_groups(
    lake: Lake, sizes: list[int], unions: int
) -> None:
    objects = _objects(sizes)
    staged = _stage(lake, objects)
    assert len(staged.inserts) == 1 and staged.inserts[0].count("UNION ALL") == unions
    assert len(staged.groups) > 1 and sum(staged.groups) == len(objects)
    assert max(staged.groups) <= _VECTOR_ROWS and staged.read == objects
