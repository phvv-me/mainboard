import io
import json
import string
import struct
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock

import duckdb
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.render import Stamp
from mainboard.render.query import Query, is_execution_timeout
from mainboard.render.query.sinks import Rendered, Window
from mainboard.render.stamps import ResultHash, Value, canonical

from ..footer import Footer

_PROPERTY = settings(database=None, deadline=None, max_examples=200)
_TEXTS = st.text(max_size=6) | st.sampled_from(["", "NULL", '"NULL"', '"', "\t"])
_SCALARS = st.none() | st.booleans() | st.integers() | st.floats() | _TEXTS
_VALUES = st.recursive(
    _SCALARS,
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=4), inner, max_size=3)
    ),
    max_leaves=8,
)
_CELLS = st.none() | _TEXTS
_QUIET_NAN_BITS = 0x7FF8000000000000


def _nan(payload: int) -> float:
    return struct.unpack(">d", (_QUIET_NAN_BITS | payload).to_bytes(8))[0]


def _reversed_maps(value: Value) -> Value:
    """`value` with the insertion order of every map in it reversed."""
    match value:
        case dict():
            return {key: _reversed_maps(item) for key, item in reversed(value.items())}
        case list():
            return [_reversed_maps(item) for item in value]
    return value


def _stamp(rows: Sequence[Sequence[str]], *, columns: Sequence[str] = ("a", "b")) -> Stamp:
    digest = ResultHash(columns)
    for row in rows:
        digest.add(row)
    return digest.stamp(0)


def _kept(rows: Sequence[Rendered], *, max_rows: int, max_bytes: int) -> Window:
    """A window that has been offered every row in turn."""
    window = Window(max_rows, max_bytes=max_bytes)
    for counted, row in enumerate(rows, start=1):
        window.keep(row, counted)
    return window


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, "NULL", id="null-is-bare"),
        pytest.param("NULL", '"NULL"', id="literal-null-is-quoted"),
        pytest.param("", "", id="empty-text-stays-empty"),
        pytest.param(False, "False", id="false"),
        pytest.param(True, "True", id="true"),
        pytest.param(0, "0", id="zero"),
        pytest.param(-123, "-123", id="negative-integer"),
        pytest.param(9007199254740993, "9007199254740993", id="integer-past-float-precision"),
        pytest.param(1.23456789012345, "1.23456789012", id="twelve-significant-digits"),
        pytest.param(1.00000000000001, "1", id="roundoff-below-twelve-digits-is-hidden"),
        pytest.param(-0.0, "0", id="signed-zero-is-zero"),
        pytest.param(float("inf"), "Infinity", id="infinity"),
        pytest.param(_nan(1), "NaN", id="nan-with-one-payload"),
        pytest.param(_nan(2), "NaN", id="nan-with-another-payload"),
        pytest.param([True, 1, None], "[true,1,null]", id="list-is-compact-json"),
        pytest.param({"b": 2, "a": 1}, '{"a":1,"b":2}', id="map-keys-are-sorted"),
    ],
)
def test_canonical_cells_preserve_scalar_and_nested_text(value, expected: str) -> None:
    assert canonical(value) == expected


@_PROPERTY
@given(value=_VALUES, text=_TEXTS)
def test_a_canonical_cell_is_one_field_that_ignores_map_order_and_decodes_text(
    value, text
) -> None:
    cell, quoted = canonical(value), canonical(text)
    assert not {"\t", "\n", "\r"} & set(cell + quoted)
    assert canonical(_reversed_maps(value)) == cell
    assert (json.loads(quoted) if quoted.startswith('"') else quoted) == text
    assert quoted != canonical(None)


@_PROPERTY
@given(
    rows=st.lists(st.tuples(_CELLS, _CELLS), min_size=1, max_size=8),
    other=_CELLS,
    data=st.data(),
)
def test_the_result_hash_is_the_multiset_of_the_rendered_rows(rows, other, data) -> None:
    assume(other != rows[0][0])
    cells = [[canonical(cell) for cell in row] for row in rows]
    changed = [[canonical(other), cells[0][1]], *cells[1:]]
    assert _stamp(data.draw(st.permutations(cells))) == _stamp(cells)
    assert _stamp([*cells, cells[0]]).sha256 != _stamp(cells).sha256
    assert _stamp(cells, columns=("b", "a")).sha256 != _stamp(cells).sha256
    assert _stamp(changed).sha256 != _stamp(cells).sha256


@settings(max_examples=300)
@given(
    texts=st.lists(st.text(string.ascii_lowercase + "日", max_size=24), max_size=40),
    max_rows=st.integers(0, 12),
    max_bytes=st.integers(0, 150),
)
def test_a_window_keeps_an_exact_prefix_and_suffix_within_both_caps(
    texts: Sequence[str], max_rows: int, max_bytes: int
) -> None:
    rows = [Rendered(cells=(text,), text=text) for text in texts]
    window = _kept(rows, max_rows=max_rows, max_bytes=max_bytes)
    tail = list(window.tail)
    shown = [*window.head, *tail]
    marker = window.marker(Stamp(rows=len(rows), sha256="0" * 64, snapshot_id=0))
    assert window.head == rows[: len(window.head)] and tail == rows[len(rows) - len(tail) :]
    assert len(shown) <= len(rows) and bool(window.limits) == (len(shown) < len(rows))
    assert not max_rows or len(shown) <= max_rows
    assert not max_bytes or sum(len(row.text.encode()) + 1 for row in shown) <= max_bytes
    assert not window.limits or marker.startswith(f"… {len(rows) - len(shown)} rows omitted")


@settings(max_examples=40)
@given(
    description=st.sampled_from([None, []]),
    mode=st.sampled_from([None, "json", "human"]),
    max_rows=st.integers(-1, 5),
    max_bytes=st.integers(-1, 5),
)
def test_a_statement_without_result_columns_reports_success(
    description: list[tuple[str, ...]] | None, mode: str | None, max_rows: int, max_bytes: int
) -> None:
    out, err = io.StringIO(), io.StringIO()
    cursor = Mock(spec=duckdb.DuckDBPyConnection, description=description)
    with redirect_stdout(out), redirect_stderr(err):
        query = Query(cursor, snapshot_id=0)
        stamp = query.render(mode=mode, max_rows=max_rows, max_bytes=max_bytes)
    status = err.getvalue() if mode == "json" else out.getvalue()
    assert status.startswith("ok\n") and "displayed=0; omitted=0" in status and stamp.complete
    assert Footer.of(status) == Footer(rows=0, sha256=stamp.sha256, snapshot_id=0)
    assert (json.loads(out.getvalue()) if mode == "json" else []) == []


@pytest.mark.parametrize(
    ("fault", "timed_out"),
    [
        pytest.param(
            duckdb.IOException("Timeout was reached error for HTTP POST"), True, id="quack-http"
        ),
        pytest.param(duckdb.IOException("remote query interrupted"), True, id="quack-interrupt"),
        pytest.param(duckdb.IOException("maximum execution time"), True, id="server-limit"),
        pytest.param(duckdb.InterruptException("stopped"), True, id="local-interrupt"),
        pytest.param(duckdb.CatalogException("table not found"), False, id="another-fault"),
    ],
)
def test_forwarded_timeout_errors_are_recognized(fault: duckdb.Error, timed_out: bool) -> None:
    assert is_execution_timeout(fault) is timed_out


def test_a_timeout_that_aborted_the_transaction_refuses_without_binding_the_query_again() -> None:
    connection = Mock(spec=duckdb.DuckDBPyConnection, description=None)
    connection.execute.side_effect = duckdb.InterruptException(
        "Query exceeded maximum execution time"
    )
    connection.sql.side_effect = duckdb.TransactionException(
        "Current transaction is aborted (please ROLLBACK)"
    )

    with pytest.raises(MissionError, match="query timed out"):
        Query.started(connection, "SELECT 1", 0, timeout=0.02)

    assert connection.sql.call_count == 1
