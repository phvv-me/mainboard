from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard.render.values import columns_of, to_row

if TYPE_CHECKING:
    from mainboard.render.values import Cell, Node, Row


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gold", "gold"),
        (8, 8),
        (True, True),
        (None, None),
        ({"limit_bytes": 100, "capped": True}, '{"limit_bytes": 100, "capped": true}'),
        ([{"name": "a100"}, {"name": "h100"}], '[{"name": "a100"}, {"name": "h100"}]'),
        (("ib0", "ib1"), '["ib0", "ib1"]'),
    ],
)
def test_a_row_keeps_its_scalars_and_folds_a_nested_value_into_one_json_cell(
    value: Node, expected: Cell
) -> None:
    """A cell is one terminal column, so anything with structure arrives as compact JSON."""
    assert to_row({"field": value}) == {"field": expected}


@pytest.mark.parametrize(
    ("rows", "fields", "expected"),
    [
        ([{"a": 1, "b": 2}], ["b"], ["b"]),
        ([{"a": 1}, {"a": 2, "b": 3}], None, ["a"]),
        ([], None, []),
    ],
)
def test_the_column_order_is_the_given_fields_or_the_first_rows_keys(
    rows: Sequence[Row], fields: Sequence[str] | None, expected: list[str]
) -> None:
    """Asking for columns is a projection, and asking for none reads them off the data."""
    assert columns_of(rows, fields) == expected
