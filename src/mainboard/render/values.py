import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

type Cell = str | int | float | bool | None
type Row = Mapping[str, Cell]
type Node = Cell | Mapping[str, Node] | list[Node] | tuple[Node, ...]


def to_row(record: Mapping[str, Node]) -> dict[str, Cell]:
    """One flat row from `record`, a nested value folded into a compact JSON text cell.

    record: a shallow-or-nested mapping, typically a model's `model_dump()`.
    """
    return {
        key: json.dumps(value) if isinstance(value, Mapping | list | tuple) else value
        for key, value in record.items()
    }


def columns_of(rows: Sequence[Row], fields: Sequence[str] | None) -> list[str]:
    """The column order for `rows`, `fields` verbatim when given else the first row's keys."""
    if fields is not None:
        return list(fields)
    return list(rows[0].keys()) if rows else []


def totals(
    rows: Sequence[Mapping[str, Node]],
    *,
    columns: Sequence[str],
    summing: Sequence[str],
    label: str = "total",
) -> dict[str, Cell]:
    """One closing row adding up `summing`'s columns, for a table a reader has to budget from.

    The row keeps the table's own shape, so it renders, projects and encodes exactly like every
    other row rather than needing a second pass to print.

    rows: the rows being added up.
    columns: the table's column order, the first of which carries `label`.
    summing: the columns to add; every other one comes back empty.
    label: what the first column says on the closing row.
    """
    return {
        column: label
        if at == 0
        else (
            sum(
                cell if isinstance(cell := row.get(column), int | float) else 0
                for row in rows
                if not isinstance(row.get(column), bool)
            )
            if column in summing
            else ""
        )
        for at, column in enumerate(columns)
    }


def pairs_of(row: Row, *, fields: Sequence[str] | None) -> list[dict[str, Cell]]:
    """`row`'s items as one field/value row per key, narrow and legible for a wide entity.

    A single entity (host facts, an execution plan) routinely holds more fields than a
    terminal is wide, so it renders down its own rows rather than across columns.

    fields: the field names to keep, every key when None.
    """
    return [
        {"field": key, "value": value}
        for key, value in row.items()
        if fields is None or key in fields
    ]
