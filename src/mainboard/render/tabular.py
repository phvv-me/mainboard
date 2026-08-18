import json
from typing import TYPE_CHECKING

from .values import columns_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .values import Cell, Row


def encode(rows: Sequence[Row], *, fields: Sequence[str] | None = None) -> str:
    """Columns-once text, a header line of field names then one tab-separated line per row.

    A cell holding a tab, a newline, or a leading quote is JSON-encoded, the leading-quote
    case closing the one ambiguity that would otherwise collide with an encoded cell on
    `decode`; every other cell keeps its plain string form.

    rows: the records to encode, each a flat field-name to value mapping.
    fields: the column order and projection, every key from the first row when None.
    """
    columns = columns_of(rows, fields)
    lines = ["\t".join(columns)]
    lines.extend("\t".join(_cell(row.get(column)) for column in columns) for row in rows)
    return "\n".join(lines)


def decode(text: str) -> list[dict[str, str]]:
    """The rows `encode` produced, its header names mapped back onto each row's cells."""
    header, *body = text.split("\n")
    columns = header.split("\t") if header else []
    return [
        dict(
            zip(
                columns,
                (json.loads(cell) if cell.startswith('"') else cell for cell in line.split("\t")),
                strict=True,
            )
        )
        for line in body
    ]


def _cell(value: Cell) -> str:
    text = "" if value is None else str(value)
    return json.dumps(text) if text.startswith('"') or "\t" in text or "\n" in text else text
