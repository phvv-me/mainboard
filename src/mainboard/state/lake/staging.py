"""Statements a lake is sent: rows staged as one NDJSON file DuckDB reads itself, a built
statement compiled to DuckDB SQL, and the literals a served lake needs in place of parameters."""

import base64
import json
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

import duckdb
from sqlalchemy import JSON, ClauseElement, Column, DateTime, LargeBinary, Table

from .. import schema
from ..constants import ALIAS
from ..schema import kind
from ..storage.database import quoted
from ..storage.records import ndjson


def compiled(statement: ClauseElement) -> tuple[str, list[Any]]:
    """`statement` as DuckDB SQL, and the values its `?` placeholders bind in order."""
    done = schema.dialect.statement_compiler(
        schema.dialect, statement, compile_kwargs={"render_postcompile": True}
    )
    bound = done.construct_params()
    return done.string, [bound[name] for name in done.positiontup or ()]


def inlined(sql: str, parameters: Sequence[object]) -> str:
    """`sql` with each `?` outside a string literal replaced by its parameter as a literal.

    What a served lake is sent, since a Quack attach drops bound parameters. Splitting on quotes
    leaves literals at the odd pieces, an escaped `''` included.
    """
    pieces = sql.split("'")
    outside = range(0, len(pieces), 2)
    marks = sum(pieces[index].count("?") for index in outside)
    if marks != len(parameters):
        raise ValueError(f"{len(parameters)} parameters for {marks} placeholders in {sql!r}")
    values = iter(parameters)
    for index in outside:
        head, *rest = pieces[index].split("?")
        pieces[index] = head + "".join(_literal(next(values)) + tail for tail in rest)
    return "'".join(pieces)


def insert(
    connection: duckdb.DuckDBPyConnection, table: Table, rows: Iterable[Mapping[str, object]]
) -> int:
    """Append `rows` to the attached lake's `table` in one statement, returning how many.

    Each row maps column names to values; a column a row leaves out is NULL. Timestamps may be
    ISO strings or datetimes, JSON columns documents or Python values, BLOB columns bytes. The
    batch travels as one NDJSON file DuckDB reads itself, never row by row.
    """
    listed = list(rows)
    if not listed:
        return 0
    with staged(table, listed) as select:
        connection.execute(f"INSERT INTO {ALIAS}.{table.name} BY NAME {select}")
    return len(listed)


@contextmanager
def staged(table: Table, rows: Iterable[Mapping[str, object]]) -> Generator[str]:
    """Stage `rows` for one statement as a SELECT typed like `table`'s columns (see `insert`).

    The rows wait in a temporary NDJSON file while the block runs, which the SELECT reads.
    """
    columns = ", ".join(_decoded(column) for column in table.columns)
    cells = (
        {column.name: _cell(column, row.get(column.name)) for column in table.columns}
        for row in rows
    )
    with ndjson(cells) as source:
        yield f"SELECT {columns} FROM {source}"


def _cell(column: Column, value: object) -> object:
    """`value` as one NDJSON field of `column`, None kept as SQL NULL.

    A JSON column takes a string as already JSON and embeds anything else as it is; bytes travel
    as base64 and a timestamp as its text, both of which the insert decodes.
    """
    match column.type, value:
        case _, None:
            return None
        case JSON(), str():
            return json.loads(value)
        case LargeBinary(), bytes():
            return base64.b64encode(value).decode("ascii")
        case DateTime(), _:
            return str(value)
    return value


def _decoded(column: Column) -> str:
    """The SQL reading `column` back out of a staged NDJSON object."""
    name = column.name
    match column.type:
        case JSON():
            return f"json->'{name}' AS {name}"
        case LargeBinary():
            return f"from_base64(json->>'{name}') AS {name}"
    return f"CAST(json->>'{name}' AS {kind(column)}) AS {name}"


def _literal(value: object) -> str:
    """`value` as a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool | int | float):
        return str(value).lower() if isinstance(value, bool) else repr(value)
    if isinstance(value, date):
        typed = "TIMESTAMPTZ" if isinstance(value, datetime) else "DATE"
        return f"{typed} {quoted(value.isoformat())}"
    return quoted(str(value))
