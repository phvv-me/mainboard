"""What DuckDB makes of a SQL text, and the records staged for a SQL text to read.

`names` asks DuckDB's own parser, the one that runs the text later, so the answer never disagrees
with execution (`FROM runs r` names `runs`, a CTE names nothing). `ndjson` stages records as one
NDJSON file that DuckDB reads itself.
"""

import json
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from pydantic import JsonValue

from .database import bare, quoted, scratch

# The parser's connection, and every table function DuckDB itself ships, so the rest of the calls
# a text makes are the table macros a caller defines.
_PARSER = bare()
_SHIPPED = frozenset(
    name
    for (name,) in _PARSER.execute(
        "SELECT function_name FROM duckdb_functions() "
        "WHERE function_type IN ('table', 'table_macro')"
    ).fetchall()
)
# DuckDB's own largest JSON object by default, the floor of a staged file's object limit. A
# reader buffers twice its limit, so the limit follows the longest staged line: a fixed 2 GiB,
# set so a 25 MB source file could be sealed (2026-09-29), reserved 4 GiB for every append and
# ran the 7 GB macOS CI runner out of memory (2026-10-07).
_JSON_OBJECT_BYTES = 16 << 20


def names(sql: str) -> tuple[frozenset[str], frozenset[str]]:
    """The tables `sql` reads and the table macros it calls.

    A table is dotted when qualified (`lake.runs`); the text's own common table expressions are not
    tables. Raises ValueError when DuckDB cannot serialize `sql`: a syntax error, or a statement
    that is not a SELECT (EXPLAIN, COPY, PRAGMA).
    """
    with _PARSER.cursor() as cursor:
        (text,) = cursor.execute("SELECT json_serialize_sql(?)", [sql]).fetchall()[0]
    tree = json.loads(text)
    if tree["error"]:
        raise ValueError(tree["error_message"])
    named = [pair for node in _nodes(tree["statements"]) for pair in _named(node)]
    kinds = {
        kind: {name for each, name in named if each == kind} for kind in ("table", "call", "cte")
    }
    return frozenset(kinds["table"] - kinds["cte"]), frozenset(kinds["call"] - _SHIPPED)


@contextmanager
def ndjson(
    records: Iterable[Mapping[str, object] | str],
    *,
    typed: bool = False,
    columns: Mapping[str, str] | None = None,
) -> Generator[str]:
    """`records` as a temporary NDJSON file, answered as the call reading it.

    Records are mappings or lines already JSON, and the object limit is the longest line's. Each
    record is one `json` object, or with `typed` the columns `read_json` infers from every record,
    a key one lacks read as null, or with `columns` exactly those, each name to its DuckDB type.

    DuckDB reads a file like this in a fraction of a second, while this build binds a list
    parameter at two milliseconds an element: twenty thousand rows took forty seconds.
    """
    with scratch(".ndjson") as staging:
        longest = _written(records, staging)
        path, limit = quoted(staging.as_posix()), max(_JSON_OBJECT_BYTES, longest + 1)
        kinds = ", ".join(
            f"{quoted(column)}: {quoted(kind)}" for column, kind in (columns or {}).items()
        )
        pinned = f", columns = {{{kinds}}}" if columns else ""
        yield (
            f"read_json({path}, format = 'newline_delimited', sample_size = -1, "
            f"maximum_object_size = {limit}{pinned})"
            if typed
            else f"read_ndjson_objects({path}, maximum_object_size = {limit})"
        )


def _named(node: Mapping[str, JsonValue]) -> list[tuple[str, str]]:
    """What one serialized node names, as `table`, `call` or `cte` entries.

    A `table` is a base table, a `call` a table function, and `cte` a name a WITH clause binds.
    """
    match node:
        case {
            "type": "BASE_TABLE",
            "catalog_name": catalog,
            "schema_name": schema,
            "table_name": table,
        }:
            return [("table", ".".join(str(part) for part in (catalog, schema, table) if part))]
        case {"type": "TABLE_FUNCTION", "function": {"function_name": str(function)}}:
            return [("call", function)]
        case {"cte_map": {"map": list(entries)}}:
            return [("cte", str(entry["key"])) for entry in entries if isinstance(entry, dict)]
    return []


def _nodes(tree: JsonValue) -> Iterator[dict[str, JsonValue]]:
    """Every object anywhere in `tree`."""
    pending = [tree]
    while pending:
        match node := pending.pop():
            case dict():
                yield node
                pending.extend(node.values())
            case list():
                pending.extend(node)


def _written(records: Iterable[Mapping[str, object] | str], staging: Path) -> int:
    """Write `records` one JSON line each to `staging`, answering the longest line's bytes."""
    longest = 0
    with staging.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            line = record if isinstance(record, str) else json.dumps(record, default=str)
            longest = max(longest, len(line.encode()))
            out.write(line + "\n")
    return longest
