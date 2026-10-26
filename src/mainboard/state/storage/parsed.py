# What a SQL text names, read by DuckDB's own parser: the one that runs the text later, so the
# answer never disagrees with execution (`FROM runs r` names `runs`, a CTE names nothing).

import json
from collections.abc import Iterator, Mapping

from pydantic import JsonValue

from .database import bare

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
