from collections.abc import Mapping, Sequence

import duckdb

type Table = tuple[Mapping[str, str], Sequence[tuple]]


def _loaded(connection: duckdb.DuckDBPyConnection, name: str, table: Table) -> None:
    """Create the table `name` from (typed columns, rows), its rows stored in the order given."""
    columns, rows = table
    typed = ", ".join(f'"{column}" {sql}' for column, sql in columns.items())
    connection.execute(f"CREATE TABLE {name} ({typed})")
    for row in rows:
        connection.execute(f"INSERT INTO {name} VALUES ({', '.join('?' * len(columns))})", row)


def macros(**relations: Table) -> duckdb.DuckDBPyConnection:
    """An in-memory database where each of `relations` is a table macro `name()` of a family."""
    connection = duckdb.connect()
    for name, table in relations.items():
        _loaded(connection, f"{name}_rows", table)
        connection.execute(f"CREATE MACRO {name}() AS TABLE SELECT * FROM {name}_rows")
    return connection


def views(**relations: Table) -> duckdb.DuckDBPyConnection:
    """An in-memory database holding each of `relations` as a view `name` of a family."""
    connection = duckdb.connect()
    for name, table in relations.items():
        _loaded(connection, name, table)
    return connection
