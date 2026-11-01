from collections.abc import Iterable

import duckdb
from pathspec import GitIgnoreSpec

from mainboard.analysis.catalog import Bundle


def staged(bundle: Bundle) -> duckdb.DuckDBPyConnection:
    """An in-memory database holding the tables `bundle.staged()` fills."""
    connection = duckdb.connect()
    for statement in bundle.staged():
        connection.execute(statement.sql, statement.parameters)
    return connection


def gitignored(lines: Iterable[str], path: str) -> bool:
    """Whether gitignore `lines` ignore `path`, read as pathspec's forward evaluation does.

    The last matching line decides, except that a `!` line matching only through a parent folder
    (pathspec's `ps_d` group) does not overrule a line that matched the file itself. No public
    backend of pathspec reaches this order: re2 breaks ties arbitrarily and `simple` reads the
    lines in reverse. Git agrees except where a `!` line re-includes below an excluded parent.
    """
    ignored, decided = False, 0
    for pattern in GitIgnoreSpec.from_lines(lines).patterns:
        if pattern.include is None or (found := pattern.match_file(path)) is None:
            continue
        reach = 1 if found.match.groupdict().get("ps_d") else 2
        if pattern.include or reach >= decided:
            ignored, decided = pattern.include, reach
    return ignored
