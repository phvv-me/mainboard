"""A relation `Results.query` answers keeps its column types without pyarrow, even when empty, and
is read in once; a query that cannot run is refused."""

import sys
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from mainboard import MissionError, Results
from mainboard.state import records


@pytest.fixture
def arrowless_results(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Results:
    """The workspace's results, with pyarrow made unimportable."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    return Results(workspace)


def test_relation_queries_preserve_types_without_pyarrow(arrowless_results: Results) -> None:
    relation = arrowless_results.query("""
        SELECT i::TINYINT AS x,
               CASE WHEN i = 2 THEN NULL ELSE i / 10 END::DECIMAL(4,2) AS y,
               [struct_pack(n := i::SMALLINT)] AS nested
        FROM range(4) t(i) ORDER BY i DESC
    """)
    assert [str(kind) for kind in relation.types] == [
        "TINYINT",
        "DECIMAL(4,2)",
        "STRUCT(n SMALLINT)[]",
    ]
    # The query read everything in, so the relation outlives what it read.
    assert records(relation) == [
        {"x": i, "y": None if i == 2 else Decimal(i) / 10, "nested": [{"n": i}]}
        for i in (3, 2, 1, 0)
    ]


def test_an_empty_relation_keeps_its_column_types_without_pyarrow(
    arrowless_results: Results,
) -> None:
    empty = arrowless_results.query("""
        SELECT NULL::SMALLINT AS x, []::STRUCT(n TINYINT)[] AS nested WHERE false
    """)
    assert empty.fetchall() == []
    assert [str(kind) for kind in empty.types] == ["SMALLINT", "STRUCT(n TINYINT)[]"]


def test_a_query_that_cannot_run_is_refused_without_pyarrow(arrowless_results: Results) -> None:
    with pytest.raises(MissionError, match="not_a_column"):
        arrowless_results.query("SELECT not_a_column")
    with pytest.raises(MissionError, match="Could not convert"):
        arrowless_results.query("SELECT CAST('bad' AS INTEGER)")
    with pytest.raises(ValueError, match="one statement"):
        arrowless_results.query("SELECT 1; SELECT 2")


def test_relation_queries_execute_exactly_once(workspace, monkeypatch) -> None:
    results = Results(workspace)
    original = results.views.load

    def views(connection: duckdb.DuckDBPyConnection, project: str, *, sql: str) -> None:
        connection.execute("CREATE SEQUENCE query_counter")
        original(connection, project, sql=sql)

    monkeypatch.setattr(results.views, "load", views)
    relation = results.query("SELECT nextval('query_counter') AS n FROM range(4)")
    assert relation.fetchall() == [(1,), (2,), (3,), (4,)]
    assert relation.fetchall() == [(1,), (2,), (3,), (4,)]
