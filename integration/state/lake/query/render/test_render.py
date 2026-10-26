import json
from collections.abc import Callable
from pathlib import Path

import pytest
from dirty_equals import IsJson, IsPartialDict, IsStr

from mainboard import Results

from ..constants import MAPS
from ..rendering import Rendering

_RANGE = "SELECT * FROM range(20)"
_WIDE = "SELECT i, repeat('x', 20) AS text FROM range(100) t(i)"
_BIG = "SELECT i, repeat('x', 6000) AS text FROM range(10) t(i)"
_NULLS = "SELECT NULL AS absent, '' AS empty"
_X_ROW = "x" * 20
_TABLES = r"(?: results \n(?:.*\n)*?└───────┘\n){3}"


def _lines(*lines: str) -> str:
    return "".join(f"{line}\n" for line in lines)


_MAPS_JSON = _lines(
    json.dumps(
        [
            {
                "dates": {"2020-01-01": {"items": [2, 3]}, "2020-01-02": {"items": [1, None]}},
                "times": {"2020-01-01 12:00:00": [4, 5]},
            }
        ],
        separators=(",", ":"),
    )
)


@pytest.mark.parametrize(
    ("sql", "caps", "body", "rows", "displayed"),
    [
        pytest.param(
            f"{_NULLS}, 'NULL' AS literal",
            {},
            _lines("absent\tempty\tliteral", 'NULL\t\t"NULL"'),
            1,
            1,
            id="null-empty-and-literal-null-are-told-apart",
        ),
        pytest.param(
            f"{_NULLS} WHERE false",
            {},
            _lines("absent\tempty"),
            0,
            0,
            id="an-empty-result-keeps-its-header",
        ),
        pytest.param(
            _RANGE,
            {"max_rows": 6, "max_bytes": 0},
            _lines("range", "0", "1", "2", "… 14 rows omitted (--max-rows 6) …", "17", "18", "19"),
            20,
            6,
            id="the-row-cap-keeps-a-prefix-and-a-suffix-around-a-counted-middle",
        ),
        pytest.param(
            _RANGE,
            {"max_rows": 0, "max_bytes": -1},
            _lines("range", *map(str, range(20))),
            20,
            20,
            id="a-cap-of-zero-or-minus-one-lifts-it",
        ),
        pytest.param(
            _WIDE,
            {"max_rows": 0, "max_bytes": 300},
            _lines(
                "i\ttext",
                *(f"{i}\t{_X_ROW}" for i in range(6)),
                "… 89 rows omitted (--max-bytes 300) …",
                *(f"{i}\t{_X_ROW}" for i in range(95, 100)),
            ),
            100,
            11,
            id="the-byte-cap-bounds-what-is-shown",
        ),
        pytest.param(
            "SELECT repeat('x', 6000) AS text",
            {"max_bytes": 100},
            _lines("text", "… 1 rows omitted (--max-bytes 100; 0 rows displayed) …"),
            1,
            0,
            id="a-row-wider-than-the-byte-cap-is-omitted-whole",
        ),
        pytest.param(
            _BIG,
            {},
            _lines(
                "i\ttext",
                f"0\t{'x' * 6000}",
                "… 9 rows omitted (--max-bytes 10000; 1 row displayed) …",
            ),
            10,
            1,
            id="the-first-row-fits-the-default-byte-cap",
        ),
        pytest.param(
            "SELECT repeat('x', 60) AS text FROM range(3)",
            {"max_rows": 1, "max_bytes": 100},
            _lines(
                "text",
                "x" * 60,
                "… 2 rows omitted (--max-rows 1; --max-bytes 100; 1 row displayed) …",
            ),
            3,
            1,
            id="both-caps-are-named-when-both-were-hit",
        ),
        pytest.param(
            _RANGE,
            {"mode": "json", "max_rows": 6, "max_bytes": 0},
            IsJson(
                [
                    *({"range": i} for i in range(3)),
                    {
                        "_truncated": IsPartialDict(
                            omitted=14,
                            displayed=6,
                            limits=["--max-rows 6"],
                            rows=20,
                            complete=True,
                        )
                    },
                    *({"range": i} for i in range(17, 20)),
                ]
            ),
            20,
            6,
            id="a-json-array-stays-an-array-with-a-truncation-object",
        ),
        pytest.param(
            _NULLS,
            {"mode": "json", "max_rows": -1, "max_bytes": 0},
            '[{"absent":null,"empty":""}]\n',
            1,
            1,
            id="json-keeps-null-apart-from-empty-text",
        ),
        pytest.param(
            "SELECT repeat('x', 6000) AS text",
            {"mode": "json", "max_bytes": 1},
            IsJson(
                [
                    {
                        "_truncated": IsPartialDict(
                            columns=["text"], displayed=0, omitted=1, limits=["--max-bytes 1"]
                        )
                    }
                ]
            ),
            1,
            0,
            id="a-json-byte-cap-reports-the-columns-it-omitted",
        ),
        pytest.param(
            MAPS,
            {"mode": "json"},
            _MAPS_JSON,
            1,
            1,
            id="nested-maps-print-in-sorted-key-order",
        ),
        pytest.param(
            "SELECT * FROM range(33)",
            {"mode": "human", "max_rows": 0, "max_bytes": 0},
            IsStr(regex=_TABLES),
            33,
            33,
            id="uncapped-human-output-streams-one-table-per-batch",
        ),
        pytest.param(
            _NULLS,
            {"mode": "human"},
            IsStr(regex=r"(?s).*absent.*empty.*"),
            1,
            1,
            id="human-output-names-the-columns",
        ),
        pytest.param(
            "CREATE TABLE scratch AS SELECT 1 AS a",
            {},
            _lines("Count", "1"),
            1,
            1,
            id="a-write-answers-its-count",
        ),
        pytest.param(
            "SELECT * FROM query('SELECT 7 AS n')",
            {"mode": "json"},
            '[{"n":7}]\n',
            1,
            1,
            id="a-table-function-runs-a-nested-statement",
        ),
        pytest.param(
            "SELECT * FROM range(2)",
            {"mode": "json"},
            '[{"range":0},{"range":1}]\n',
            2,
            2,
            id="a-table-function-ranges",
        ),
        pytest.param(
            "PRAGMA version",
            {},
            IsStr(regex=r"library_version\tsource_id\tcodename\nv.*\n"),
            1,
            1,
            id="a-pragma-answers-its-table",
        ),
    ],
)
def test_a_query_renders_within_both_caps_and_counts_what_it_leaves_out(
    render: Callable[..., Rendering],
    workspace: Path,
    sql: str,
    caps: dict[str, str | int],
    body: str,
    rows: int,
    displayed: int,
) -> None:
    shown = render(sql, **caps)
    assert shown.body == body
    assert (shown.footer.rows, shown.stamp.rows, shown.displayed) == (rows, rows, displayed)
    assert shown.footer.snapshot_id == 0 and not (workspace / ".mb").exists()


def test_a_runaway_query_stops_after_its_extra_rows_and_labels_the_partial_hash(
    render: Callable[..., Rendering],
) -> None:
    shown = render("SELECT * FROM range(1000000000000)")
    assert not shown.stamp.complete and not shown.stamp.timed_out
    assert 1000 <= shown.stamp.rows <= 11001 and shown.footer.rows == shown.stamp.rows
    assert all(
        text in shown.out for text in ("at least", "query stopped early", "(partial)", "omitted")
    )
    assert len(shown.out.encode()) < 10500 and len(shown.out.splitlines()) <= 1003


@pytest.mark.parametrize(
    ("sql", "marker", "lines"),
    [
        pytest.param("EXPLAIN SELECT 1 AS number", "Projection", 6, id="explain"),
        pytest.param("EXPLAIN ANALYZE SELECT 1 AS number", "Projection", 6, id="explain-analyze"),
        pytest.param("DESCRIBE SELECT 1 AS number", "number", 3, id="describe"),
        pytest.param("SUMMARIZE SELECT 1 AS number", "number", 3, id="summarize"),
        pytest.param(
            "/* 日本 */ EXPLAIN (FORMAT TEXT) SELECT 'ANALYZE' AS literal",
            "Projection",
            6,
            id="explain-after-a-comment-with-the-format-option",
        ),
        pytest.param(
            "EXPLAIN (SELECT 'ANALYZE' AS literal)", "Projection", 6, id="explain-a-parenthesis"
        ),
        pytest.param(
            "EXPLAIN ((SELECT 'ANALYZE' AS literal))",
            "Projection",
            6,
            id="explain-a-nested-parenthesis",
        ),
        pytest.param(
            "EXPLAIN (FORMAT TEXT) (SELECT 'ANALYZE' AS literal)",
            "Projection",
            6,
            id="explain-the-format-option-then-a-parenthesis",
        ),
    ],
)
def test_inspection_statements_render_their_plans_over_real_lines_and_can_be_materialized(
    render: Callable[..., Rendering], workspace: Path, sql: str, marker: str, lines: int
) -> None:
    shown = render(sql)
    assert marker in shown.out and "\\n" not in shown.out
    assert len(shown.out.splitlines()) >= lines and shown.footer.rows > 0
    assert Results(workspace).query(sql).fetchall()
