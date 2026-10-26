"""Rules over how async tests are run (mcmr's PY-TEST async_checks group)."""

from typing import Literal

from ....catalog import Example, Relation, Rule


class TestingLastImports(Relation):
    """The last line of the last top-level import statement of each Python file."""

    query = """
    SELECT path, max(end_line) AS line FROM import_binding_nodes
    WHERE role = 'declaration' AND start_column = 0 GROUP BY path
"""


class SynchronousTestAsyncioRunCount(Rule, id="PY-TEST0009"):
    """Synchronous tests that start their own event loop with `asyncio.run`."""

    query = """
        SELECT f.*, n AS value FROM test_function_facts f JOIN (
          SELECT c.fact_id, count(*) AS n FROM test_function_tests_calls c
          JOIN test_function_tests t ON t.fact_id = c.fact_id AND t.record_id = c.parent_id
          WHERE c.qualified_name = 'asyncio.run' AND t.is_collected AND NOT t.is_async
          GROUP BY c.fact_id) USING (fact_id)
    """
    message = "synchronous test asyncio run count is {value} for `{fact_id}`"


class UnownedAsyncTestCount(Rule, id="PY-TEST0010", applicability="review"):
    """Async tests that no anyio or asyncio mark or backend fixture owns."""

    discovery: Literal["explicit", "automatic"] = "explicit"

    query = """
        SELECT f.*, n AS value FROM test_function_facts f JOIN (
          SELECT fact_id, count(*) AS n FROM test_function_tests t
          WHERE is_collected AND is_async AND $discovery = 'explicit' AND NOT EXISTS (
            SELECT 1 FROM test_function_values v
            WHERE v.fact_id = t.fact_id AND v.parent_id = t.record_id
              AND ((v.relation = 'tests.marks'
                    AND v.string_value IN ('pytest.mark.anyio', 'pytest.mark.asyncio',
                                           'anyio', 'asyncio'))
                   OR (v.relation IN ('tests.fixture_names', 'tests.requested_fixture_names')
                       AND v.string_value = 'anyio_backend')))
          GROUP BY fact_id) USING (fact_id)
    """
    message = "unowned async test count is {value} for `{fact_id}`"
    # Each unowned async test gets `@pytest.mark.anyio` above its first decorator (or its
    # `def`), at its own indentation; the facts read a test's own marks, so the proof sees each
    # one owned.
    fix = """
        SELECT f.finding_key, t.node_span_path AS path, t.node_span_start_line AS start_line,
               0 AS start_column, t.node_span_start_line AS end_line, 0 AS end_column,
               repeat(' ', t.node_span_start_column) || '@pytest.mark.anyio' || chr(10) AS text,
               'import pytest' AS "import"
        FROM finding f JOIN test_function_tests t USING (fact_id)
        WHERE t.is_collected AND t.is_async AND NOT EXISTS (
          SELECT 1 FROM test_function_values v
          WHERE v.fact_id = t.fact_id AND v.parent_id = t.record_id
            AND ((v.relation = 'tests.marks'
                  AND v.string_value IN ('pytest.mark.anyio', 'pytest.mark.asyncio',
                                         'anyio', 'asyncio'))
                 OR (v.relation IN ('tests.fixture_names', 'tests.requested_fixture_names')
                     AND v.string_value = 'anyio_backend')))
    """
    examples = (
        Example(
            bad="import asyncio\n\n\nasync def test_wait():\n    await asyncio.sleep(0)\n",
            good=(
                """import asyncio
import pytest


@pytest.mark.anyio
async def test_wait():
    await asyncio.sleep(0)
"""
            ),
        ),
    )
