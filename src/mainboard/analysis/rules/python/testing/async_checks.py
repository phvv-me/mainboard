"""Rules over how async tests are run (mcmr's PY-TEST async_checks group)."""

from typing import Literal

from ....catalog import Example, Relation, Rule


class TestingLastImports(Relation):
    """The last line of the last top-level import statement of each Python file."""

    query = """
    SELECT path, max(end_line) AS line FROM import_binding_nodes
    WHERE role = 'declaration' AND start_column = 0 GROUP BY path
"""


class SynchronousTestAsyncioRunCount(Rule, id="PY-TEST0009", applicability="review"):
    """Synchronous tests that start their own event loop with `asyncio.run`."""

    query = """
        SELECT f.*, n AS value FROM test_function_facts f JOIN (
          SELECT c.fact_id, count(*) AS n FROM test_function_tests_calls c
          JOIN test_function_tests t ON t.fact_id = c.fact_id AND t.record_id = c.parent_id
          WHERE c.qualified_name = 'asyncio.run' AND t.is_collected AND NOT t.is_async
          GROUP BY c.fact_id) USING (fact_id)
    """
    message = "synchronous test asyncio run count is {value} for `{fact_id}`"
    change = "made the {value} synchronous tests of `{fact_id}` async, awaiting where they ran"
    # Every such test in the file becomes `async def` under `@pytest.mark.anyio`, and each
    # `asyncio.run(coroutine)` of its own body becomes `await coroutine` (in parentheses where an
    # attribute, subscript or call follows). The file is fixed whole or left alone: the finding is
    # the file's. Left alone: a test of a `TestCase`, one under `given`, anyio or asyncio marks, a
    # call spelled any other way than `asyncio.run(one_argument)`, one holding a comment, one in
    # a lambda or generator expression, or one whose `def` is indented with tabs. The loop moves from
    # `asyncio.run` to the one anyio runs, a change to review. One `rewrite-node` row per `def`
    # keyword and per call, all of the finding: a row that refuses withholds the whole file.
    fix = r"""
        WITH tests AS (
          SELECT f.finding_key, t.fact_id, t.path, t.test, t.holder, n.start_line, n.start_column,
                 d.node_id AS keyword,
                 (SELECT count(*) FROM test_function_calls c
                  WHERE c.fact_id = t.fact_id AND c.test = t.test
                    AND c.qualified_name = 'asyncio.run') AS expected
          FROM finding f
          JOIN test_function_declared t ON t.path = f.path
          JOIN test_function_nodes n ON n.fact_id = t.fact_id AND n.node_id = t.test
          JOIN test_function_nodes d ON d.fact_id = t.fact_id AND d.parent_id = t.test
            AND d.type = 'def'
          WHERE t.collected AND NOT t.is_async
        ), runs AS (
          SELECT t.finding_key, t.fact_id, t.test, c.node_id AS call, r.node_id AS callee,
                 a.node_id AS argument, c.ptype IN ('attribute', 'subscript', 'call') AS wrapped
          FROM tests t
          JOIN test_function_inside c ON c.fact_id = t.fact_id AND c.in_test = t.test
            AND c.type = 'call' AND c.owner = c.in_test
          JOIN test_function_nodes r ON r.fact_id = c.fact_id AND r.parent_id = c.node_id
            AND r.sibling_index = 0 AND r.name = 'asyncio.run'
          JOIN test_function_nodes l ON l.fact_id = c.fact_id AND l.parent_id = c.node_id
            AND l.type = 'argument_list' AND l.children_count = 3
          JOIN test_function_nodes a ON a.fact_id = c.fact_id AND a.parent_id = l.node_id
            AND a.sibling_index = 1 AND a.type IN ('call', 'identifier', 'attribute', 'subscript')
          WHERE NOT EXISTS (
            SELECT 1 FROM test_function_nodes q
            WHERE q.fact_id = c.fact_id AND q.node_id > t.test AND q.node_id <= c.node_id
              AND q.node_id + q.descendant_count >= c.node_id
              AND q.type IN ('lambda', 'generator_expression'))
            AND NOT EXISTS (
              SELECT 1 FROM test_function_nodes k
              WHERE k.fact_id = c.fact_id AND k.type = 'comment'
                AND k.node_id BETWEEN c.node_id AND c.node_id + c.descendant_count)
        ), judged AS (
          SELECT t.finding_key, t.fact_id, t.path, t.test, t.start_column, t.keyword,
                 coalesce(substr(x.text, 1, t.start_column), '') AS indent,
                 (SELECT count(*) FROM runs r WHERE r.fact_id = t.fact_id AND r.test = t.test)
                   = t.expected
                 AND NOT EXISTS (
                   SELECT 1 FROM test_function_decorators d
                   WHERE d.fact_id = t.fact_id AND d.fn = t.test
                     AND (regexp_extract(d.mark, '[^.]*$') IN ('anyio', 'asyncio', 'given')))
                 AND NOT EXISTS (
                   SELECT 1 FROM test_function_nodes k
                   JOIN test_function_nodes b ON b.fact_id = k.fact_id AND b.parent_id = k.node_id
                     AND b.type = 'argument_list'
                   JOIN test_function_nodes u ON u.fact_id = b.fact_id AND u.parent_id = b.node_id
                   WHERE k.fact_id = t.fact_id AND k.type = 'class_definition'
                     AND k.node_id = (SELECT h.parent_id FROM test_function_nodes h
                                      WHERE h.fact_id = t.fact_id AND h.node_id = t.holder)
                     AND ends_with(u.name, 'TestCase')) AS fixable
          FROM tests t
          LEFT JOIN source_lines() x ON x.path = t.path AND x.line = t.start_line
          WHERE t.expected > 0
        ), whole AS (
          SELECT finding_key FROM judged GROUP BY finding_key
          HAVING bool_and(fixable AND indent = repeat(' ', start_column))
        )
        SELECT j.finding_key, 'rewrite-node' AS operation, j.path, j.keyword AS node,
               MAP {'OLD': [j.keyword]} AS captures,
               '@pytest.mark.anyio' || chr(10) || 'async def' AS template,
               ['import pytest'] AS imports
        FROM judged j JOIN whole USING (finding_key)
        UNION ALL
        SELECT j.finding_key, 'rewrite-node', j.path, r.call,
               MAP {'CALLEE': [r.callee], 'ARGUMENT': [r.argument]},
               if(r.wrapped, '(await $ARGUMENT)', 'await $ARGUMENT'), []::VARCHAR[]
        FROM judged j JOIN whole USING (finding_key)
        JOIN runs r ON r.fact_id = j.fact_id AND r.test = j.test
    """
    examples = (
        Example(
            bad=(
                """import asyncio


def test_two():
    asyncio.run(asyncio.sleep(0))
"""
            ),
            good=(
                """import asyncio

import pytest


@pytest.mark.anyio
async def test_two():
    await asyncio.sleep(0)
"""
            ),
        ),
        Example(
            bad=(
                """import asyncio

import pytest


async def fetch():
    return 1


class TestFetch:
    @pytest.mark.slow
    def test_value(self):
        value = asyncio.run(fetch())
        assert value

    def test_name(self):
        assert asyncio.run(fetch()).real
"""
            ),
            good=(
                """import pytest


async def fetch():
    return 1


class TestFetch:
    @pytest.mark.slow
    @pytest.mark.anyio
    async def test_value(self):
        value = await fetch()
        assert value

    @pytest.mark.anyio
    async def test_name(self):
        assert (await fetch()).real
"""
            ),
        ),
        Example(
            bad=(
                """import asyncio
import unittest


class TestWait(unittest.TestCase):
    def test_wait(self):
        asyncio.run(asyncio.sleep(0))


def test_plain():
    asyncio.run(asyncio.sleep(0))
"""
            ),
            good=(
                """import asyncio
import unittest


class TestWait(unittest.TestCase):
    def test_wait(self):
        asyncio.run(asyncio.sleep(0))


def test_plain():
    asyncio.run(asyncio.sleep(0))
"""
            ),
        ),
        Example(
            bad=(
                """import asyncio


def test_debug():
    asyncio.run(asyncio.sleep(0), debug=True)
"""
            ),
            good=(
                """import asyncio


def test_debug():
    asyncio.run(asyncio.sleep(0), debug=True)
"""
            ),
        ),
        Example(
            bad=(
                """import asyncio


async def fetch(n):
    return n


def test_noted():
    asyncio.run(
        fetch(
            1,  # one
        )
    )
"""
            ),
            good=(
                """import asyncio


async def fetch(n):
    return n


def test_noted():
    asyncio.run(
        fetch(
            1,  # one
        )
    )
"""
            ),
        ),
    )


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
