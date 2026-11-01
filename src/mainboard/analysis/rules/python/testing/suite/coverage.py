"""Rules over test size and the cases a test covers (mcmr's PY-TEST coverage group)."""

from .....catalog import Example, Part, Rule


class OwnedTestStatementCount(Part):
    """Tests whose own body states more statements than one dense test needs."""

    maximum: int = 10

    query = """
        SELECT fact_id, node_span_path AS path, node_span_start_line AS start_line,
               node_span_start_column AS start_column, owned_statement_count AS value
        FROM test_function_tests WHERE is_collected AND owned_statement_count > $maximum
        QUALIFY row_number() OVER (PARTITION BY fact_id
                                   ORDER BY owned_statement_count DESC, ordinal) = 1
    """
    message = "owned test statement count is {value} for `{fact_id}`"


class FiniteRangeHypothesisCandidateCount(Rule, id="PY-TEST0014", applicability="review"):
    """Parametrized finite ranges wide enough for a Hypothesis strategy."""

    minimum_cases: int = 10

    query = """
        SELECT f.*, n AS value, renamed FROM test_function_facts f JOIN (
          SELECT v.fact_id, count(*) AS n,
                 array_to_string(list_sort(list_distinct(list(
                   s.scope || t.name || '[*] -> ' || s.scope || t.name))), ', ') AS renamed
          FROM test_function_values v
          JOIN test_function_tests t ON t.fact_id = v.fact_id AND t.record_id = v.parent_id
            AND t.is_collected
          JOIN collected_test_scopes() s ON s.path = t.path AND s.ordinal = t.ordinal
          WHERE v.relation = 'tests.parametrized_range_sizes' AND v.integer_value >= $minimum_cases
          GROUP BY v.fact_id) USING (fact_id)
    """
    message = "finite range hypothesis candidate count is {value} for `{fact_id}`"
    # The cases of each test become the one test: the id map the collection proof reads.
    change = "turned the {value} wide ranges of `{fact_id}` into Hypothesis integers: {renamed}"
    # `@pytest.mark.parametrize("n", range(20))` becomes `@given(n=st.integers(0, 19))`, both
    # names imported from `hypothesis`. The file is fixed whole or left alone: the finding is the
    # file's. Left alone: a test with other parameters than the one parametrized (a fixture
    # trips Hypothesis's health check, a second parametrize adds a parameter; `self` of a method
    # is fine), an async test, a negative bound, several argument names, `ids` or any other
    # keyword, a comment in the decorator, and a file where `given` or `st` mean something else.
    # A random draw replaces the walk through every value, a change to review. The row replaces
    # the decorator's call and captures its callee and arguments to drop them (`rewrite-node`).
    fix = r"""
        WITH wide AS (
          SELECT f.finding_key, f.path, t.fact_id, t.test, d.expr, d.ordinal AS decorator
          FROM finding f
          JOIN test_function_declared t ON t.path = f.path AND t.collected
          JOIN test_function_decorators d ON d.fact_id = t.fact_id AND d.fn = t.test
            AND ends_with(d.mark, 'parametrize')
          JOIN test_function_sizes s ON s.fact_id = t.fact_id AND s.test = t.test
            AND s.decorator = d.ordinal
          WHERE s.size >= $minimum_cases
        ), drawn AS (
          SELECT w.*, sc.name AS argument, b.ints, f.node_id AS callee, a1.node_id AS name_node,
                 a2.node_id AS bound_node
          FROM wide w
          JOIN test_function_nodes e ON e.fact_id = w.fact_id AND e.node_id = w.expr
          JOIN test_function_nodes f ON f.fact_id = w.fact_id AND f.parent_id = w.expr
            AND f.sibling_index = 0
          JOIN test_function_nodes l ON l.fact_id = w.fact_id AND l.parent_id = w.expr
            AND l.type = 'argument_list' AND l.children_count = 4
          JOIN test_function_arguments a1 ON a1.fact_id = w.fact_id AND a1.fn = w.test
            AND a1.decorator = w.decorator AND a1.position = 1 AND a1.type = 'string'
          JOIN test_function_nodes s1 ON s1.fact_id = w.fact_id AND s1.node_id = a1.node_id
            AND s1.children_count = 1
          JOIN test_function_nodes sc ON sc.fact_id = w.fact_id AND sc.parent_id = a1.node_id
            AND sc.type = 'string_content' AND regexp_matches(sc.name, '^[A-Za-z_][A-Za-z0-9_]*$')
          JOIN test_function_arguments a2 ON a2.fact_id = w.fact_id AND a2.fn = w.test
            AND a2.decorator = w.decorator AND a2.position = 2
          JOIN test_function_nodes rl ON rl.fact_id = w.fact_id AND rl.parent_id = a2.node_id
            AND rl.type = 'argument_list'
          JOIN (SELECT fact_id, parent_id, count(*) AS written,
                       list(try_cast(name AS BIGINT) ORDER BY sibling_index) AS ints
                FROM test_function_nodes WHERE type NOT IN ('(', ')')
                GROUP BY fact_id, parent_id) b
            ON b.fact_id = w.fact_id AND b.parent_id = rl.node_id
          WHERE b.written BETWEEN 1 AND 2 AND len(list_filter(b.ints, lambda x: x IS NOT NULL)) = b.written
            AND NOT EXISTS (SELECT 1 FROM test_function_nodes k
                            WHERE k.fact_id = w.fact_id AND k.type = 'comment'
                              AND k.node_id BETWEEN w.expr AND w.expr + e.descendant_count)
        ), judged AS (
          SELECT w.finding_key, w.path, w.fact_id, w.test, w.argument, w.expr, w.callee,
                 w.name_node, w.bound_node,
                 if(len(w.ints) = 1, 0, w.ints[1]) AS low,
                 w.ints[len(w.ints)] - 1 AS high
          FROM drawn w
          WHERE NOT EXISTS (SELECT 1 FROM test_function_declared t
                            WHERE t.fact_id = w.fact_id AND t.test = w.test AND t.is_async)
            AND NOT EXISTS (SELECT 1 FROM test_function_params p
                            WHERE p.fact_id = w.fact_id AND p.fn = w.test
                              AND p.name NOT IN (w.argument, 'self'))
        ), whole AS (
          SELECT w.finding_key FROM wide w
          GROUP BY w.finding_key
          HAVING count(*) = (SELECT count(*) FROM judged j WHERE j.finding_key = w.finding_key)
             AND NOT bool_or(EXISTS (
               SELECT 1 FROM (VALUES ('given', 'given'), ('st', 'strategies')) k(local, origin)
               WHERE EXISTS (SELECT 1 FROM import_binding_facts i
                             WHERE i.path = w.path AND i.name = k.local
                               AND (i.module <> 'hypothesis' OR i.imported_name <> k.origin))
                  OR (NOT EXISTS (SELECT 1 FROM import_binding_facts i
                                  WHERE i.path = w.path AND i.name = k.local
                                    AND i.module = 'hypothesis' AND i.imported_name = k.origin)
                      AND EXISTS (SELECT 1 FROM test_function_nodes n
                                  WHERE n.fact_id = w.fact_id AND n.type = 'identifier'
                                    AND n.name = k.local))))
        )
        SELECT j.finding_key, 'rewrite-node' AS operation, j.path, j.expr AS node,
               MAP {'CALLEE': [j.callee], 'ARGUMENTS': [j.name_node, j.bound_node]} AS captures,
               'given(' || j.argument || '=st.integers(' || j.low || ', ' || j.high || '))'
                 AS template,
               ['from hypothesis import given', 'from hypothesis import strategies as st']
                 AS imports
        FROM judged j JOIN whole USING (finding_key)
    """
    examples = (
        Example(
            bad=(
                """import pytest


@pytest.mark.parametrize("n", range(20))
def test_range(n):
    assert n >= 0
"""
            ),
            good=(
                """from hypothesis import given
from hypothesis import strategies as st


@given(n=st.integers(0, 19))
def test_range(n):
    assert n >= 0
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


class TestSpan:
    @pytest.mark.slow
    @pytest.mark.parametrize('size', range(5, 30))
    def test_size(self, size):
        assert size
"""
            ),
            good=(
                """import pytest
from hypothesis import given
from hypothesis import strategies as st


class TestSpan:
    @pytest.mark.slow
    @given(size=st.integers(5, 29))
    def test_size(self, size):
        assert size
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


@pytest.mark.parametrize("n", range(20))
def test_files(tmp_path, n):
    assert (tmp_path / str(n)).parent
"""
            ),
            good=(
                """import pytest


@pytest.mark.parametrize("n", range(20))
def test_files(tmp_path, n):
    assert (tmp_path / str(n)).parent
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


@pytest.mark.parametrize("n", range(0, 40, 2))
@pytest.mark.parametrize("m", range(20))
def test_pairs(n, m):
    assert n + m >= 0
"""
            ),
            good=(
                """import pytest


@pytest.mark.parametrize("n", range(0, 40, 2))
@pytest.mark.parametrize("m", range(20))
def test_pairs(n, m):
    assert n + m >= 0
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


@pytest.mark.parametrize("n", range(-5, 20))
def test_signed(n):
    assert n
"""
            ),
            good=(
                """import pytest


@pytest.mark.parametrize("n", range(-5, 20))
def test_signed(n):
    assert n
"""
            ),
        ),
        Example(
            bad=(
                """import pytest


@pytest.mark.parametrize("n", range(20))
async def test_waits(n):
    assert n >= 0
"""
            ),
            good=(
                """import pytest


@pytest.mark.parametrize("n", range(20))
async def test_waits(n):
    assert n >= 0
"""
            ),
        ),
        Example(
            bad=(
                """import pytest

st = 3


@pytest.mark.parametrize("n", range(20))
def test_range(n):
    assert n >= st
"""
            ),
            good=(
                """import pytest

st = 3


@pytest.mark.parametrize("n", range(20))
def test_range(n):
    assert n >= st
"""
            ),
        ),
    )
