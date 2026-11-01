"""Rules over tests that repeat each other's intent (mcmr's PY-TEST redundancy group)."""

from ....catalog import Example, Relation, Rule
from ....operations.python.suites import MergeTests
from .cases import MergeableTestGroups

_TEST = """t.fact_order, t.ordinal, t.path, t.node_span_start_line AS start_line,
    t.node_span_start_column AS start_column, t.name, t.body_shape"""


class PytestReaching(Relation):
    """The `pytest_reaching()` relation."""

    query = f"""
    SELECT {_TEST}, l.*
    FROM test_function_tests t JOIN (
      SELECT fact_id, parent_id AS record_id,
        list(string_value ORDER BY ordinal) FILTER (relation = 'tests.reachable_targets')
          AS reachable_targets,
        list(string_value ORDER BY ordinal) FILTER (relation = 'tests.fixture_names')
          AS fixture_names,
        list(string_value ORDER BY ordinal) FILTER (relation = 'tests.literal_values')
          AS literal_values
      FROM test_function_values
      WHERE entry_kind = 'value' AND relation IN
        ('tests.reachable_targets', 'tests.fixture_names', 'tests.literal_values')
      GROUP BY fact_id, parent_id HAVING reachable_targets IS NOT NULL) l
      USING (fact_id, record_id)
    WHERE t.is_collected
"""


_CLUSTER = """
    arg_min(path, (fact_order, ordinal)) AS path,
    arg_min(start_line, (fact_order, ordinal)) AS start_line,
    arg_min(start_column, (fact_order, ordinal)) AS start_column,
    count(*) AS test_count, string_agg(name, '`, `' ORDER BY name) AS test_names
"""
_INTENTS = "count(DISTINCT body_shape) AS intent_count"


# What makes tests duplicates: the same body shape, literals, fixtures and production code reached.
_KEYS = "body_shape, literal_values, fixture_names, reachable_targets"
# The SQL condition that a duplicate test `d` and a row `c` or `f` (finding) share their cluster.
_SAME = {
    row: " AND ".join(f"d.{key} IS NOT DISTINCT FROM {row}.{key}" for key in _KEYS.split(", "))
    for row in ("c", "f")
}
_LIB = {"lib.py": "def b(i):\n    return i\n"}
# A refusal: the duplicate has a mark of its own, which the deletion would lose.
_SLOWED = """import pytest
from lib import b


def test_b_is_positive():
    assert b(1) > 0


@pytest.mark.slow
def test_b_is_positive_and_slow():
    assert b(1) > 0
"""
# A refusal: no two names share a word beyond `test`, so no merged name can be given.
_UNNAMED = """from lib import b


def test_ant():
    assert b(1) > 0


def test_bee():
    assert b(2) > 0


def test_cat():
    assert b(3) > 0


def test_dog():
    assert b(4) > 0


def test_eel():
    assert b(1) > 0
    assert b(1) < 100


def test_fox():
    assert b(2) > 0
    assert b(2) < 100


def test_gnu():
    assert b(3) > 0
    assert b(3) < 100


def test_hen():
    assert b(4) > 0
    assert b(4) < 100
"""


class DuplicateTests(Relation):
    """Each test of a cluster of duplicate tests (one body shape, literals, fixtures and production
    code reached) none of which carries a decorator, so that all but the first can go.

    `identifier` is the test's pytest node id and `dropped` whether it goes: every test of its
    cluster but the first in file order.
    """

    query = f"""
    WITH cases AS (
      SELECT r.fact_order, r.ordinal, r.path, r.body_shape, r.literal_values, r.fixture_names,
             r.reachable_targets, d.test, i.scope || d.name AS identifier,
             EXISTS (SELECT 1 FROM test_function_decorators k
                     WHERE k.fact_id = d.fact_id AND k.fn = d.test) AS decorated
      FROM pytest_reaching() r
      JOIN test_function_declared d ON d.path = r.path AND d.ordinal = r.ordinal
      JOIN collected_test_scopes() i ON i.path = d.path AND i.ordinal = d.ordinal
    ), clustered AS (
      SELECT *, count(*) OVER one AS size, bool_or(decorated) OVER one AS any_decorated,
             row_number() OVER (one ORDER BY fact_order, ordinal) AS place
      FROM cases WINDOW one AS (PARTITION BY {_KEYS})
    )
    SELECT * EXCLUDE (decorated, any_decorated, size, place), place > 1 AS dropped
    FROM clustered WHERE size > 1 AND NOT any_decorated
"""


class DuplicateTestIntentClusterCount(Rule, id="PY-TEST0015", applicability="review"):
    """Tests with the same body shape, literals and fixtures reaching the same production code."""

    minimum_tests: int = 2

    query = f"""
        SELECT c.*, d.dropped, d.kept FROM (
          SELECT {_CLUSTER}, {_KEYS} FROM pytest_reaching()
          GROUP BY {_KEYS}
          HAVING test_count >= $minimum_tests) c
        LEFT JOIN (
          SELECT {_KEYS},
                 string_agg(identifier || ' -> gone', ', ' ORDER BY fact_order, ordinal)
                   FILTER (dropped) AS dropped,
                 arg_min(identifier, (fact_order, ordinal)) AS kept
          FROM {DuplicateTests.name()}() GROUP BY {_KEYS}) d
          ON {_SAME["c"]}
    """
    message = "duplicate test intent reaches the same production graph in `{test_names}`"
    change = "removed the duplicates {dropped}, kept {kept}"
    # Only a cluster whose tests all go bare is offered, and it keeps its first test.
    fix = f"""
        SELECT f.finding_key, 'delete-node' AS operation, d.path, d.test AS node
        FROM finding f JOIN {DuplicateTests.name()}() d ON {_SAME["f"]}
        WHERE d.dropped
    """
    examples = (
        Example(
            bad=(
                """from lib import b


def test_b_is_positive():
    assert b(1) > 0


def test_b_stays_positive():
    assert b(1) > 0
"""
            ),
            good=(
                """from lib import b


def test_b_is_positive():
    assert b(1) > 0
"""
            ),
            context=_LIB,
        ),
        Example(bad=_SLOWED, good=_SLOWED, context=_LIB),
    )


class ConcentratedTestReachClusterCount(Rule, id="PY-TEST0016", applicability="review"):
    """Many tests of few intents reaching the same production code through the same fixtures."""

    minimum_tests: int = 4

    query = f"""
        SELECT c.*, m.renamed FROM (
          SELECT {_CLUSTER}, {_INTENTS}, fixture_names, reachable_targets FROM pytest_reaching()
          GROUP BY fixture_names, reachable_targets
          HAVING test_count >= $minimum_tests AND intent_count > 1
            AND intent_count * 2 <= test_count) c
        LEFT JOIN (
          SELECT fixture_names, reachable_targets,
                 string_agg(renamed, ', ' ORDER BY path, tests[1]) AS renamed
          FROM {MergeableTestGroups.name()}() GROUP BY fixture_names, reachable_targets) m
          ON m.fixture_names IS NOT DISTINCT FROM c.fixture_names
            AND m.reachable_targets IS NOT DISTINCT FROM c.reachable_targets
    """
    message = "repeated test intent concentrates the same production reach across `{test_names}`"
    change = "parametrized {renamed}"
    # Each intent held by two or more tests of the cluster becomes one parametrized test.
    fix = f"""
        SELECT f.finding_key, '{MergeTests.name}' AS operation, g.path, g.holder AS node, g.tests,
               g.merged AS name, g.ids
        FROM finding f JOIN {MergeableTestGroups.name()}() g
          ON g.fixture_names IS NOT DISTINCT FROM f.fixture_names
            AND g.reachable_targets IS NOT DISTINCT FROM f.reachable_targets
    """
    examples = (
        Example(
            bad=(
                """from lib import b


def test_positive_one():
    assert b(1) > 0


def test_positive_two():
    assert b(2) > 0


def test_positive_three():
    assert b(3) > 0


def test_bounded_one():
    assert b(1) > 0
    assert b(1) < 100


def test_bounded_two():
    assert b(2) > 0
    assert b(2) < 100


def test_bounded_three():
    assert b(3) > 0
    assert b(3) < 100
"""
            ),
            good=(
                """import pytest

from lib import b


@pytest.mark.parametrize(
    "case_0",
    [pytest.param(1, id="one"), pytest.param(2, id="two"), pytest.param(3, id="three")],
)
def test_positive(case_0):
    assert b(case_0) > 0


@pytest.mark.parametrize(
    "case_0",
    [pytest.param(1, id="one"), pytest.param(2, id="two"), pytest.param(3, id="three")],
)
def test_bounded(case_0):
    assert b(case_0) > 0
    assert b(case_0) < 100
"""
            ),
            context={"lib.py": "def b(i):\n    return i\n"},
        ),
        Example(
            bad=(
                """from lib import b


def test_positive_one():
    assert b(1) > 0


def test_positive_two():
    assert b(2) > 0


def test_positive_three():
    assert b(3) > 0


def test_bounded():
    assert b(1) > 0
    assert b(1) < 100
"""
            ),
            good=(
                """import pytest

from lib import b


@pytest.mark.parametrize(
    "case_0",
    [pytest.param(1, id="one"), pytest.param(2, id="two"), pytest.param(3, id="three")],
)
def test_positive(case_0):
    assert b(case_0) > 0


def test_bounded():
    assert b(1) > 0
    assert b(1) < 100
"""
            ),
            context={"lib.py": "def b(i):\n    return i\n"},
        ),
        Example(
            bad=(
                """import pytest
from lib import b


@pytest.mark.skipif(b(0) == 1, reason="one")
def test_positive_one():
    assert b(1) > 0


@pytest.mark.skipif(b(0) == 2, reason="two")
def test_positive_two():
    assert b(2) > 0


def test_bounded_one():
    assert b(1) > 0
    assert b(1) < 100


def test_bounded_two():
    assert b(2) > 0
    assert b(2) < 100
"""
            ),
            good=(
                """import pytest
from lib import b


@pytest.mark.skipif(b(0) == 1, reason="one")
def test_positive_one():
    assert b(1) > 0


@pytest.mark.skipif(b(0) == 2, reason="two")
def test_positive_two():
    assert b(2) > 0


def test_bounded_one():
    assert b(1) > 0
    assert b(1) < 100


def test_bounded_two():
    assert b(2) > 0
    assert b(2) < 100
"""
            ),
            context={"lib.py": "def b(i):\n    return i\n"},
        ),
    )


class ProductionReachHotspotCount(Rule, id="PY-TEST0019", applicability="review"):
    """Production code that many tests of few intents reach."""

    minimum_tests: int = 8

    query = f"""
        SELECT c.*, m.renamed FROM (
          SELECT {_CLUSTER}, {_INTENTS}, reachable FROM (
            SELECT {_TEST}, v.string_value AS reachable
            FROM test_function_tests t JOIN test_function_values v
              ON v.fact_id = t.fact_id AND v.parent_id = t.record_id
            WHERE t.is_collected AND v.relation = 'tests.reachable_targets'
              AND v.entry_kind = 'value')
          GROUP BY reachable HAVING test_count >= $minimum_tests AND intent_count * 2 <= test_count
          ) c
        LEFT JOIN (
          SELECT r.reachable, string_agg(g.renamed, ', ' ORDER BY g.path, g.tests[1]) AS renamed
          FROM {MergeableTestGroups.name()}() g, unnest(g.reachable_targets) AS r(reachable)
          GROUP BY r.reachable) m ON m.reachable = c.reachable
    """
    message = "low-diversity tests repeatedly reach `{reachable}` through `{test_names}`"
    change = "parametrized {renamed}"
    # Each intent held by two or more tests that reach the code becomes one parametrized test.
    fix = f"""
        SELECT f.finding_key, '{MergeTests.name}' AS operation, g.path, g.holder AS node, g.tests,
               g.merged AS name, g.ids
        FROM finding f JOIN {MergeableTestGroups.name()}() g
          ON list_contains(g.reachable_targets, f.reachable)
    """
    examples = (
        Example(
            bad=(
                """from lib import b


def test_positive_one():
    assert b(1) > 0


def test_positive_two():
    assert b(2) > 0


def test_positive_three():
    assert b(3) > 0


def test_positive_four():
    assert b(4) > 0


def test_bounded_one():
    assert b(1) > 0
    assert b(1) < 100


def test_bounded_two():
    assert b(2) > 0
    assert b(2) < 100


def test_bounded_three():
    assert b(3) > 0
    assert b(3) < 100


def test_bounded_four():
    assert b(4) > 0
    assert b(4) < 100
"""
            ),
            good=(
                """import pytest

from lib import b


@pytest.mark.parametrize(
    "case_0",
    [
        pytest.param(1, id="one"),
        pytest.param(2, id="two"),
        pytest.param(3, id="three"),
        pytest.param(4, id="four"),
    ],
)
def test_positive(case_0):
    assert b(case_0) > 0


@pytest.mark.parametrize(
    "case_0",
    [
        pytest.param(1, id="one"),
        pytest.param(2, id="two"),
        pytest.param(3, id="three"),
        pytest.param(4, id="four"),
    ],
)
def test_bounded(case_0):
    assert b(case_0) > 0
    assert b(case_0) < 100
"""
            ),
            context=_LIB,
        ),
        Example(bad=_UNNAMED, good=_UNNAMED, context=_LIB),
    )
