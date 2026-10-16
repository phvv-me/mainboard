"""Rules over tests that repeat each other's intent (mcmr's PY-TEST redundancy group)."""

from ...catalog import Relation, Rule

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


class DuplicateTestIntentClusterCount(Rule, id="PY-TEST0015"):
    """Tests with the same body shape, literals and fixtures reaching the same production code."""

    minimum_tests: int = 2

    query = f"""
        SELECT {_CLUSTER} FROM pytest_reaching()
        GROUP BY body_shape, literal_values, fixture_names, reachable_targets
        HAVING test_count >= $minimum_tests
    """
    message = "duplicate test intent reaches the same production graph in `{test_names}`"


class ConcentratedTestReachClusterCount(Rule, id="PY-TEST0016"):
    """Many tests of few intents reaching the same production code through the same fixtures."""

    minimum_tests: int = 4

    query = f"""
        SELECT {_CLUSTER}, {_INTENTS} FROM pytest_reaching()
        GROUP BY fixture_names, reachable_targets
        HAVING test_count >= $minimum_tests AND intent_count > 1 AND intent_count * 2 <= test_count
    """
    message = "repeated test intent concentrates the same production reach across `{test_names}`"


class ProductionReachHotspotCount(Rule, id="PY-TEST0019"):
    """Production code that many tests of few intents reach."""

    minimum_tests: int = 8

    query = f"""
        SELECT {_CLUSTER}, {_INTENTS}, reachable FROM (
          SELECT {_TEST}, v.string_value AS reachable
          FROM test_function_tests t JOIN test_function_values v
            ON v.fact_id = t.fact_id AND v.parent_id = t.record_id
          WHERE t.is_collected AND v.relation = 'tests.reachable_targets'
            AND v.entry_kind = 'value')
        GROUP BY reachable HAVING test_count >= $minimum_tests AND intent_count * 2 <= test_count
    """
    message = "low-diversity tests repeatedly reach `{reachable}` through `{test_names}`"
