"""mcmr's TestFunctionFact, derived from sitting_duck's Python trees.

The family keeps the tests a runner collects: module-level ones and the methods of a `Test` class
without an initializer. The helper relations (`tests_bodies`, `tests_behavior`, `tests_reach`)
come first; the case-group family reads them too, so a run derives both (TestCaseGroupFact names
this family's relations).
"""

from ...catalog import Family
from .tests_behavior import BEHAVIOR
from .tests_bodies import BODIES
from .tests_reach import REACH

_FACT = "'testfunctionfact:' || t.path"
_RECORD = f"{_FACT} || '/tests:' || t.ordinal"


class TestFunction(Family):
    family = "TestFunctionFact"
    verified = True
    relations = {
        "facts": """
            SELECT row_number() OVER (ORDER BY f.path) - 1 AS fact_order,
                   'testfunctionfact:' || f.path AS fact_id, f.path, 1 AS start_line,
                   0 AS start_column, m.end_line, 0 AS end_column, 'python' AS language
            FROM ast_files f JOIN ast_nodes m ON m.fact_id = f.fact_id AND m.parent_id IS NULL
            WHERE f.language = 'python' AND classes_is_test(f.path)
        """,
        **BODIES,
        **BEHAVIOR,
        **REACH,
        "entries": """
            SELECT fact_id, fn AS test, 'tests.requested_fixture_names' AS relation, ordinal,
                   name AS string_value, NULL::BIGINT AS integer_value
            FROM test_function_params
            UNION ALL
            SELECT fact_id, test, 'tests.fixture_names',
                   row_number() OVER (PARTITION BY fact_id, test ORDER BY name) - 1, name, NULL
            FROM test_function_reached
            UNION ALL
            SELECT fact_id, fn, 'tests.marks', ordinal, mark, NULL FROM test_function_decorators
            UNION ALL
            SELECT fact_id, test, 'tests.literal_values', generate_subscripts(literals, 1) - 1,
                   unnest(literals), NULL
            FROM test_function_shapes
            UNION ALL
            SELECT fact_id, test, 'tests.parametrized_range_sizes',
                   row_number() OVER (PARTITION BY fact_id, test ORDER BY decorator) - 1, NULL,
                     size
            FROM test_function_sizes WHERE size >= 0
            UNION ALL
            SELECT fact_id, test, 'tests.direct_targets',
                   row_number() OVER (PARTITION BY fact_id, test ORDER BY name) - 1, name, NULL
            FROM (SELECT DISTINCT c.fact_id, c.test, n.name
                  FROM test_function_called c
                  JOIN test_function_named n ON n.path = c.callee_path AND n.node_id = c.callee
                  WHERE NOT c.is_test)
            UNION ALL
            SELECT fact_id, test, 'tests.reachable_targets',
                   row_number() OVER (PARTITION BY fact_id, test ORDER BY name) - 1, name, NULL
            FROM (SELECT DISTINCT c.fact_id, c.test, n.name
                  FROM test_function_called c
                  JOIN test_function_closure w ON w.root_path = c.callee_path AND w.root = c.callee
                  JOIN test_function_named n ON n.path = w.path AND n.node_id = w.node_id)
        """,
        "tests": f"""
            SELECT {_FACT} AS fact_id, {_FACT} AS parent_id, {_RECORD} AS record_id, f.fact_order,
                   t.ordinal, t.name, t.path, t.is_async, t.collected AS is_collected,
                   t.path AS node_span_path, t.start_line AS node_span_start_line,
                   t.start_column AS node_span_start_column, t.end_line AS node_span_end_line,
                   t.end_column AS node_span_end_column, coalesce(s.body_shape, '') AS body_shape,
                   c.owned_conditional_count, c.owned_statement_count,
                   c.module_state_mutation_count, coalesce(g.n, 0) AS
                     generated_parametrization_count
            FROM test_function_declared t
            JOIN test_function_facts f ON f.path = t.path
            JOIN test_function_counts c ON c.fact_id = t.fact_id AND c.test = t.test
            LEFT JOIN test_function_shapes s ON s.fact_id = t.fact_id AND s.test = t.test
            LEFT JOIN test_function_generations g ON g.fact_id = t.fact_id AND g.test = t.test
        """,
        "tests.calls": f"""
            SELECT {_FACT} AS fact_id, {_RECORD} AS parent_id,
                   {_RECORD} || '/tests.calls:' || c.ordinal AS record_id, c.ordinal, t.path,
                   c.qualified_name
            FROM test_function_calls c
            JOIN test_function_declared t ON t.fact_id = c.fact_id AND t.test = c.test
        """,
        "quarantined_tests": f"""
            SELECT {_FACT} AS fact_id, {_FACT} AS parent_id,
                   {_FACT} || '/quarantined_tests:'
                     || (row_number() OVER (PARTITION BY q.fact_id ORDER BY q.test, q.ordinal) - 1)
                     AS record_id,
                   q.name, q.age_days, q.owner, q.has_remediation_evidence, q.recurred_after_repair
            FROM test_function_quarantines q
            JOIN test_function_declared t ON t.fact_id = q.fact_id AND t.test = q.test
        """,
        "values": f"""
            SELECT {_FACT} AS fact_id, e.relation, {_RECORD} AS parent_id, 'value' AS entry_kind,
                   e.ordinal, e.string_value, e.integer_value
            FROM test_function_entries e
            JOIN test_function_declared t ON t.fact_id = e.fact_id AND t.test = e.test
        """,
    }
