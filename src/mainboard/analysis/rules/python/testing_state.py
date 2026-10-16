"""Rules over what tests mutate and branch on (mcmr's PY-TEST state group)."""

from ...catalog import Rule


class DirectSharedTestStateMutationCount(Rule, id="PY-TEST0008"):
    """Collected tests that mutate module-level state directly."""

    query = """
        SELECT f.*, n AS value FROM test_function_facts f JOIN (
          SELECT fact_id, sum(module_state_mutation_count) AS n FROM test_function_tests
          WHERE is_collected GROUP BY fact_id HAVING n > 0) USING (fact_id)
    """
    message = "direct shared test state mutation count is {value} for `{fact_id}`"


class ConditionalTestBranchCount(Rule, id="PY-TEST0011"):
    """Tests that branch on conditionals in their own body."""

    maximum: int = 2

    query = """
        SELECT fact_id, node_span_path AS path, node_span_start_line AS start_line,
               node_span_start_column AS start_column, owned_conditional_count AS value
        FROM test_function_tests WHERE is_collected AND owned_conditional_count > $maximum
        QUALIFY row_number() OVER (PARTITION BY fact_id
                                   ORDER BY owned_conditional_count DESC, ordinal) = 1
    """
    message = "conditional test branch count is {value} for `{fact_id}`"
