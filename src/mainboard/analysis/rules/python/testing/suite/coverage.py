"""Rules over test size and the cases a test covers (mcmr's PY-TEST coverage group)."""

from .....catalog import Rule


class OwnedTestStatementCount(Rule, id="PY-TEST0012"):
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


class FiniteRangeHypothesisCandidateCount(Rule, id="PY-TEST0014"):
    """Parametrized finite ranges wide enough for a Hypothesis strategy."""

    minimum_cases: int = 10

    query = """
        SELECT f.*, n AS value FROM test_function_facts f JOIN (
          SELECT fact_id, count(*) AS n FROM test_function_values
          WHERE relation = 'tests.parametrized_range_sizes' AND integer_value >= $minimum_cases
            AND (fact_id, parent_id) IN
                (SELECT fact_id, record_id FROM test_function_tests WHERE is_collected)
          GROUP BY fact_id) USING (fact_id)
    """
    message = "finite range hypothesis candidate count is {value} for `{fact_id}`"
