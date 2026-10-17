"""Rules over tests and test modules (mcmr's ALL-TEST family)."""

from ...catalog import Rule


class FlakyTestQuarantineDebt(Rule, id="ALL-TEST0002"):
    """Quarantined flaky tests without timely owned remediation."""

    maximum_age_days: int = 14
    require_owner: bool = True

    query = """
        SELECT f.path, f.start_line, f.start_column, f.fact_id, v.value
        FROM test_function_facts f
        JOIN (SELECT fact_id, count(*) AS value FROM test_function_quarantined_tests
              WHERE age_days IS NULL OR age_days > $maximum_age_days
                 OR ($require_owner AND owner = '') OR recurred_after_repair
                 OR NOT has_remediation_evidence
              GROUP BY fact_id) v USING (fact_id)
    """
    message = "flaky test quarantine debt is {value} for `{fact_id}`"


class TestModuleMemberCount(Rule, id="ALL-TEST0003"):
    """Top-level helpers, classes and tests of a test module as one suite navigation width."""

    maximum: int = 20

    query = """
        SELECT path, start_line, start_column, fact_id,
               CAST(class_count + function_count AS DOUBLE) AS value
        FROM module_facts
        WHERE CASE WHEN is_test THEN class_count + function_count ELSE 0 END > $maximum
    """
    message = "test module member count is {value} for `{fact_id}`"
