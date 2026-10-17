"""Rules over the pytest configuration as a whole (mcmr's PY-TEST suite group)."""

from ...catalog import Example, Relation, Rule


class TestingTomlTables(Relation):
    """The table headers of each TOML file: the dotted `name`, its `header_line`, and the line of
    the next header (`end_line`) that closes the table."""

    query = r"""
    SELECT path, line AS header_line,
           regexp_extract(text, '^\s*\[\s*([^\[\]]+?)\s*\]\s*(#.*)?$', 1) AS name,
           coalesce(lead(line) OVER (PARTITION BY path ORDER BY line), 2147483647) AS end_line
    FROM source_lines() WHERE path LIKE '%.toml' AND regexp_matches(text, '^\s*\[')
"""


class TestingTomlKeys(Relation):
    """The `key = value` lines of each TOML table: the `key` (a quoted one keeps its quotes), the
    line, the byte column the scalar `value` starts at (a quoted string with its quotes, else up
    to a space or comment) and the whole `text`."""

    query = r"""
    SELECT t.path, t.name AS table_name, l.line,
           regexp_extract(l.text, '^\s*("[^"]*"|[A-Za-z0-9_-]+)\s*=', 1) AS key,
           strlen(regexp_extract(l.text, '^(\s*(?:"[^"]*"|[A-Za-z0-9_-]+)\s*=\s*)', 1))
             AS value_column,
           regexp_extract(l.text, '^\s*(?:"[^"]*"|[A-Za-z0-9_-]+)\s*=\s*'
             || '("[^"]*"|\x27[^\x27]*\x27|[^#\s]+)', 1) AS value,
           l.text
    FROM testing_toml_tables() t
    JOIN source_lines() l ON l.path = t.path AND l.line > t.header_line AND l.line < t.end_line
    WHERE regexp_matches(l.text, '^\s*(?:"[^"]*"|[A-Za-z0-9_-]+)\s*=')
"""


class TestingTomlEnds(Relation):
    """Where a new table is appended to each TOML file: the end position and the `prefix` that
    keeps one blank line before it."""

    query = """
    SELECT path, line, CASE WHEN text = '' THEN 0 ELSE strlen(text) END AS column_,
           CASE WHEN text <> '' THEN chr(10) || chr(10) WHEN coalesce(previous_blank, true)
                THEN '' ELSE chr(10) END AS prefix
    FROM (SELECT path, line, text, lag(trim(text) = '') OVER (PARTITION BY path ORDER BY line)
                 AS previous_blank, max(line) OVER (PARTITION BY path) AS last
          FROM source_lines() WHERE path LIKE '%.toml')
    WHERE line = last
"""


class TestingTomlSet(Relation):
    """The edit that makes `wanted_key = wanted_value` stand in the TOML table `wanted_table` of
    each file that lacks it: the value replaced when the key is there with another one, a line
    after the header when only the key is missing, a new table at the end when the table is.
    `wanted_value` is the TOML literal (`true`, `"strict"`)."""

    wanted_table: str
    wanted_key: str
    wanted_value: str

    query = """
    SELECT path, line AS start_line, value_column AS start_column, line AS end_line,
           value_column + strlen(value) AS end_column, wanted_value AS text
    FROM testing_toml_keys()
    WHERE table_name = wanted_table AND key = wanted_key AND value <> wanted_value
    UNION ALL BY NAME
    SELECT path, header_line + 1 AS start_line, 0 AS start_column, header_line + 1 AS end_line,
           0 AS end_column, wanted_key || ' = ' || wanted_value || chr(10) AS text
    FROM testing_toml_tables() t
    WHERE name = wanted_table AND NOT EXISTS (
      SELECT 1 FROM testing_toml_keys() k
      WHERE k.path = t.path AND k.table_name = wanted_table AND k.key = wanted_key)
    UNION ALL BY NAME
    SELECT path, line AS start_line, column_ AS start_column, line AS end_line,
           column_ AS end_column,
           prefix || '[' || wanted_table || ']' || chr(10)
             || wanted_key || ' = ' || wanted_value || chr(10) AS text
    FROM testing_toml_ends() e
    WHERE NOT EXISTS (SELECT 1 FROM testing_toml_tables() t
                      WHERE t.path = e.path AND t.name = wanted_table)
"""


class TestingSuiteLocal(Relation):
    """The test suites whose effective pytest options come from the manifest itself, so a table
    written there is the one pytest reads (an enclosing manifest's options would be shadowed)."""

    query = """
    SELECT f.fact_id, f.path FROM test_suite_facts f
    JOIN test_suite_evidence e ON e.fact_id = f.fact_id AND e.signal = 'pytest-selection'
    WHERE e.source = f.path
"""


PYTEST_TABLE = "tool.pytest.ini_options"


def pytest_option_fix(key: str, value: str, *, where: str = "TRUE") -> str:
    """The fix SQL setting `key` to the TOML literal `value` in the pytest table of the manifest
    that owns the suite's options, for the findings that satisfy `where`."""
    return f"""
        SELECT f.finding_key, e.path, e.start_line, e.start_column, e.end_line, e.end_column,
               e.text
        FROM finding f JOIN testing_suite_local() l ON l.path = f.path
        JOIN testing_toml_set('{PYTEST_TABLE}', '{key}', '{value}') e ON e.path = f.path
        WHERE {where}
    """


class PytestConfigurationStrictness(Rule, id="PY-TEST0004", applicability="review"):
    """Pytest configurations that leave some of the four strictness controls off."""

    query = """
        WITH controls(control) AS (VALUES ('strict_config'), ('strict_markers'),
                                          ('strict_parametrization_ids'), ('strict_xfail')),
        enabled AS (
          SELECT DISTINCT fact_id, map_key AS control FROM test_suite_values
          WHERE relation = 'strict_controls' AND boolean_value)
        SELECT f.path, f.start_line, f.start_column,
          '`' || f.path || '` turns on '
            || CASE count(e.control) WHEN 1 THEN '1 strictness control'
                    ELSE count(e.control) || ' strictness controls' END
            || ' of the 4 there are, leaving `'
            || string_agg(c.control, '`, `' ORDER BY c.control) FILTER (WHERE e.control IS NULL)
            || '` off' AS message
        FROM test_suite_facts f CROSS JOIN controls c
        LEFT JOIN enabled e ON e.fact_id = f.fact_id AND e.control = c.control
        GROUP BY f.fact_id, f.path, f.start_line, f.start_column HAVING count(e.control) < 4
    """
    # `strict = true` turns all four on (a control written `false` stays off by design). Newly
    # failing tests are the point of review.
    fix = pytest_option_fix("strict", "true")
    examples = (
        Example(
            bad='[tool.pytest.ini_options]\naddopts = "-q"\n',
            good='[tool.pytest.ini_options]\nstrict = true\naddopts = "-q"\n',
        ),
    )


class AsyncRunnerAutoModeConflict(Rule, id="PY-TEST0006", applicability="review"):
    """Configurations that put both anyio and asyncio in auto mode."""

    query = """
        SELECT * FROM test_suite_facts WHERE anyio_mode = 'auto' AND asyncio_mode = 'auto'
    """
    message = "`{fact_id}` satisfies async runner auto mode conflict"
    # Strict asyncio mode asks async tests for their own mark, so none is claimed by both runners.
    fix = pytest_option_fix("asyncio_mode", '"strict"')
    examples = (
        Example(
            bad='[tool.pytest.ini_options]\nanyio_mode = "auto"\nasyncio_mode = "auto"\n',
            good='[tool.pytest.ini_options]\nanyio_mode = "auto"\nasyncio_mode = "strict"\n',
        ),
    )


class CoverageWithoutBranchMeasurement(Rule, id="PY-TEST0007", applicability="review"):
    """Configurations that measure coverage without measuring branches."""

    query = """
        SELECT * FROM test_suite_facts
        WHERE is_coverage_configured AND NOT is_branch_coverage_enabled
    """
    message = "`{fact_id}` satisfies coverage without branch measurement"
    # Written in the manifest's own coverage table (coverage.py reads no enclosing manifest);
    # branch coverage lowers the reported percentage, which a coverage gate may then fail.
    fix = """
        SELECT f.finding_key, e.path, e.start_line, e.start_column, e.end_line, e.end_column,
               e.text
        FROM finding f JOIN testing_toml_set('tool.coverage.run', 'branch', 'true') e
          ON e.path = f.path
    """
    examples = (
        Example(
            bad='[tool.pytest.ini_options]\naddopts = "--cov=pkg"\n',
            good=(
                '[tool.pytest.ini_options]\naddopts = "--cov=pkg"\n\n'
                "[tool.coverage.run]\nbranch = true\n"
            ),
        ),
    )
