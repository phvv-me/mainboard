"""mcmr's TestSuiteFact, derived from the root `pyproject.toml`'s syntax tree.

How strictly the project runs pytest is read from `[tool.pytest.ini_options]` (`addopts` words
and the `strict*` keys) and `[tool.coverage.run]`. mcmr also walks the directories above the root
for a tool-only manifest; the derived fact reads the project's own manifest only. The test
function and case families are in tests_functions.py and tests_cases.py.
"""

from ...catalog import Family


class TestSuite(Family):
    family = "TestSuiteFact"
    verified = True
    relations = {
        "facts": r"""
            WITH RECURSIVE toml AS (
              SELECT * FROM ast_nodes
              WHERE path = 'pyproject.toml'
                AND fact_id IN (SELECT fact_id FROM ast_files WHERE language = 'toml')
            ), pairs AS (
              SELECT p.node_id, CASE WHEN t.type = 'table' THEN t.name || '.' || p.name ELSE
                p.name END
                       AS key
              FROM toml p JOIN toml t ON t.node_id = p.parent_id AND t.type IN ('table',
                'document')
              WHERE p.type = 'pair'
              UNION ALL
              SELECT c.node_id, f.key || '.' || c.name
              FROM pairs f
              JOIN toml i ON i.parent_id = f.node_id AND i.type = 'inline_table'
              JOIN toml c ON c.parent_id = i.node_id AND c.type = 'pair'
            ), settings AS (
              SELECT f.key, v.type, CASE WHEN v.type = 'string' THEN
                       CASE WHEN starts_with(v.name, repeat(chr(34), 3)) OR starts_with(v.name,
                         repeat(chr(39), 3))
                            THEN substr(v.name, 4, length(v.name) - 6)
                            ELSE substr(v.name, 2, length(v.name) - 2) END
                     ELSE v.name END AS text
              FROM pairs f JOIN toml v ON v.parent_id = f.node_id AND v.sibling_index = 2
            ), options AS (
              SELECT coalesce(max(text) FILTER (WHERE key = 'tool.pytest.ini_options.addopts'
                                                  AND type = 'string'), '') AS addopts,
                     bool_or(text = 'true') FILTER (WHERE key = 'tool.pytest.ini_options.strict'
                                                      AND type = 'boolean') AS strict,
                     bool_or(text = 'true') FILTER (WHERE key =
                       'tool.pytest.ini_options.strict_config'
                                                      AND type = 'boolean') AS strict_config,
                     bool_or(text = 'true') FILTER (WHERE key =
                       'tool.pytest.ini_options.strict_markers'
                                                      AND type = 'boolean') AS strict_markers,
                     bool_or(text = 'true') FILTER (
                       WHERE key = 'tool.pytest.ini_options.strict_parametrization_ids'
                         AND type = 'boolean') AS strict_parametrization_ids,
                     bool_or(text = 'true') FILTER (WHERE key =
                       'tool.pytest.ini_options.strict_xfail'
                                                      AND type = 'boolean') AS strict_xfail,
                     coalesce(max(text) FILTER (WHERE key = 'tool.pytest.ini_options.import_mode'
                                                  AND type = 'string'), '') AS import_mode,
                     coalesce(max(text) FILTER (WHERE key = 'tool.pytest.ini_options.anyio_mode'
                                                  AND type = 'string'), '') AS anyio_mode,
                     coalesce(max(text) FILTER (WHERE key = 'tool.pytest.ini_options.asyncio_mode'
                                                  AND type = 'string'), '') AS asyncio_mode,
                     coalesce(bool_or(text = 'true') FILTER (WHERE key = 'tool.coverage.run.branch'
                                                               AND type = 'boolean'), false) AS
                                                                 branch
              FROM (SELECT true AS present FROM toml LIMIT 1) LEFT JOIN settings ON true
              GROUP BY present
            ), flags AS (
              SELECT *,
                     regexp_matches(addopts, '(^|\s)--cov(=\S*)?(\s|$)') AS cov,
                     coalesce(strict, false) OR regexp_matches(addopts,
                       '(^|\s)--strict(=\S*)?(\s|$)')
                       AS strictly,
                     regexp_matches(addopts, '(^|\s)--strict-config(=\S*)?(\s|$)') AS
                       strict_config_flag,
                     regexp_matches(addopts, '(^|\s)--strict-markers(=\S*)?(\s|$)') AS
                       strict_markers_flag,
                     nullif(regexp_extract(addopts, '(^|\s)--import-mode(=|\s+)(\S+)', 3), '')
                       AS import_mode_flag
              FROM options
            )
            SELECT 0 AS fact_order, 'suite:pytest' AS fact_id, 'pyproject.toml' AS path,
                   1 AS start_line, 0 AS start_column, 1 AS end_line, 0 AS end_column,
                   'python' AS language,
                   coalesce(nullif(coalesce(import_mode_flag, import_mode), ''), 'prepend') AS
                     import_mode,
                   anyio_mode, asyncio_mode, cov AS is_coverage_configured,
                   branch AS is_branch_coverage_enabled,
                   coalesce(strict_config, strictly OR strict_config_flag) AS strict_config,
                   coalesce(strict_markers, strictly OR strict_markers_flag) AS strict_markers,
                   coalesce(strict_parametrization_ids, strictly) AS strict_parametrization_ids,
                   coalesce(strict_xfail, strictly) AS strict_xfail
            FROM flags
        """,
        "values": """
            SELECT fact_id, 'strict_controls' AS relation, fact_id AS parent_id,
                   'value' AS entry_kind, c.ordinal, c.map_key, c.boolean_value
            FROM test_suite_facts, LATERAL (VALUES
              (0, 'strict_config', strict_config), (1, 'strict_markers', strict_markers),
              (2, 'strict_parametrization_ids', strict_parametrization_ids),
              (3, 'strict_xfail', strict_xfail)) AS c(ordinal, map_key, boolean_value)
        """,
        "evidence": """
            SELECT fact_id, e.ordinal, e.signal, 'pyproject.toml' AS source, 1.0 AS confidence
            FROM test_suite_facts, LATERAL (VALUES (0, 'configuration:pyproject.toml'),
                                                   (1, 'pytest-selection')) AS e(ordinal, signal)
        """,
    }
