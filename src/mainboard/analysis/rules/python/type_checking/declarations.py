"""Rules over what a project declares about its types and Python target (mcmr's PY-TYPE)."""

from typing import Annotated

from pydantic import Field

from ....catalog import Example, Relation, Rule


class FutureAnnotationsImport(Rule, id="PY-TYPE0001", applicability="review"):
    """`from __future__ import annotations` in a project whose target Python no longer needs it."""

    query = """
        SELECT i.*, d.path AS declaration_path, d.start_line AS declaration_start_line,
               d.start_column AS declaration_start_column, d.end_line AS declaration_end_line,
               d.end_column AS declaration_end_column
        FROM import_binding_facts i LEFT JOIN (
          SELECT fact_id, path, start_line, start_column, end_line, end_column
          FROM import_binding_nodes WHERE role = 'declaration') d USING (fact_id)
        WHERE i.module = '__future__'
          AND CASE WHEN i.imported_name <> '' THEN i.imported_name ELSE i.name END = 'annotations'
          AND (SELECT coalesce(min(python_target_project_minimum_minor), 14)
               FROM project_configuration_facts) >= 14
    """
    message = "`{fact_id}` satisfies future annotations import"
    # The statement's line goes, with the blank line after it when a blank line (or the top of
    # the file) already precedes it, so no double gap is left. An import that shares its line
    # with another statement or names other features stays for a human.
    fix = r"""
        SELECT finding_key, declaration_path AS path, declaration_start_line AS start_line,
               0 AS start_column,
               declaration_end_line + 1 + (blank_before AND blank_after)::INT AS end_line,
               0 AS end_column, '' AS text
        FROM (
          SELECT f.*, coalesce(trim(p.text) = '', true) AS blank_before,
                 coalesce(trim(n.text) = '' AND n.line < last.line, false) AS blank_after
          FROM finding f
          JOIN source_lines() l
            ON l.path = f.declaration_path AND l.line = f.declaration_start_line
          LEFT JOIN source_lines() p
            ON p.path = f.declaration_path AND p.line = f.declaration_start_line - 1
          LEFT JOIN source_lines() n
            ON n.path = f.declaration_path AND n.line = f.declaration_end_line + 1
          JOIN (SELECT path, max(line) AS line FROM source_lines() GROUP BY path) last
            ON last.path = f.declaration_path
          WHERE f.declaration_start_line = f.declaration_end_line
            AND regexp_matches(l.text, '^from\s+__future__\s+import\s+annotations\s*(#.*)?$'))
    """
    examples = (
        Example(
            bad="from __future__ import annotations\n\nimport os\n",
            good="import os\n",
        ),
    )


class NullableBooleanAnnotation(Rule, id="PY-TYPE0002"):
    """`bool | None` annotations, a three-state value hiding in a flag."""

    query = r"""
        WITH spelled AS (
          SELECT a.*, syntax_slice(s.content, a.start_line, a.start_column, a.end_line,
                                   a.end_column) AS node_text
          FROM core_annotation a JOIN source_files s USING (path)
        )
        SELECT path, start_line, start_column, node_text
        FROM spelled a
        WHERE regexp_matches(node_text, '^(bool|None)(\s*\|\s*(bool|None))+$')
          AND contains(node_text, 'bool') AND contains(node_text, 'None')
          AND NOT (role IN ('parameter', 'return')
                   AND EXISTS (SELECT 1 FROM core_decor d
                               WHERE d.file_id = a.file_id AND d.def_id = a.owner
                                 AND regexp_extract(d.name, '[^.]*$') IN ('command', 'callback')))
    """
    message = "nullable Boolean annotation `{node_text}`"


class TypeCheckingPythonTargetEdits(Relation):
    """The edits aligning each project's Python declarations on one target minor: its
    `requires-python` raised to `minimum_minor` when lower (or written when absent from an
    existing `[project]` table), then every tool's target, and every per-file ruff target, set to
    the project's minimum. A tool target whose value is not a quoted string (black's list) cannot
    be edited and stays."""

    minimum_minor: int

    query = r"""
    WITH facts AS (
      SELECT fact_id, path, python_target_project_minimum_minor AS project_minor,
             greatest(coalesce(python_target_project_minimum_minor, 0), minimum_minor) AS target
      FROM project_configuration_facts
    ), tools AS (
      SELECT f.fact_id, f.path, f.target, v.string_value AS tool, m.integer_value AS current
      FROM facts f
      JOIN project_configuration_values v
        ON v.fact_id = f.fact_id AND v.relation = 'python_target.configured_tools'
      LEFT JOIN project_configuration_values m ON m.fact_id = f.fact_id
        AND m.relation = 'python_target.tool_target_minors' AND m.map_key = v.string_value
    ), specs(tool, table_name, key) AS (
      VALUES ('ruff', 'tool.ruff', 'target-version'), ('mypy', 'tool.mypy', 'python_version'),
             ('pyright', 'tool.pyright', 'pythonVersion'),
             ('basedpyright', 'tool.basedpyright', 'pythonVersion'),
             ('pyrefly', 'tool.pyrefly', 'python-version'),
             ('ty', 'tool.ty.environment', 'python-version')
    ), targets AS (
      SELECT t.fact_id, k.path, k.line, k.value_column, k.value,
             t.target, t.current
      FROM tools t JOIN testing_toml_keys() k ON k.path = t.path
        AND k.table_name IN ('tool.' || t.tool, 'tool.' || t.tool || '.environment')
        AND k.key IN ('target-version', 'python_version', 'python-version', 'pythonVersion')
      WHERE t.current IS DISTINCT FROM t.target
    )
    SELECT fact_id, path, line AS start_line, value_column AS start_column, line AS end_line,
           value_column + strlen(value) AS end_column,
           CASE WHEN regexp_matches(value, 'py3[0-9]+')
                THEN regexp_replace(value, 'py3[0-9]+', 'py3' || target)
                ELSE regexp_replace(value, '3[.][0-9]+', '3.' || target) END AS text
    FROM targets
    WHERE regexp_matches(value,
      '^(?:"(?:py3[0-9]+|3[.][0-9]+)"|\x27(?:py3[0-9]+|3[.][0-9]+)\x27)$')
    UNION ALL BY NAME
    SELECT t.fact_id, tab.path, tab.header_line + 1 AS start_line, 0 AS start_column,
           tab.header_line + 1 AS end_line, 0 AS end_column,
           s.key || ' = "' || CASE WHEN t.tool = 'ruff' THEN 'py3' ELSE '3.' END || t.target
             || '"' || chr(10) AS text
    FROM tools t JOIN specs s ON s.tool = t.tool
    JOIN testing_toml_tables() tab ON tab.path = t.path AND tab.name = s.table_name
    WHERE t.current IS NULL AND NOT EXISTS (
      SELECT 1 FROM testing_toml_keys() k WHERE k.path = t.path
        AND k.table_name IN ('tool.' || t.tool, 'tool.' || t.tool || '.environment')
        AND k.key IN ('target-version', 'python_version', 'python-version', 'pythonVersion'))
    UNION ALL BY NAME
    SELECT t.fact_id, e.path, e.line AS start_line, e.column_ AS start_column, e.line AS end_line,
           e.column_ AS end_column,
           e.prefix || '[' || s.table_name || ']' || chr(10) || s.key || ' = "'
             || CASE WHEN t.tool = 'ruff' THEN 'py3' ELSE '3.' END || t.target || '"' || chr(10)
             AS text
    FROM tools t JOIN specs s ON s.tool = t.tool JOIN testing_toml_ends() e ON e.path = t.path
    WHERE t.current IS NULL AND NOT EXISTS (
      SELECT 1 FROM testing_toml_tables() tab WHERE tab.path = t.path AND tab.name = s.table_name)
      AND NOT EXISTS (
      SELECT 1 FROM testing_toml_keys() k WHERE k.path = t.path
        AND k.table_name IN ('tool.' || t.tool, 'tool.' || t.tool || '.environment')
        AND k.key IN ('target-version', 'python_version', 'python-version', 'pythonVersion'))
    UNION ALL BY NAME
    SELECT f.fact_id, k.path, k.line AS start_line, k.value_column AS start_column,
           k.line AS end_line, k.value_column + strlen(k.value) AS end_column,
           regexp_replace(k.value, 'py3[0-9]+', 'py3' || f.target) AS text
    FROM facts f JOIN testing_toml_keys() k
      ON k.path = f.path AND k.table_name = 'tool.ruff.per-file-target-version'
    WHERE regexp_matches(k.value, '^(?:"py3[0-9]+"|\x27py3[0-9]+\x27)$')
      AND CAST(regexp_extract(k.value, 'py3([0-9]+)', 1) AS INTEGER) < f.target
    UNION ALL BY NAME
    SELECT f.fact_id, k.path, k.line AS start_line, k.value_column AS start_column,
           k.line AS end_line, k.value_column + strlen(k.value) AS end_column,
           regexp_replace(k.value, '(>=\s*)3[.][0-9]+', '\1' || '3.' || minimum_minor) AS text
    FROM facts f JOIN testing_toml_keys() k
      ON k.path = f.path AND k.table_name = 'project' AND k.key = 'requires-python'
    WHERE coalesce(f.project_minor, 0) < minimum_minor
      AND regexp_matches(k.value,
        '^(?:"[^"]*>=\s*3[.][0-9]+[^"]*"|\x27[^\x27]*>=\s*3[.][0-9]+[^\x27]*\x27)$')
    UNION ALL BY NAME
    SELECT f.fact_id, tab.path, tab.header_line + 1 AS start_line, 0 AS start_column,
           tab.header_line + 1 AS end_line, 0 AS end_column,
           'requires-python = ">=3.' || minimum_minor || '"' || chr(10) AS text
    FROM facts f JOIN testing_toml_tables() tab ON tab.path = f.path AND tab.name = 'project'
    WHERE f.project_minor IS NULL AND NOT EXISTS (
      SELECT 1 FROM testing_toml_keys() k
      WHERE k.path = f.path AND k.table_name = 'project' AND k.key = 'requires-python')
"""


class MinimumPythonDeclaration(Rule, id="PY-TYPE0003", applicability="review"):
    """Python version declarations that disagree with each other or with the minimum."""

    minimum_version: Annotated[str, Field(pattern=r"^3\.\d+$")] = "3.14"

    query = """
        WITH issues AS (
          SELECT t.fact_id FROM project_configuration_values t
          JOIN project_configuration_facts f USING (fact_id)
          LEFT JOIN project_configuration_values m ON m.fact_id = t.fact_id
            AND m.relation = 'python_target.tool_target_minors' AND m.map_key = t.string_value
          WHERE t.relation = 'python_target.configured_tools'
            AND (m.integer_value IS NULL OR (f.python_target_project_minimum_minor IS NOT NULL
                 AND m.integer_value <> f.python_target_project_minimum_minor))
          UNION ALL
          SELECT v.fact_id FROM project_configuration_values v
          JOIN project_configuration_facts f USING (fact_id)
          WHERE v.relation = 'python_target.per_file_target_minors'
            AND (f.python_target_project_minimum_minor IS NULL
                 OR v.integer_value < f.python_target_project_minimum_minor)
        )
        SELECT f.*, (f.python_target_project_minimum_minor IS NULL)::INT
          + coalesce(f.python_target_project_minimum_minor
                     < CAST(split_part($minimum_version, '.', 2) AS INTEGER), false)::INT
          + coalesce(n, 0) AS value
        FROM project_configuration_facts f
        LEFT JOIN (SELECT fact_id, count(*) AS n FROM issues GROUP BY fact_id) USING (fact_id)
        WHERE value > 0
    """
    message = "minimum python declaration is {value} for `{fact_id}`"
    # Raising the target changes what the type checkers accept, so the fix is for review.
    fix = """
        SELECT f.finding_key, e.path, e.start_line, e.start_column, e.end_line, e.end_column,
               e.text
        FROM finding f
        JOIN type_checking_python_target_edits(
          CAST(split_part($minimum_version, '.', 2) AS INTEGER)) e ON e.fact_id = f.fact_id
    """
    examples = (
        Example(
            bad='[project]\nrequires-python = ">=3.12"\n\n[tool.ruff]\ntarget-version = "py312"\n',
            good=(
                '[project]\nrequires-python = ">=3.14"\n\n[tool.ruff]\ntarget-version = "py314"\n'
            ),
        ),
    )
