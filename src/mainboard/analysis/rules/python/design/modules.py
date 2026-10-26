"""Rules over barrels, package initializers and explicit exports (mcmr's PY-MODU family)."""

from ....catalog import Example, Rule


class NonInitReexportModule(Rule, id="PY-MODU0001"):
    """Barrel modules that define no behavior beyond re-exporting imports."""

    query = """
        SELECT * FROM module_facts
        WHERE has_only_imports_and_all AND NOT is_package_initializer
          AND NOT ends_with(path, 'conftest.py')
    """
    message = "`{fact_id}` satisfies non init reexport module"


class EmptyPackageInitializer(Rule, id="PY-MODU0002", applicability="review"):
    """Package initializers that state no executable package surface."""

    query = """
        SELECT * FROM module_facts
        WHERE is_package_initializer AND NOT is_test AND executable_statement_count = 0
    """
    message = "`{fact_id}` satisfies empty package initializer"
    examples = (
        Example(
            bad="\n",
            good=None,
            path="src/pkg/blank/__init__.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": '"""The package."""\n',
                "src/pkg/blank/mod.py": "x = 1\n",
            },
        ),
    )
    # The blank initializer goes and its package becomes a namespace package. One holding a
    # docstring or comments says something, so it stays.
    fix = """
        SELECT path, 'remove' AS operation FROM finding
        WHERE EXISTS (SELECT 1 FROM source_files s
                      WHERE s.path = finding.path AND trimmed(s.content) = '')
    """


class InitializerDeclaration(Rule, id="PY-MODU0003"):
    """Ordinary function and class declarations in package initializers."""

    query = """
        SELECT f.*, c.value FROM module_facts f
        JOIN (SELECT fact_id, count(*) AS value FROM module_members
              WHERE kind = 'class'
                 OR (kind = 'function' AND name NOT IN ('__getattr__', '__dir__'))
              GROUP BY fact_id) c ON c.fact_id = f.fact_id
        WHERE f.is_package_initializer
    """
    message = "initializer declaration is {value} for `{fact_id}`"


class ExplicitAllOnlyInInitializer(Rule, id="PY-MODU0004", applicability="review"):
    """Explicit package export lists outside package initializers."""

    query = """
        SELECT f.*, d.declarations FROM module_facts f
        LEFT JOIN (
          SELECT fact_id, list({'path': span_path, 'start_line': span_start_line,
                                'start_column': span_start_column, 'end_line': span_end_line,
                                'end_column': span_end_column} ORDER BY ordinal) AS declarations
          FROM module_all_declarations GROUP BY fact_id) d ON d.fact_id = f.fact_id
        WHERE f.declares_all AND NOT f.is_package_initializer
    """
    message = "`{fact_id}` satisfies explicit all outside initializer"
    examples = (
        Example(
            bad='__all__ = ["tool"]\n\n\ndef tool():\n    return 1\n',
            good="def tool():\n    return 1\n",
        ),
    )
    # Every `__all__` statement of the module goes, together or not at all. A module whose
    # imports re-export through it is a barrel: deleting the list would change what it exports.
    fix = """
        WITH declared AS (
          SELECT finding_key, path AS module_path, unnest(declarations, recursive := true)
          FROM finding
        )
        SELECT d.finding_key, d.path,
               imports_tidy_start(d.start_line, first_gap.previous_line, last_gap.final)
                 AS start_line,
               0 AS start_column,
               imports_tidy_end(d.start_line, d.end_line, first_gap.previous_line,
                                last_gap.next_line) AS end_line,
               0 AS end_column,
               CASE WHEN bool_and(
                      trimmed(imports_before(opening.text, d.start_column::BIGINT)) = ''
                      AND regexp_matches(imports_from(closing.text, d.end_column::BIGINT),
                                         '^\\s*(#.*)?$')
                      AND NOT coalesce(imports_empties_block(
                        opening.previous_text, opening.text, closing.following_text), false))
                      OVER (PARTITION BY d.finding_key)
                     AND NOT EXISTS (SELECT 1 FROM import_binding_facts i
                                     WHERE i.path = d.module_path AND i.is_reexported)
                    THEN '' END AS text
        FROM declared d
        LEFT JOIN imports_line_gaps() first_gap
          ON first_gap.path = d.path AND first_gap.line = d.start_line
        LEFT JOIN imports_line_gaps() last_gap
          ON last_gap.path = d.path AND last_gap.line = d.end_line
        LEFT JOIN imports_code_lines() opening
          ON opening.path = d.path AND opening.line = d.start_line
        LEFT JOIN imports_code_lines() closing
          ON closing.path = d.path AND closing.line = d.end_line
    """


class UnusedExplicitExport(Rule, id="PY-MODU0005", applicability="review"):
    """Explicit package exports no repository consumer uses. The initializers of the package a
    `pyproject.toml` publishes (its `[project]` name) are left out: their exports are for the
    projects that install it, which the repository does not hold."""

    query = r"""
        SELECT e.*, (SELECT min(n.id) FROM export_public_export_nodes n
                     WHERE n.fact_id = e.fact_id) AS member_id
        FROM export_facts e
        WHERE public_export_consumer_count = 0
          AND NOT EXISTS (
            SELECT 1 FROM source_files t
            WHERE ends_with(t.path, 'pyproject.toml') AND regexp_matches(e.path, '__init__\.pyi?$')
              AND starts_with(e.path, regexp_replace(t.path, 'pyproject\.toml$', ''))
              AND regexp_matches(t.content, '(?ms)^\[project\]\s*$.*?^name\s*=\s*"[^"]+"')
              AND replace(lower(regexp_extract(t.content,
                    '(?ms)^\[project\]\s*$.*?^name\s*=\s*"([^"]+)"', 1)), '-', '_')
                  = split_part(e.public_export_module, '.', 1))
    """
    message = (
        "`{public_export_name}` explicitly exports `{public_export_target}`, but no repository "
        "consumer uses that public route"
    )
    examples = (
        Example(
            bad='from .tool import tool\n\n__all__ = ["tool"]\n',
            good="",
        ),
    )
    # The item leaves `__all__` (the whole statement when it was the last), and with it the
    # import that only re-exported the name, since that import would be PY-IMPO0003's next
    # finding: together or not at all. A `conftest.py` is skipped (pytest finds its hooks and
    # fixtures by name, not by import), as is an import that states its re-export as `x as x`.
    fix = """
        WITH reexports AS (
          SELECT f.finding_key, i.binding_id AS member_id
          FROM finding f
          JOIN import_binding_facts i ON i.path = f.path AND i.name = f.public_export_name
          JOIN import_binding_nodes b
            ON b.fact_id = i.fact_id AND b.node_id = i.binding_id AND b.role = 'binding'
          WHERE i.reference_count = 0 AND NOT (i.has_qualifying_use OR i.has_documented_side_effect
            OR i.is_wildcard OR i.module = '__future__')
            AND b.text <> i.name || ' as ' || i.name
        ), candidates AS (
          SELECT finding_key, member_id FROM finding
          WHERE member_id IS NOT NULL AND NOT ends_with(path, 'conftest.py')
          UNION ALL
          SELECT finding_key, member_id FROM reexports
          WHERE finding_key IN (SELECT finding_key FROM finding
                                WHERE NOT ends_with(path, 'conftest.py'))
        ), tried AS (
          SELECT * FROM imports_member_removals((SELECT list(member_id) FROM candidates))
        ), chosen AS (
          SELECT finding_key, member_id FROM candidates
          WHERE finding_key IN (
            SELECT c.finding_key FROM candidates c LEFT JOIN tried t USING (member_id)
            GROUP BY c.finding_key HAVING bool_and(coalesce(t.feasible, false)))
        )
        SELECT c.finding_key, r.path, r.start_line, r.start_column, r.end_line, r.end_column,
               '' AS text
        FROM chosen c
        JOIN imports_member_removals((SELECT list(member_id) FROM chosen)) r USING (member_id)
        WHERE r.start_line IS NOT NULL
    """
