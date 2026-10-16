"""Rules over unreachable code (mcmr's PY-DEAD family)."""

from ...catalog import Example, Rule


class UnreferencedPrivateFunction(Rule, id="PY-DEAD0001", applicability="review"):
    """Undecorated private module functions nothing in their module calls."""

    query = """
            SELECT * FROM function_functions f
            WHERE scope = 'module' AND visibility <> 'public'
              AND reference_count = CAST(is_recursive AS UBIGINT)
              AND NOT EXISTS (SELECT 1 FROM function_decorators d
                              WHERE d.function_id = f.entity_id)
        """
    message = "`{fact_id}` satisfies unreferenced private function"
    # The definition's lines go with the blank lines that follow it (at the end of a file, the
    # ones before it), unless another module imports or reaches the name.
    fix = """
        SELECT finding_key, definition_path AS path,
               if(next_code IS NULL, coalesce(previous_code, 0) + 1, definition_start_line)
                 AS start_line,
               0 AS start_column, coalesce(next_code, definition_end_line + 1) AS end_line,
               0 AS end_column, '' AS text
        FROM (
          SELECT f.*,
                 (SELECT min(line) FROM source_lines() s WHERE s.path = f.definition_path
                    AND s.line > f.definition_end_line AND trimmed(s.text) <> '') AS next_code,
                 (SELECT max(line) FROM source_lines() s WHERE s.path = f.definition_path
                    AND s.line < f.definition_start_line AND trimmed(s.text) <> '')
                   AS previous_code
          FROM finding f
          WHERE NOT EXISTS (SELECT 1 FROM import_binding_facts i WHERE i.imported_name = f.name)
            AND NOT EXISTS (SELECT 1 FROM core_ref a
                            WHERE a.parent_type = 'attribute' AND a.place = 1
                              AND a.name = f.name))
    """
    examples = (
        Example(
            bad="def _unused(value):\n    return value\n\n\ndef used():\n    return 1\n",
            good="def used():\n    return 1\n",
        ),
    )
