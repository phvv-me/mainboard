"""Rules over comprehensions (mcmr's PY-COMP family)."""

from ....catalog import Example, Rule


class ComprehensionLoopCount(Rule, id="PY-COMP0001"):
    """Modules holding a comprehension with more loops than the project accepts."""

    maximum: int = 2

    query = """
            SELECT f.path, 1 AS start_line, 0 AS start_column,
                   'comprehensionfact:' || f.path AS fact_id, max(n) AS value
            FROM (SELECT c.file_id, count(*) AS n
                  FROM core_node c
                  JOIN core_node k ON k.file_id = c.file_id AND k.parent_id = c.node_id
                                  AND k.type = 'for_in_clause'
                  WHERE c.type IN ('list_comprehension', 'set_comprehension',
                                   'dictionary_comprehension', 'generator_expression')
                  GROUP BY c.file_id, c.node_id) x
            JOIN core_file f USING (file_id)
            WHERE f.language = 'python'
            GROUP BY f.path HAVING max(n) > $maximum
        """
    message = "comprehension loop count is {value} for `{fact_id}`"


class ManualSetComprehension(Rule, id="PY-COMP0002", applicability="safe"):
    """Fresh sets populated by a loop that can be one set comprehension."""

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM comprehension_set_loop_candidates c JOIN comprehension_facts f USING (fact_id)
            WHERE c.has_unshadowed_set_initialization AND c.loop_is_synchronous
              AND c.only_effect_is_add AND c.conditional_count <= 1 AND NOT c.has_else
            GROUP BY ALL
        """
    message = "manual set comprehension is {value} for `{fact_id}`"
    # `values = set()` and the `for` loop that follows it (nothing but blank lines between)
    # become `values = {element for target in iterable if condition}`, one edit per loop. Left
    # alone: a loop inside no function (its target may be read later at module level), one whose
    # target a later statement of its function reads (a comprehension keeps it to itself), one
    # reading the set it fills, and an iterable that needs parentheses inside a comprehension.
    fix = r"""
        WITH candidates AS (
          SELECT f.path, hash(c.record_id) AS finding_key, c.fact_id, c.name, c.element_text,
                 c.target_text, c.iterable_text, k.text AS condition_text,
                 c.initialization_span_start_line AS first_line,
                 c.initialization_span_start_column AS first_column,
                 c.initialization_span_end_line AS init_end_line,
                 c.loop_span_start_line AS loop_start_line,
                 c.loop_span_end_line AS last_line, c.loop_span_end_column AS last_column
          FROM finding f
          JOIN comprehension_set_loop_candidates c ON c.fact_id = f.fact_id
          LEFT JOIN comprehension_set_loop_candidates_conditions k ON k.parent_id = c.record_id
          WHERE c.has_unshadowed_set_initialization AND c.loop_is_synchronous
            AND c.only_effect_is_add AND c.conditional_count <= 1 AND NOT c.has_else
        ), checked AS (
          SELECT *, init_end_line < loop_start_line AND init_end_line = first_line
            AND NOT EXISTS (
              SELECT 1 FROM source_lines() l
              WHERE l.path = c.path AND l.line > c.init_end_line AND l.line < c.loop_start_line
                AND trimmed(l.text) <> '')
            AND NOT regexp_matches(iterable_text, '\b(if|else|lambda)\b|^[^()\[\]{}]*,')
            AND NOT regexp_matches(coalesce(condition_text, '') || ' ' || element_text || ' '
                                   || iterable_text, '\b' || name || '\b')
            AND EXISTS (
              SELECT 1 FROM syntax_facts s
              WHERE s.path = c.path AND s.kind = 'callable'
                AND s.start_line <= c.first_line AND s.end_line >= c.last_line)
            AND NOT EXISTS (
              SELECT 1 FROM syntax_facts s JOIN syntax_nodes n ON n.fact_id = s.fact_id
              WHERE s.path = c.path AND s.kind = 'callable'
                AND s.start_line <= c.first_line AND s.end_line >= c.last_line
                AND n.kind = 'name' AND n.start_line > c.last_line
                AND list_contains(regexp_extract_all(c.target_text, '[A-Za-z_][A-Za-z0-9_]*'),
                                  n.name)) AS rewritable
          FROM candidates c
        )
        SELECT finding_key, path, first_line AS start_line, first_column AS start_column,
               last_line AS end_line, last_column AS end_column,
               name || ' = {' || element_text || ' for ' || target_text || ' in ' || iterable_text
                 || coalesce(' if ' || condition_text, '') || '}' AS text
        FROM checked WHERE rewritable
    """
    examples = (
        Example(
            bad=(
                """def doubled(items):
    seen = set()
    for item in items:
        seen.add(item * 2)
    return seen
"""
            ),
            good="def doubled(items):\n    seen = {item * 2 for item in items}\n    return seen\n",
        ),
    )
