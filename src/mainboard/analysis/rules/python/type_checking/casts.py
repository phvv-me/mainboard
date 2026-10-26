"""Rules over `typing.cast` and conversions the types already prove (mcmr's PY-TYPE casts)."""

from ....catalog import Example, Rule

_ARGUMENTS = """
    SELECT call_id, count(*) AS argument_count,
      max(text) FILTER (root_ordinal = 0) AS first_text,
      max(resolved_type) FILTER (root_ordinal = 0) AS first_type,
      max(node_text) FILTER (root_ordinal = 0) AS first_source,
      max(text) FILTER (root_ordinal = 1) AS second_text,
      max(qualified_name) FILTER (root_ordinal = 1) AS second_name
    FROM call_expressions
    WHERE root_relation = 'argument' AND depth = 0 AND call_id IN (SELECT call_id FROM chosen)
    GROUP BY call_id
"""


class RepeatedCastPatterns(Rule, id="PY-TYPE0005"):
    """The same `cast` to the same target from the same producer written again and again."""

    minimum_repetitions: int = 3

    query = """
        WITH chosen AS (
          SELECT * FROM core_call WHERE qualified_name IN ('typing.cast', 'typing_extensions.cast')
        ), arguments AS (
          SELECT a.file_id, a.call_id, count(*) AS argument_count,
                 max(syntax_slice(s.content, a.start_line, a.start_column, a.end_line,
                                  a.end_column)) FILTER (a.ordinal = 0) AS first_text,
                 max(syntax_slice(s.content, a.start_line, a.start_column, a.end_line,
                                  a.end_column)) FILTER (a.ordinal = 1) AS second_text,
                 max(p.qualified_name) FILTER (a.ordinal = 1) AS second_name
          FROM core_call_arg a JOIN chosen c USING (file_id, call_id)
          JOIN source_files s ON s.path = c.path
          LEFT JOIN core_call p ON p.file_id = a.file_id AND p.call_id = a.node_id
          GROUP BY a.file_id, a.call_id
        )
        SELECT first_text AS cast_target,
          CASE WHEN second_name <> '' THEN second_name ELSE second_text END AS producer,
          count(*) AS group_count, len(list_distinct(list(path))) AS file_count,
          array_to_string(list_slice(list('`' || path || ':' || start_line
            || CASE WHEN end_line > start_line THEN '-' || end_line ELSE '' END
            || '`' ORDER BY path, call_id), 1, 32), ', ') AS locations,
          arg_min(path, (path, call_id)) AS path,
          arg_min(start_line, (path, call_id)) AS start_line,
          arg_min(start_column, (path, call_id)) AS start_column
        FROM chosen JOIN arguments USING (file_id, call_id) WHERE argument_count = 2
        GROUP BY first_text, producer HAVING group_count >= $minimum_repetitions
    """
    message = (
        "cast to `{cast_target}` from `{producer}` repeats {group_count} times across "
        "{file_count} files at {locations}"
    )


class RedundantBooleanConversion(Rule, id="PY-TYPE0008", applicability="review"):
    """`bool(x)` where `x` is already proven to be a `bool`."""

    query = f"""
        WITH chosen AS (
          SELECT * FROM call_calls c
          WHERE qualified_name = 'builtins.bool' AND NOT is_shadowed
            AND NOT EXISTS (SELECT 1 FROM call_keywords k WHERE k.call_id = c.call_id)
        ), arguments AS ({_ARGUMENTS})
        SELECT node_path AS path, node_start_line AS start_line, node_start_column AS start_column,
               node_end_line AS end_line, node_end_column AS end_column, first_text, first_source,
               node_text AS call_text
        FROM chosen JOIN arguments USING (call_id)
        WHERE argument_count = 1 AND first_type = 'bool'
    """
    message = "`bool` redundantly converts the proven Boolean `{first_text}`"
    # Review, not safe: the kernel types every comparison as `bool`, but `==` on numpy or torch
    # operands yields `np.bool_` or a tensor, which `bool(...)` turned into a Python bool.
    # The operand replaces the call. An operand that is not a name, call, member or index keeps
    # parentheses unless the call stands alone as an assigned, returned, passed, collected or
    # tested value (`flag = bool(a == b)` becomes `flag = a == b`; `bool(a == b) and c` keeps
    # `(a == b)`).
    # Module-level code has no syntax facts, so a statement `name = bool(...)` is read from its
    # line. A call holding a comment is left, since the comment would go with the call.
    fix = r"""
        SELECT f.finding_key, f.path, f.start_line, f.start_column, f.end_line, f.end_column,
               CASE WHEN NOT coalesce(operand.is_composite, true)
                         OR (NOT contains(f.first_source, chr(10))
                             AND coalesce(p.kind IN ('binding', 'call', 'comprehension', 'return',
                                                     'collection', 'expression', 'branch',
                                                     'loop', 'statement', 'effect'),
                                          c.node_id IS NULL
                                          AND regexp_matches(
                                            node_text(opening.text, 0, f.start_column),
                                            '^[^()\[\]{}]*[^=!<>]=\s*$')
                                          AND regexp_matches(
                                            node_text(closing.text, f.end_column,
                                              strlen(closing.text) - f.end_column),
                                            '^\s*(#.*)?$')))
                    THEN f.first_source ELSE '(' || f.first_source || ')' END AS text
        FROM finding f
        JOIN source_lines() opening ON opening.path = f.path AND opening.line = f.start_line
        JOIN source_lines() closing ON closing.path = f.path AND closing.line = f.end_line
        LEFT JOIN syntax_nodes c ON c.path = f.path AND c.kind = 'call' AND c.name = 'bool'
          AND (c.start_line, c.start_column, c.end_line, c.end_column)
              = (f.start_line, f.start_column, f.end_line, f.end_column)
        LEFT JOIN syntax_nodes p ON p.fact_id = c.fact_id AND p.node_id = c.parent_id
        LEFT JOIN (
          SELECT fact_id, parent_id,
                 bool_or(kind NOT IN ('name', 'call', 'member', 'index', 'literal', 'text'))
                   AS is_composite
          FROM syntax_nodes GROUP BY fact_id, parent_id) operand
          ON operand.fact_id = c.fact_id AND operand.parent_id = c.node_id
        WHERE NOT contains(f.call_text, '#')
    """
    examples = (
        Example(
            bad="flag = bool(left == right)\n",
            good="flag = left == right\n",
        ),
    )


class CastCalls(Rule, id="PY-TYPE0009"):
    """Every `typing.cast`, an assertion in place of a checked annotation."""

    query = """
        WITH chosen AS (
          SELECT * FROM core_call
          WHERE qualified_name IN ('typing.cast', 'typing_extensions.cast')
        ), ordered AS (
          SELECT a.file_id, a.call_id, a.start_line, a.start_column, a.end_line, a.end_column,
                 row_number() OVER (PARTITION BY a.file_id, a.call_id ORDER BY a.node_id) - 1
                   AS ordinal
          FROM core_call_arg a JOIN chosen USING (file_id, call_id)
          WHERE NOT a.is_keyword AND a.type <> 'dictionary_splat'
        ), arguments AS (
          SELECT p.file_id, p.call_id,
                 max(syntax_slice(s.content, p.start_line, p.start_column, p.end_line,
                                  p.end_column)) FILTER (p.ordinal = 0) AS first_text,
                 max(syntax_slice(s.content, p.start_line, p.start_column, p.end_line,
                                  p.end_column)) FILTER (p.ordinal = 1) AS second_text
          FROM ordered p JOIN chosen c USING (file_id, call_id)
          JOIN source_files s ON s.path = c.path
          GROUP BY p.file_id, p.call_id
        ), written AS (
          SELECT a.file_id, a.call_id,
                 string_agg(syntax_slice(s.content, a.start_line, a.start_column, a.end_line,
                                         a.end_column), ', ' ORDER BY a.node_id) AS text
          FROM core_call_arg a JOIN chosen c USING (file_id, call_id)
          JOIN source_files s ON s.path = c.path
          GROUP BY a.file_id, a.call_id
        )
        SELECT path, start_line, start_column,
          CASE WHEN second_text IS NOT NULL
               THEN '`cast` to `' || first_text || '` asserts about `' || second_text
                    || '` in place of a checked annotation'
               ELSE '`cast(' || coalesce(w.text, '')
                    || ')` asserts a type in place of a checked annotation' END AS message
        FROM chosen LEFT JOIN arguments USING (file_id, call_id)
        LEFT JOIN written w USING (file_id, call_id)
    """
