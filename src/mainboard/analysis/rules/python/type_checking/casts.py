"""Rules over `typing.cast` and conversions the types already prove (mcmr's PY-TYPE casts)."""

from ....catalog import Example, Rule
from ....operations.nodes import Rewrite

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


class CastCalls(Rule, id="PY-TYPE0009", applicability="review"):
    """Every `typing.cast`, an assertion in place of a checked annotation."""

    # Review: the cast may be what keeps a checker quiet, so the proof (ty and pyrefly may not
    # report more) decides whether the call goes. The call is replaced by its second argument. The
    # query names the callee, so the pattern need not, whatever the cast is imported as, and
    # names a node only for a cast written with positional arguments. One with a comment inside is
    # left alone by the rewrite, since the comment would go with the call. One under a conditional
    # is left alone too: the dialect's binding parents do not yet include `conditional_expression`,
    # so a second argument that is itself a conditional would lose its parentheses. So is a cast
    # with a comment on its statement's line, on the line above it (alone there) or on a line of
    # the call: the house rule wants a cast's reason stated at the site, and a reasoned cast
    # (`types.new_class` builds a type no checker can see) is sanctioned, so it stays a finding
    # for a person to judge and no fix deletes it with its reason left behind.
    rewrites = (Rewrite(match="$_($T, $X)", to="$X"),)
    query = r"""
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
                                         a.end_column), ', ' ORDER BY a.node_id) AS text,
                 bool_and(NOT a.is_keyword AND a.type NOT IN ('list_splat', 'dictionary_splat'))
                   AS positional
          FROM core_call_arg a JOIN chosen c USING (file_id, call_id)
          JOIN source_files s ON s.path = c.path
          GROUP BY a.file_id, a.call_id
        ), nested AS (
          SELECT c.file_id, c.call_id, k.path AS home, k.start_line AS opens, k.end_line AS closes,
                 p.type AS parent_type
          FROM chosen c JOIN ast_nodes k ON k.path = c.path AND k.node_id = c.call_id
          JOIN ast_nodes p ON p.path = k.path AND p.node_id = k.parent_id
        ), stated AS (
          SELECT n.file_id, n.call_id, max_by(s.start_line, s.node_id) AS first_line
          FROM nested n
          JOIN ast_nodes s ON s.path = n.home AND s.node_id <= n.call_id
            AND n.call_id <= s.node_id + s.descendant_count
          JOIN ast_nodes b ON b.path = s.path AND b.node_id = s.parent_id
            AND b.type IN ('block', 'module')
          GROUP BY n.file_id, n.call_id
        ), reasoned AS (
          SELECT DISTINCT n.file_id, n.call_id
          FROM nested n JOIN stated t USING (file_id, call_id)
          JOIN ast_nodes m ON m.path = n.home AND m.type = 'comment'
          LEFT JOIN source_lines() l ON l.path = m.path AND l.line = m.start_line
          WHERE m.start_line BETWEEN n.opens AND n.closes OR m.start_line = t.first_line
             OR (m.start_line = t.first_line - 1 AND regexp_matches(l.text, '^\s*#'))
        )
        SELECT path, start_line, start_column,
          CASE WHEN second_text IS NOT NULL
               THEN '`cast` to `' || first_text || '` asserts about `' || second_text
                    || '` in place of a checked annotation'
               ELSE '`cast(' || coalesce(w.text, '')
                    || ')` asserts a type in place of a checked annotation' END AS message,
          CASE WHEN w.positional AND coalesce(n.parent_type, '') <> 'conditional_expression'
                    AND r.call_id IS NULL
               THEN call_id END AS node
        FROM chosen LEFT JOIN arguments USING (file_id, call_id)
        LEFT JOIN written w USING (file_id, call_id)
        LEFT JOIN nested n USING (file_id, call_id)
        LEFT JOIN reasoned r USING (file_id, call_id)
    """
    examples = (
        Example(
            bad='from typing import cast\n\n\ndef first(row):\n    return cast(int, row["a"])\n',
            good='def first(row):\n    return row["a"]\n',
        ),
        Example(
            bad="import typing\n\n\ndef scaled(a, b):\n    return typing.cast(int, a + b) * 2\n",
            good="def scaled(a, b):\n    return (a + b) * 2\n",
        ),
        Example(
            bad="import typing as t\n\n\ndef wrapped(x):\n    return t.cast(int, t.cast(str, x))\n",
            good="def wrapped(x):\n    return x\n",
        ),
        Example(
            bad="from typing import cast\n\n\ndef raw(x):\n    return cast(\n        int,\n        x,  # the raw value\n    )\n",
            good="from typing import cast\n\n\ndef raw(x):\n    return cast(\n        int,\n        x,  # the raw value\n    )\n",
        ),
        Example(
            bad="from typing import cast\n\n\ndef named(x):\n    return cast(typ=int, val=x)\n",
            good="from typing import cast\n\n\ndef named(x):\n    return cast(typ=int, val=x)\n",
        ),
        Example(
            bad="from typing import cast\n\n\ndef pick(a, b, c, d):\n    return cast(int, a if b else c) if d else 0\n",
            good="from typing import cast\n\n\ndef pick(a, b, c, d):\n    return cast(int, a if b else c) if d else 0\n",
        ),
        Example(
            bad='from typing import cast\n\n\ndef show(x):\n    return f"{cast(dict, {x: 1})}"\n',
            good='def show(x):\n    return f"{ ({x: 1}) }"\n',
        ),
        Example(
            bad='import types\nfrom typing import cast\n\n\ndef build(name):\n    # cast: new_class makes a type no checker can see\n    return cast("type[Experiment]", types.new_class(name))\n',
            good='import types\nfrom typing import cast\n\n\ndef build(name):\n    # cast: new_class makes a type no checker can see\n    return cast("type[Experiment]", types.new_class(name))\n',
        ),
        Example(
            bad="from typing import cast\n\n\ndef first(row):\n    return cast(int, row[0])  # cast: the schema says int\n",
            good="from typing import cast\n\n\ndef first(row):\n    return cast(int, row[0])  # cast: the schema says int\n",
        ),
        Example(
            bad="from typing import cast\n\n\ndef first(row, n):\n    n += 1  # counted\n    return cast(int, row[n])\n",
            good="def first(row, n):\n    n += 1  # counted\n    return row[n]\n",
        ),
    )
