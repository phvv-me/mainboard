"""Rules over helpers that outlive or outgrow the code that owns them (mcmr ALL-FUNC)."""

from ....catalog import Example, Function, Relation, Rule


def _next_code(lines: str, *, line: str) -> str:
    """SQL: the first non-blank line after `line` of `lines`, NULL when only blank ones follow."""
    blank = "lambda l: trim(l, ' \t\r') <> ''"
    return f"({line} + list_position(list_transform({lines}[{line} + 1:], {blank}), true))"


def _previous_code(lines: str, *, line: str) -> str:
    """SQL: the last non-blank line before `line` of `lines`, NULL when only blank ones precede."""
    blank = "lambda l: trim(l, ' \t\r') <> ''"
    return (
        f"({line} - list_position(list_reverse(list_transform({lines}[1:{line} - 1], {blank})), "
        "true))"
    )


class FunctionsDeleteStart(Function):
    """The first line deleting the definition on lines `first_line` to `last_line` of a file's
    `lines` removes: its own, or, at the end of a file, the blank lines before it too."""

    lines: list[str]
    first_line: int
    last_line: int

    expression = (
        f"CASE WHEN {_next_code('lines', line='last_line')} IS NULL "
        f"THEN coalesce({_previous_code('lines', line='first_line')}, 0) + 1 ELSE first_line END"
    )


class FunctionsDeleteEnd(Function):
    """The line after the last one deleting the definition ending at `last_line` removes: the
    blank lines that follow it go with it, up to the next code."""

    lines: list[str]
    last_line: int

    expression = f"coalesce({_next_code('lines', line='last_line')}, last_line + 1)"


class FunctionsMemberStart(Function):
    """The first line deleting a class member that starts at `first_line` of a file's `lines`
    removes: the blank lines before it go with it, so the blank lines after it still part its
    class from what follows (delete through `last_line`)."""

    lines: list[str]
    first_line: int

    expression = f"coalesce({_previous_code('lines', line='first_line')}, 0) + 1"


_ATOMIC = (
    r"""^(-?[0-9][0-9_.]*|[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"""
    r"""|"[^"\\]*"|''[^''\\]*'')$"""
)


class FunctionsInlined(Relation):
    """For each one-expression function called once in its own file with plain positional
    arguments, the text of that call once the function's body stands in its place.

    The facts hold the body's syntax (`return` and its expression, the `name` nodes that read a
    parameter) and the call's arguments, so a parameter becomes its argument at exactly the nodes
    that read it, an argument that is not a plain name or literal in parentheses where it would
    bind loosely. Left out are the calls whose arguments could be evaluated a different number of
    times or in a different order (a non-plain argument must feed its parameter once, and in
    parameter order), whose parameter is read inside a nested lambda or comprehension (it may be
    shadowed there), and every function with variadic parameters or defaulted arguments left out
    of the call.
    """

    query = f"""
        WITH candidates AS (
          SELECT f.entity_id, f.name, f.definition_path AS path,
                 f.definition_start_line AS first_line, r.start_line AS ref_line,
                 r.start_column AS ref_column, r.end_line AS ref_end_line,
                 r.end_column AS ref_end_column,
                 (SELECT count(*) FROM function_parameters p
                  WHERE p.function_id = f.entity_id AND NOT p.is_receiver) AS parameter_count,
                 regexp_matches(regexp_extract(f.definition_text,
                   '(?s)^(?:async\\s+)?def\\s+\\w+\\s*\\(.*?\\)\\s*(?:->[^\\n]*?)?:', 0),
                   '[(,]\\s*\\*{1, 2}\\s*[A-Za-z_]') AS has_variadic
          FROM function_functions f
          JOIN function_references r ON r.function_id = f.entity_id AND r.path = f.definition_path
          WHERE f.reference_count = 1 AND NOT f.is_recursive AND NOT f.is_first_class_reference
            AND NOT f.is_async AND starts_with(r.text, f.name || '(')
            AND NOT EXISTS (SELECT 1 FROM function_decorators d WHERE d.function_id = f.entity_id)
            AND (SELECT count(*) FROM function_references x
                 WHERE x.function_id = f.entity_id) = 1
        ), calls AS (
          SELECT c.*, k.arguments, k.keywords
          FROM candidates c
          JOIN (SELECT cc.node_path, cc.node_start_line, cc.node_start_column, cc.node_end_line,
                       cc.node_end_column,
                       coalesce(list(e.text ORDER BY e.ordinal) FILTER (WHERE e.text IS NOT NULL),
                                []) AS arguments,
                       coalesce((SELECT list({{'name': w.name, 'value': w.value_text}})
                                 FROM call_keywords w WHERE w.call_id = cc.call_id), [])
                         AS keywords
                FROM call_calls cc
                LEFT JOIN call_expressions e
                  ON e.call_id = cc.call_id AND e.depth = 0 AND e.relation = 'argument'
                WHERE NOT cc.has_starred_arguments
                  AND NOT regexp_matches(cc.node_text, '[(,]\\s*\\*')
                  AND NOT EXISTS (SELECT 1 FROM call_keywords w WHERE w.call_id = cc.call_id
                                    AND NOT regexp_matches(w.value_text, '{_ATOMIC}'))
                GROUP BY ALL) k
            ON k.node_path = c.path AND k.node_start_line = c.ref_line
           AND k.node_start_column = c.ref_column AND k.node_end_line = c.ref_end_line
           AND k.node_end_column = c.ref_end_column
          WHERE NOT c.has_variadic
            AND len(k.arguments) + len(k.keywords) = c.parameter_count
            AND NOT EXISTS (SELECT 1 FROM function_parameters p
                            WHERE p.function_id = c.entity_id AND NOT p.is_receiver
                              AND p.ordinal < len(k.arguments)
                              AND list_contains(list_transform(k.keywords, lambda w: w.name), p.name))
        ), bodies AS (
          SELECT c.*, s.source, cn.fact_id, x.ordinal AS expr_ordinal, x.subtree_end AS expr_end,
                 x.kind AS expr_kind, x.byte_start AS expr_start, x.byte_length AS expr_length,
                 coalesce(site_parent.kind IN ('return', 'binding', 'effect')
                          OR (site_parent.kind = 'call'
                              AND NOT (site_parent.start_line = site.start_line
                                       AND site_parent.start_column = site.start_column)),
                          false) AS site_takes_any_expression
          FROM calls c
          JOIN syntax_nodes cn ON cn.path = c.path AND cn.kind = 'callable' AND cn.name = c.name
                              AND cn.start_line = c.first_line
          JOIN syntax_facts s ON s.fact_id = cn.fact_id
          JOIN syntax_nodes ret
            ON ret.fact_id = cn.fact_id AND ret.kind = 'return' AND ret.parent_id = cn.node_id
          JOIN syntax_children ch
            ON ch.fact_id = cn.fact_id AND ch.parent_ordinal = ret.ordinal AND ch.child_order = 0
          JOIN syntax_nodes x ON x.fact_id = cn.fact_id AND x.ordinal = ch.child_ordinal
          LEFT JOIN syntax_nodes site
            ON site.path = c.path AND site.kind = 'call' AND site.start_line = c.ref_line
           AND site.start_column = c.ref_column AND site.end_line = c.ref_end_line
           AND site.end_column = c.ref_end_column
          LEFT JOIN syntax_nodes site_parent
            ON site_parent.fact_id = site.fact_id AND site_parent.node_id = site.parent_id
          QUALIFY row_number() OVER (PARTITION BY c.entity_id ORDER BY cn.depth, cn.fact_id,
                                                                    site.ordinal) = 1
        ), parameters AS (
          SELECT entity_id, ordinal, name, argument, regexp_matches(argument, '{_ATOMIC}') AS atomic
          FROM (SELECT b.entity_id, p.ordinal, p.name,
                       coalesce(b.arguments[(p.ordinal + 1)::BIGINT],
                                list_filter(b.keywords, lambda w: w.name = p.name)[1].value) AS argument
                FROM bodies b
                JOIN function_parameters p ON p.function_id = b.entity_id AND NOT p.is_receiver)
        ), uses AS (
          SELECT b.entity_id, u.ordinal AS use_ordinal, u.byte_start, u.byte_length,
                 q.ordinal AS parameter_ordinal, q.atomic, q.argument,
                 parent.kind = 'call' AND NOT (parent.start_line = u.start_line
                                               AND parent.start_column = u.start_column)
                   AS is_call_argument,
                 EXISTS (SELECT 1 FROM syntax_nodes k,
                           LATERAL (SELECT flatten(list_transform(
                                      regexp_extract_all(
                                        node_text(b.source, k.byte_start, k.byte_length),
                                        '\\bfor\\s+(.+?)\\s+in\\b', 1),
                                      lambda t: regexp_extract_all(t, '[A-Za-z_]\\w*')))
                                    AS bound)
                         WHERE k.fact_id = u.fact_id AND k.kind IN ('comprehension', 'callable')
                           AND k.ordinal BETWEEN b.expr_ordinal AND b.expr_end
                           AND u.ordinal > k.ordinal AND u.ordinal <= k.subtree_end
                           AND (k.kind = 'callable'
                                OR list_contains(bound, u.name)
                                OR list_has_any(bound,
                                                regexp_extract_all(q.argument, '[A-Za-z_]\\w*'))))
                   AS shadowed
          FROM bodies b
          JOIN syntax_nodes u ON u.fact_id = b.fact_id AND u.kind = 'name'
                             AND u.ordinal BETWEEN b.expr_ordinal AND b.expr_end
          JOIN parameters q ON q.entity_id = b.entity_id AND q.name = u.name
          JOIN syntax_nodes parent ON parent.fact_id = u.fact_id AND parent.node_id = u.parent_id
        ), readings AS (
          SELECT q.entity_id, q.ordinal, q.atomic, count(u.use_ordinal) AS uses,
                 coalesce(bool_or(u.shadowed), false) AS shadowed, min(u.byte_start) AS first_use
          FROM parameters q LEFT JOIN uses u
            ON u.entity_id = q.entity_id AND u.parameter_ordinal = q.ordinal
          GROUP BY ALL
        ), sound AS (
          (SELECT entity_id FROM readings GROUP BY entity_id
           HAVING NOT bool_or(shadowed) AND NOT bool_or(NOT atomic AND uses <> 1)
           -- A function of no parameter has no reading to judge, and substitutes nothing.
           UNION ALL
           SELECT entity_id FROM bodies
           WHERE entity_id NOT IN (SELECT entity_id FROM parameters))
          EXCEPT
          SELECT entity_id FROM (
            SELECT entity_id, first_use,
                   lag(first_use) OVER (PARTITION BY entity_id ORDER BY ordinal) AS previous_use
            FROM readings WHERE NOT atomic)
          WHERE first_use < previous_use
        ), pieces AS (
          SELECT u.entity_id, u.byte_start, u.byte_length,
                 CASE WHEN u.atomic OR u.is_call_argument THEN u.argument
                      ELSE '(' || u.argument || ')' END AS replacement,
                 coalesce(lag(u.byte_start + u.byte_length)
                            OVER (PARTITION BY u.entity_id ORDER BY u.byte_start),
                          b.expr_start) AS previous_end,
                 b.source, b.expr_start
          FROM uses u JOIN bodies b USING (entity_id)
        ), heads AS (
          SELECT entity_id,
                 string_agg(node_text(source, previous_end, (byte_start - previous_end)::BIGINT)
                              || replacement, '' ORDER BY byte_start) AS head,
                 max(byte_start + byte_length)::BIGINT AS last_end
          FROM pieces GROUP BY entity_id
        ), assembled AS (
          SELECT b.entity_id, b.path, b.ref_line, b.ref_column, b.ref_end_line, b.ref_end_column,
                 b.expr_kind, b.site_takes_any_expression,
                 coalesce(h.head, '')
                   || node_text(b.source, coalesce(h.last_end, b.expr_start),
                                (b.expr_start + b.expr_length
                                 - coalesce(h.last_end, b.expr_start))::BIGINT) AS body,
                 coalesce((SELECT bool_and(atomic) FROM parameters q
                           WHERE q.entity_id = b.entity_id), true) AS arguments_atomic
          FROM bodies b JOIN sound USING (entity_id) LEFT JOIN heads h USING (entity_id)
        )
        SELECT entity_id, path, ref_line, ref_column, ref_end_line, ref_end_column,
               CASE WHEN site_takes_any_expression
                      OR expr_kind IN ('call', 'member', 'index', 'text', 'literal')
                      OR (expr_kind = 'name' AND arguments_atomic)
                      OR (expr_kind IN ('collection', 'comprehension')
                          AND regexp_matches(body, '^[\\[{{(]'))
                    THEN replace(body, chr(13), '')
                    ELSE '(' || replace(body, chr(13), '') || ')' END AS inlined
        FROM assembled
    """


class OwnershipPrivateHelpers(Relation):
    """The `ownership_private_helpers()` relation."""

    query = """
    SELECT * FROM function_functions f
    WHERE scope = 'module' AND visibility <> 'public'
      AND NOT EXISTS (SELECT 1 FROM function_decorators d WHERE d.function_id = f.entity_id)
"""


class SingleUseTrivialHelper(Rule, id="ALL-FUNC0002", applicability="review"):
    """A private one-line helper with only one local reference."""

    maximum_lines: int = 1
    ignore_names: tuple[str, ...] = ()

    query = """
        SELECT * FROM ownership_private_helpers()
        WHERE direct_statement_count = 1 AND implementation_lines <= $maximum_lines
          AND reference_count = 1 AND NOT is_recursive AND NOT is_first_class_reference
          AND NOT is_pass_body AND NOT is_raise_body
          AND NOT list_contains($ignore_names::VARCHAR[], name)
    """
    message = "`{fact_id}` satisfies single use trivial helper"
    examples = (
        Example(
            bad=(
                """def _double(value):
    return value * 2


def run(items):
    return _double(items)
"""
            ),
            good="def run(items):\n    return items * 2\n",
        ),
    )
    # The call becomes the helper's expression with its parameters replaced by the call's
    # arguments (`functions_inlined()`), and the helper goes with the blank lines after it.
    fix = """
        SELECT f.finding_key, i.path, i.ref_line AS start_line, i.ref_column AS start_column,
               i.ref_end_line AS end_line, i.ref_end_column AS end_column, i.inlined AS text
        FROM finding f JOIN functions_inlined() i ON i.entity_id = f.entity_id
        UNION ALL BY NAME
        SELECT f.finding_key, f.definition_path AS path,
               functions_delete_start(string_split(c.content, chr(10)), f.definition_start_line,
                                      f.definition_end_line) AS start_line,
               0 AS start_column,
               functions_delete_end(string_split(c.content, chr(10)), f.definition_end_line)
                 AS end_line,
               0 AS end_column, '' AS text
        FROM finding f JOIN functions_inlined() i ON i.entity_id = f.entity_id
        JOIN source_files c ON c.path = f.definition_path
    """


class ClassOwnedModuleHelper(Rule, id="ALL-FUNC0003", applicability="review"):
    """A private module helper used exclusively by one class method."""

    minimum_lines: int = 2
    ignore_names: tuple[str, ...] = ()

    # The one reference sits in a method: a call from the class body itself (`relations =
    # _relations()`) runs while the class is still being built, when no static method of it can.
    query = """
        SELECT * FROM ownership_private_helpers()
        WHERE implementation_lines >= $minimum_lines AND reference_count = 1
          AND sole_reference_owner_class <> ''
          AND sole_reference_owner_definition_start_line IS NOT NULL
          AND NOT list_contains($ignore_names::VARCHAR[], name)
    """
    message = "`{fact_id}` satisfies class owned module helper"
    examples = (
        Example(
            bad=(
                """def _total(items):
    count = 0
    for item in items:
        count += item
    return count


class Basket:
    def total(self, items):
        return _total(items)
"""
            ),
            good=(
                """class Basket:
    def total(self, items):
        return Basket._total(items)

    @staticmethod
    def _total(items):
        count = 0
        for item in items:
            count += item
        return count
"""
            ),
        ),
    )
    # The helper moves to the end of its owning module-level class as a `@staticmethod` and its
    # one call is qualified with the class. Left alone: a class that already holds the name, a
    # protocol or `TypedDict`, and a helper with a multi-line string other than its docstring
    # (indenting it would change the string). Helpers only it called now belong to the class in
    # turn, and later passes move them.
    fix = r"""
        WITH moved AS (
          SELECT f.finding_key, f.path, f.name, f.sole_reference_owner_class AS owner,
                 f.definition_start_line AS first_line, f.definition_end_line AS last_line,
                 string_split(s.content, chr(10)) AS lines, c.end_line::BIGINT AS class_end,
                 f.sole_reference_owner_definition_start_column::BIGINT AS indent,
                 string_split(replace(f.definition_text, chr(13), ''), chr(10)) AS body,
                 r.start_line AS call_line, r.start_column AS call_column
          FROM finding f
          JOIN function_references r ON r.function_id = f.entity_id AND r.path = f.path
          JOIN class_classes c ON c.path = f.path AND c.name = f.sole_reference_owner_class
            AND c.scope = 'module' AND NOT c.is_protocol
            AND c.start_line < f.sole_reference_owner_definition_start_line
            AND c.end_line >= f.sole_reference_owner_definition_end_line
          JOIN source_files s ON s.path = f.path
          WHERE starts_with(r.text, f.name)
            AND (SELECT count(*) FROM function_references x WHERE x.function_id = f.entity_id) = 1
            AND NOT EXISTS (SELECT 1 FROM class_methods m
                            WHERE m.class_id = c.class_id AND m.name = f.name)
            AND NOT EXISTS (SELECT 1 FROM class_direct_bases b
                            WHERE b.class_id = c.class_id AND b.value LIKE '%TypedDict')
            AND len(string_split(f.definition_text, repeat(chr(34), 3)))
              + len(string_split(f.definition_text, repeat(chr(39), 3))) - 2
              <= 2 * (f.docstring <> '')::INT
            AND NOT contains(f.definition_text, chr(92) || chr(10))
        )
        SELECT finding_key, path, call_line AS start_line, call_column AS start_column,
               call_line AS end_line, call_column AS end_column, owner || '.' AS text
        FROM moved
        UNION ALL BY NAME
        SELECT finding_key, path,
               functions_delete_start(lines, first_line, last_line) AS start_line,
               0 AS start_column, functions_delete_end(lines, last_line) AS end_line,
               0 AS end_column, '' AS text
        FROM moved
        UNION ALL BY NAME
        SELECT finding_key, path, class_end AS start_line,
               strlen(rtrim(lines[class_end], chr(13))) AS start_column,
               class_end AS end_line, strlen(rtrim(lines[class_end], chr(13))) AS end_column,
               chr(10) || chr(10) || repeat(' ', indent) || '@staticmethod' || chr(10)
                 || array_to_string(list_transform(body,
                      lambda l: CASE WHEN l = '' THEN '' ELSE repeat(' ', indent) || l END),
                    chr(10)) AS text
        FROM moved
    """


class UnnecessaryOneLineConcreteFunction(Rule, id="ALL-FUNC0004", applicability="review"):
    """One-line nested functions whose named boundary lacks demonstrated reuse. A function a
    decorator takes (`@render.register("html")`, `@wraps(fn)`) is used by it, though nothing
    calls it by name, so inlining or dropping it would change what the decorator registers."""

    minimum_references: int = 2

    query = """
        SELECT * FROM function_functions f
        WHERE scope = 'nested' AND implementation_lines = 1
          AND reference_count < $minimum_references
          AND NOT (is_property OR is_abstract OR is_protocol_member OR is_overload
                   OR is_protocol_name OR is_recursive OR is_first_class_reference
                   OR is_pass_body OR is_raise_body)
          AND NOT EXISTS (SELECT 1 FROM function_decorators x
                          WHERE x.function_id = f.entity_id
                            AND functions_decorator_name(x.decorator)
                                NOT IN ('staticmethod', 'classmethod'))
    """
    message = "`{fact_id}` satisfies unnecessary one line concrete function"
    examples = (
        Example(
            bad=(
                """def run(items):
    def double(value):
        return value * 2

    return double(items)
"""
            ),
            good="def run(items):\n    return items * 2\n",
        ),
    )
    # The single call becomes the body's expression with its parameters replaced by the call's
    # arguments (`functions_inlined()`); one never called is dropped. The definition goes with
    # the blank lines after it either way.
    fix = """
        SELECT f.finding_key, i.path, i.ref_line AS start_line, i.ref_column AS start_column,
               i.ref_end_line AS end_line, i.ref_end_column AS end_column, i.inlined AS text
        FROM finding f JOIN functions_inlined() i ON i.entity_id = f.entity_id
        UNION ALL BY NAME
        SELECT f.finding_key, f.definition_path AS path,
               functions_delete_start(string_split(c.content, chr(10)), f.definition_start_line,
                                      f.definition_end_line) AS start_line,
               0 AS start_column,
               functions_delete_end(string_split(c.content, chr(10)), f.definition_end_line)
                 AS end_line,
               0 AS end_column, '' AS text
        FROM finding f
        LEFT JOIN functions_inlined() i ON i.entity_id = f.entity_id
        JOIN source_files c ON c.path = f.definition_path
        WHERE i.entity_id IS NOT NULL OR f.reference_count = 0
    """
