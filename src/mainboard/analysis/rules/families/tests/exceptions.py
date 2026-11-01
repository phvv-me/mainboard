"""mcmr's TryBlockFact, derived from sitting_duck's Python trees.

A region is a `try` statement with its protected statements, its handlers and what each catches.
The relations build on one another, each reading the earlier ones by their view names
(`try_block_tries` for `tries`); `fact_id` in the helpers is the syntax tree file's and a node is
named by its `node_id` there. The nodes read are those of the files holding a `try`.

Differences from the kernel, kept on purpose: a handler body's statements carry `kind` `return` or
`statement` as mcmr's rules expect (the kernel leaves it empty there, so PY-EXCE0004 never fires
on mcmr); the callee a lone protected call resolves to comes from imports and the receiver's
annotation or constructor, which differ from the kernel's type inference in a few cases.
"""

from ....catalog import Family

_REGION = "'tryblockfact:' || t.path || '/regions:' || t.ordinal"

_TRIES = {
    "facts": """
        SELECT row_number() OVER (ORDER BY f.path) - 1 AS fact_order,
               'tryblockfact:' || f.path AS fact_id, f.path, 1 AS start_line, 0 AS start_column,
               m.end_line, 0 AS end_column, 'python' AS language
        FROM ast_files f JOIN ast_nodes m ON m.fact_id = f.fact_id AND m.parent_id IS NULL
        WHERE f.language = 'python'
    """,
    "nodes": """
        SELECT fact_id, path, node_id, parent_id, type, name, modifiers, sibling_index,
               children_count, descendant_count, start_line, start_column, end_line, end_column,
               tests_home(scope) AS home, scope['function'] AS function
        FROM ast_nodes
        WHERE fact_id IN (SELECT fact_id FROM ast_files WHERE language = 'python')
          AND fact_id IN (SELECT fact_id FROM ast_nodes WHERE type = 'try_statement')
    """,
    "tries": """
        SELECT fact_id, path, node_id AS try, node_id + descendant_count AS last, start_line,
               start_column, end_line, end_column,
               row_number() OVER (PARTITION BY fact_id ORDER BY node_id) - 1 AS ordinal, home,
               function, coalesce(home = function, false) AS in_function
        FROM try_block_nodes WHERE type = 'try_statement'
    """,
    "handlers": """
        SELECT t.fact_id, t.try, c.node_id AS handler,
               row_number() OVER (PARTITION BY t.fact_id, t.try ORDER BY c.node_id) - 1 AS ordinal,
               c.type = 'except_group_clause' OR EXISTS (
                 SELECT 1 FROM try_block_nodes s WHERE s.fact_id = c.fact_id
                   AND s.parent_id = c.node_id AND s.type = '*') AS star
        FROM try_block_tries t
        JOIN try_block_nodes c ON c.fact_id = t.fact_id AND c.parent_id = t.try
          AND c.type IN ('except_clause', 'except_group_clause')
    """,
    "shape": """
        SELECT t.fact_id, t.try, coalesce(bool_or(h.star), false) AS is_exception_group,
               EXISTS (SELECT 1 FROM try_block_nodes e WHERE e.fact_id = t.fact_id
                         AND e.parent_id = t.try AND e.type = 'else_clause') AS has_else,
               EXISTS (SELECT 1 FROM try_block_nodes e WHERE e.fact_id = t.fact_id
                         AND e.parent_id = t.try AND e.type = 'finally_clause') AS has_finally
        FROM try_block_tries t
        LEFT JOIN try_block_handlers h ON h.fact_id = t.fact_id AND h.try = t.try
        GROUP BY t.fact_id, t.try
    """,
    "parts": """
        SELECT t.fact_id, t.try, 0 AS part, b.node_id AS block, b.descendant_count AS size
        FROM try_block_tries t
        JOIN try_block_nodes b ON b.fact_id = t.fact_id AND b.parent_id = t.try AND b.type =
          'block'
        UNION ALL
        SELECT h.fact_id, h.try, 1 + h.ordinal, b.node_id, b.descendant_count
        FROM try_block_handlers h
        JOIN try_block_nodes b ON b.fact_id = h.fact_id AND b.parent_id = h.handler
          AND b.type = 'block'
        UNION ALL
        SELECT t.fact_id, t.try, 1000, b.node_id, b.descendant_count
        FROM try_block_tries t
        JOIN try_block_nodes c ON c.fact_id = t.fact_id AND c.parent_id = t.try
          AND c.type = 'else_clause'
        JOIN try_block_nodes b ON b.fact_id = c.fact_id AND b.parent_id = c.node_id
          AND b.type = 'block'
        UNION ALL
        SELECT t.fact_id, t.try, 2000, b.node_id, b.descendant_count
        FROM try_block_tries t
        JOIN try_block_nodes c ON c.fact_id = t.fact_id AND c.parent_id = t.try
          AND c.type = 'finally_clause'
        JOIN try_block_nodes b ON b.fact_id = c.fact_id AND b.parent_id = c.node_id
          AND b.type = 'block'
    """,
    "counts": """
        WITH owned AS (
          SELECT s.fact_id, s.node_id AS stmt, s.type, s.home, arg_max(t.try, t.try) AS owner
          FROM try_block_nodes s
          JOIN try_block_nodes p ON p.fact_id = s.fact_id AND p.node_id = s.parent_id
            AND p.type = 'block'
          JOIN try_block_tries t ON t.fact_id = s.fact_id AND s.node_id > t.try AND s.node_id <=
            t.last
          WHERE s.type <> 'comment'
          GROUP BY ALL)
        SELECT b.fact_id, b.try, b.part, count(s.stmt) AS value,
               row_number() OVER (PARTITION BY b.fact_id, b.try ORDER BY b.part) - 1 AS ordinal
        FROM try_block_parts b
        JOIN try_block_tries t ON t.fact_id = b.fact_id AND t.try = b.try
        LEFT JOIN owned s ON s.fact_id = b.fact_id AND s.owner = b.try AND s.home = t.home
          AND s.stmt > b.block AND s.stmt <= b.block + b.size
          AND NOT list_contains(['function_definition', 'class_definition', 'decorated_definition',
                                 'try_statement'], s.type)
        GROUP BY b.fact_id, b.try, b.part
    """,
    "imports": """
        SELECT * FROM (
          SELECT i.fact_id, i.end_line, i.end_column,
                 split_part(d.name, '.', 1) AS local, split_part(d.name, '.', 1) AS origin
          FROM try_block_nodes i
          JOIN try_block_nodes d ON d.fact_id = i.fact_id AND d.parent_id = i.node_id
            AND d.type = 'dotted_name'
          WHERE i.type = 'import_statement'
          UNION ALL
          SELECT i.fact_id, i.end_line, i.end_column, a.name, d.name
          FROM try_block_nodes i
          JOIN try_block_nodes a ON a.fact_id = i.fact_id AND a.parent_id = i.node_id
            AND a.type = 'aliased_import'
          JOIN try_block_nodes d ON d.fact_id = a.fact_id AND d.parent_id = a.node_id
            AND d.sibling_index = 0
          WHERE i.type = 'import_statement'
          UNION ALL
          SELECT i.fact_id, i.end_line, i.end_column, n.name, m.name || '.' || n.name
          FROM try_block_nodes i
          JOIN try_block_nodes m ON m.fact_id = i.fact_id AND m.parent_id = i.node_id
            AND m.sibling_index = 1 AND m.type = 'dotted_name'
          JOIN try_block_nodes n ON n.fact_id = i.fact_id AND n.parent_id = i.node_id
            AND n.sibling_index > 1 AND n.type = 'dotted_name'
          WHERE i.type = 'import_from_statement'
          UNION ALL
          SELECT i.fact_id, i.end_line, i.end_column, a.name, m.name || '.' || d.name
          FROM try_block_nodes i
          JOIN try_block_nodes m ON m.fact_id = i.fact_id AND m.parent_id = i.node_id
            AND m.sibling_index = 1 AND m.type = 'dotted_name'
          JOIN try_block_nodes a ON a.fact_id = i.fact_id AND a.parent_id = i.node_id
            AND a.sibling_index > 1 AND a.type = 'aliased_import'
          JOIN try_block_nodes d ON d.fact_id = a.fact_id AND d.parent_id = a.node_id
            AND d.sibling_index = 0
          WHERE i.type = 'import_from_statement')
        QUALIFY count(*) OVER (PARTITION BY fact_id, local) = 1
    """,
    "caught": """
        WITH handled AS (
          SELECT h.fact_id, h.try, h.ordinal, h.handler, h.star,
                 CASE WHEN count(a.node_id) > 1 THEN 'tuple'
                      ELSE any_value(coalesce(w.type, a.type)) END AS kind,
                 min(coalesce(w.node_id, a.node_id)) AS expr,
                 CASE WHEN count(a.node_id) > 1 THEN NULL
                      ELSE any_value(coalesce(w.name, a.name)) END AS expr_name,
                 arg_min(coalesce(w.start_line, a.start_line), a.node_id) AS sl,
                 arg_min(coalesce(w.start_column, a.start_column), a.node_id) AS sc,
                 arg_max(coalesce(w.end_line, a.end_line), a.node_id) AS el,
                 arg_max(coalesce(w.end_column, a.end_column), a.node_id) AS ec,
                 coalesce(any_value(alias.name), '') AS alias
          FROM try_block_handlers h
          LEFT JOIN try_block_nodes a ON a.fact_id = h.fact_id AND a.parent_id = h.handler
            AND a.type NOT IN ('except', '*', 'block', 'comment')
          LEFT JOIN try_block_nodes w ON w.fact_id = a.fact_id AND w.parent_id = a.node_id
            AND a.type = 'as_pattern' AND w.sibling_index = 0
          LEFT JOIN try_block_nodes alias ON alias.fact_id = a.fact_id AND alias.parent_id =
            a.node_id
            AND alias.type = 'as_pattern_target'
          GROUP BY h.fact_id, h.try, h.ordinal, h.handler, h.star)
        SELECT h.*, coalesce(CASE WHEN h.kind IN ('identifier', 'attribute') THEN
                 i.origin || substr(h.expr_name, length(split_part(h.expr_name, '.', 1)) + 1)
               END, '') AS caught_import
        FROM handled h
        LEFT JOIN try_block_imports i ON i.fact_id = h.fact_id
          AND i.local = split_part(h.expr_name, '.', 1) AND (i.end_line, i.end_column) <= (h.sl,
            h.sc)
    """,
    "sole": """
        SELECT t.fact_id, t.try, t.home, t.function, c.node_id AS call, f.type AS callee_type,
               f.name AS callee_name, t.start_line, t.start_column
        FROM try_block_tries t
        JOIN try_block_parts b ON b.fact_id = t.fact_id AND b.try = t.try AND b.part = 0
        JOIN try_block_nodes s ON s.fact_id = t.fact_id AND s.parent_id = b.block
          AND s.type = 'expression_statement'
        JOIN try_block_nodes e ON e.fact_id = t.fact_id AND e.parent_id = s.node_id
          AND e.sibling_index = 0
        JOIN try_block_nodes c ON c.fact_id = t.fact_id AND c.type = 'call'
          AND (c.node_id = e.node_id OR (c.parent_id = e.node_id AND c.sibling_index = 2
                                         AND e.type = 'assignment' AND e.children_count = 3))
        JOIN try_block_nodes f ON f.fact_id = t.fact_id AND f.parent_id = c.node_id
          AND f.sibling_index = 0
        WHERE (SELECT count(*) FROM try_block_nodes x WHERE x.fact_id = t.fact_id
               AND x.parent_id = b.block AND x.type <> 'comment') = 1
          AND NOT EXISTS (SELECT 1 FROM try_block_nodes x WHERE x.fact_id = t.fact_id
                          AND x.node_id > c.node_id AND x.node_id <= c.node_id + c.descendant_count
                          AND x.type IN ('call', 'await', 'lambda'))
    """,
    "callee": """
        WITH receivers AS (
          SELECT s.fact_id, s.try, split_part(s.callee_name, '.', 1) AS name,
                 regexp_replace(t.name, '\\[.*$', '') AS type_name, p.start_line, p.start_column
          FROM try_block_sole s
          JOIN try_block_nodes p ON p.fact_id = s.fact_id AND p.type = 'typed_parameter'
            AND p.name = split_part(s.callee_name, '.', 1) AND p.home = s.function
          JOIN try_block_nodes t ON t.fact_id = s.fact_id AND t.parent_id = p.node_id AND t.type =
            'type'
          WHERE s.callee_type = 'attribute' AND s.function IS NOT NULL
          UNION ALL
          SELECT s.fact_id, s.try, split_part(s.callee_name, '.', 1), f.name, a.start_line,
            a.start_column
          FROM try_block_sole s
          JOIN try_block_nodes a ON a.fact_id = s.fact_id AND a.type = 'assignment'
            AND a.children_count = 3 AND a.name = split_part(s.callee_name, '.', 1)
            AND a.home = s.home
          JOIN try_block_nodes c ON c.fact_id = s.fact_id AND c.parent_id = a.node_id
            AND c.sibling_index = 2 AND c.type = 'call'
          JOIN try_block_nodes f ON f.fact_id = s.fact_id AND f.parent_id = c.node_id
            AND f.sibling_index = 0 AND f.type IN ('identifier', 'attribute')
          WHERE s.callee_type = 'attribute'
        ), writes AS (
          SELECT s.fact_id, s.try, count(*) AS writes
          FROM try_block_sole s
          JOIN try_block_nodes w ON w.fact_id = s.fact_id AND w.type = 'identifier'
            AND w.name = split_part(s.callee_name, '.', 1) AND w.home = s.home
            AND (list_contains(w.modifiers, 'assignment_target')
                 OR list_contains(w.modifiers, 'in_parameters')
                 OR list_contains(w.modifiers, 'in_typed_parameter'))
          WHERE s.callee_type = 'attribute'
          GROUP BY s.fact_id, s.try)
        SELECT s.fact_id, s.try,
               coalesce(
                 CASE WHEN s.callee_type = 'attribute' AND w.writes = 1 AND r.type_name IS NOT NULL
                           AND ri.origin IS NOT NULL
                      THEN ri.origin || substr(r.type_name, length(split_part(r.type_name, '.',
                        1)) + 1)
                           || substr(s.callee_name, length(split_part(s.callee_name, '.', 1)) + 1)
                      WHEN s.callee_type IN ('identifier', 'attribute') AND i.origin IS NOT NULL
                      THEN i.origin || substr(s.callee_name, length(split_part(s.callee_name, '.',
                        1)) + 1)
                 END, '') AS name
        FROM try_block_sole s
        LEFT JOIN try_block_imports i ON i.fact_id = s.fact_id
          AND i.local = split_part(s.callee_name, '.', 1)
          AND (i.end_line, i.end_column) <= (s.start_line, s.start_column)
        LEFT JOIN writes w ON w.fact_id = s.fact_id AND w.try = s.try
        LEFT JOIN receivers r ON r.fact_id = s.fact_id AND r.try = s.try
          AND (r.start_line, r.start_column) < (s.start_line, s.start_column)
        LEFT JOIN try_block_imports ri ON ri.fact_id = s.fact_id
          AND ri.local = split_part(r.type_name, '.', 1)
        QUALIFY row_number() OVER (PARTITION BY s.fact_id, s.try
                                   ORDER BY r.start_line DESC NULLS LAST) = 1
    """,
    "statements": """
        SELECT t.fact_id, t.try, t.function, t.home, b.block, s.node_id AS stmt, s.type,
               s.start_line, s.start_column, s.end_line, s.end_column,
               s.node_id + s.descendant_count AS last, s.home AS scope,
               row_number() OVER (PARTITION BY t.fact_id, t.try ORDER BY s.node_id) - 1 AS idx,
               lead(s.node_id) OVER (PARTITION BY t.fact_id, t.try ORDER BY s.node_id) AS next
        FROM try_block_tries t
        JOIN try_block_shape h ON h.fact_id = t.fact_id AND h.try = t.try
        JOIN try_block_parts b ON b.fact_id = t.fact_id AND b.try = t.try AND b.part = 0
        JOIN try_block_nodes s ON s.fact_id = t.fact_id AND s.parent_id = b.block
          AND s.type <> 'comment'
        WHERE t.in_function AND NOT h.is_exception_group AND NOT h.has_finally
    """,
    "setup": """
        WITH externals AS (
          SELECT g.fact_id, g.function, i.name
          FROM try_block_nodes g
          JOIN try_block_nodes i ON i.fact_id = g.fact_id AND i.parent_id = g.node_id
            AND i.type = 'identifier'
          WHERE g.type IN ('global_statement', 'nonlocal_statement'))
        SELECT b.fact_id, b.try, b.idx, b.stmt, b.start_line, b.start_column, b.end_line,
          b.end_column
        FROM try_block_statements b
        JOIN try_block_nodes a ON a.fact_id = b.fact_id AND a.parent_id = b.stmt
          AND a.type = 'assignment' AND a.children_count = 3 AND b.type = 'expression_statement'
        JOIN try_block_nodes l ON l.fact_id = b.fact_id AND l.parent_id = a.node_id
          AND l.sibling_index = 0 AND l.type = 'identifier'
        JOIN try_block_nodes r ON r.fact_id = b.fact_id AND r.parent_id = a.node_id
          AND r.sibling_index = 2
        WHERE (r.type IN ('integer', 'float', 'true', 'false', 'none', 'ellipsis')
               OR (r.type = 'string' AND NOT regexp_matches(r.name, '^[a-zA-Z]*[fF]'))
               OR (r.type = 'concatenated_string' AND NOT EXISTS (
                     SELECT 1 FROM try_block_nodes x WHERE x.fact_id = r.fact_id
                       AND x.node_id > r.node_id AND x.node_id <= r.node_id + r.descendant_count
                       AND x.type = 'interpolation')))
          AND NOT EXISTS (SELECT 1 FROM externals e WHERE e.fact_id = b.fact_id
                          AND e.function = b.function AND e.name = l.name)
          AND NOT EXISTS (
            SELECT 1 FROM try_block_nodes c JOIN source_files z ON z.path = c.path
            WHERE c.fact_id = b.fact_id AND c.parent_id = b.block AND c.type = 'comment'
              AND c.node_id > b.stmt AND c.node_id < b.next
              AND contains(syntax_slice(z.content, c.start_line, c.start_column, c.end_line,
                                        c.end_column), '# type:'))
    """,
    "leading": """
        SELECT b.fact_id, b.try, coalesce(min(b.idx) FILTER (WHERE q.idx IS NULL), count(*)) AS
          count
        FROM try_block_statements b
        LEFT JOIN try_block_setup q ON q.fact_id = b.fact_id AND q.try = b.try AND q.idx = b.idx
        GROUP BY b.fact_id, b.try
    """,
    "raising": """
        SELECT DISTINCT b.fact_id, b.try
        FROM try_block_leading l
        JOIN try_block_statements b ON b.fact_id = l.fact_id AND b.try = l.try AND b.idx = l.count
        JOIN try_block_nodes n ON n.fact_id = b.fact_id AND n.node_id >= b.stmt
          AND n.node_id <= b.last AND n.home = b.scope
          AND list_contains(['call', 'attribute', 'subscript', 'binary_operator',
                             'comparison_operator', 'await', 'list_comprehension',
                             'set_comprehension', 'dictionary_comprehension',
                               'generator_expression',
                             'unary_operator', 'import_statement', 'import_from_statement',
                             'future_import_statement', 'assert_statement', 'raise_statement',
                             'for_statement', 'with_statement', 'augmented_assignment'], n.type)
        WHERE l.count > 0 AND NOT EXISTS (
          SELECT 1 FROM try_block_nodes t
          JOIN try_block_nodes a ON a.fact_id = t.fact_id AND a.node_id = t.parent_id
            AND a.type = 'assignment'
          WHERE t.fact_id = n.fact_id AND t.sibling_index = 0 AND t.node_id <= n.node_id
            AND n.node_id <= t.node_id + t.descendant_count)
    """,
    "clauses": """
        SELECT t.fact_id, t.path, t.try, b.part, 'tryblockfact:' || t.path || '/regions:' ||
          t.ordinal AS region,
               row_number() OVER (PARTITION BY b.fact_id, b.try, b.part ORDER BY s.node_id) - 1 AS
                 ordinal,
               CASE WHEN s.type = 'return_statement' THEN 'return' ELSE 'statement' END AS kind,
               syntax_slice(z.content, s.start_line, s.start_column, s.end_line, s.end_column) AS
                 text,
               s.start_line, s.start_column, s.end_line, s.end_column
        FROM try_block_parts b
        JOIN try_block_tries t ON t.fact_id = b.fact_id AND t.try = b.try
        JOIN try_block_nodes s ON s.fact_id = b.fact_id AND s.parent_id = b.block AND s.type <>
          'comment'
        JOIN source_files z ON z.path = t.path
        WHERE b.part < 1000
    """,
    "regions": f"""
        SELECT 'tryblockfact:' || t.path AS fact_id, 'tryblockfact:' || t.path AS parent_id,
               {_REGION} AS record_id, t.ordinal,
               t.path || ':' || t.start_line || ':' || t.start_column || ':try' AS statement_id,
               t.path AS statement_span_path, t.start_line AS statement_span_start_line,
               t.start_column AS statement_span_start_column, t.end_line AS
                 statement_span_end_line,
               t.end_column AS statement_span_end_column, h.has_else, h.has_finally,
               h.is_exception_group, coalesce(l.count, 0) AS leading_literal_assignment_count,
               r.try IS NOT NULL AS has_following_raising_operation,
               coalesce(c.name, '') AS protected_call_qualified_name
        FROM try_block_tries t
        JOIN try_block_shape h ON h.fact_id = t.fact_id AND h.try = t.try
        LEFT JOIN try_block_leading l ON l.fact_id = t.fact_id AND l.try = t.try
        LEFT JOIN try_block_raising r ON r.fact_id = t.fact_id AND r.try = t.try
        LEFT JOIN try_block_callee c ON c.fact_id = t.fact_id AND c.try = t.try
    """,
    "regions.handlers": f"""
        SELECT 'tryblockfact:' || t.path AS fact_id, {_REGION} AS parent_id,
               {_REGION} || '/regions.handlers:' || h.ordinal AS record_id, h.ordinal,
               CASE WHEN h.expr IS NULL THEN ''
                    WHEN h.kind IN ('identifier', 'attribute') THEN h.expr_name
                    ELSE syntax_slice(z.content, h.sl, h.sc, h.el, h.ec) END AS caught,
               h.caught_import, coalesce(h.kind = 'tuple', false) AS caught_is_tuple, h.alias
        FROM try_block_caught h
        JOIN try_block_tries t ON t.fact_id = h.fact_id AND t.try = h.try
        JOIN source_files z ON z.path = t.path
    """,
    "regions.handlers.body": """
        SELECT 'tryblockfact:' || path AS fact_id, region || '/regions.handlers:' || (part - 1) AS
          parent_id,
               region || '/regions.handlers:' || (part - 1) || '/regions.handlers.body:' || ordinal
                 AS record_id, ordinal, kind, text, path AS span_path, start_line AS
                   span_start_line,
               start_column AS span_start_column, end_line AS span_end_line, end_column AS
                 span_end_column
        FROM try_block_clauses WHERE part > 0
    """,
    "regions.protected_statements": """
        SELECT 'tryblockfact:' || path AS fact_id, region AS parent_id,
               region || '/regions.protected_statements:' || ordinal AS record_id, ordinal, kind,
               text, path AS span_path, start_line AS span_start_line,
               start_column AS span_start_column, end_line AS span_end_line, end_column AS
                 span_end_column
        FROM try_block_clauses WHERE part = 0
    """,
    "regions.leading_assignments": f"""
        SELECT 'tryblockfact:' || t.path AS fact_id, {_REGION} AS parent_id,
               {_REGION} || '/regions.leading_assignments:' || q.idx AS record_id, q.idx AS
                 ordinal,
               t.path AS span_path, q.start_line AS span_start_line,
               q.start_column AS span_start_column, q.end_line AS span_end_line,
               q.end_column AS span_end_column
        FROM try_block_setup q
        JOIN try_block_leading l ON l.fact_id = q.fact_id AND l.try = q.try AND q.idx < l.count
        JOIN try_block_tries t ON t.fact_id = q.fact_id AND t.try = q.try
    """,
}


class TryBlock(Family):
    family = "TryBlockFact"
    relations = _TRIES
