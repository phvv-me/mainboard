"""What the syntax trees say about the body of each test: its statements, its shape and the
module state it writes.

TestFunctionFact's first relations, each reading the earlier ones by their view names
(`test_function_nodes` for `nodes`); `fact_id` is the syntax tree file's, a node is named by its
`node_id` in it. A test is a `def test...` anywhere in a file whose path follows a test
convention; a runner collects it when it sits among the module's functions or in a `Test` class
without an initializer. A nested test is a test of its own whose body its enclosing test's body
also holds. A test's body shape is its syntax tree read out in order, a literal (string, number,
boolean, `None`) as `?`, so two bodies match when they differ in nothing but literals, however
they are formatted or commented (mcmr compares the normalized source text).
"""

BODIES = {
    "nodes": """
        WITH n AS (
          SELECT fact_id, path, node_id, parent_id, type, name, modifiers, receiver, sibling_index,
                 children_count, descendant_count, start_line, start_column, end_line, end_column,
                 tests_home(scope) AS owner
          FROM ast_nodes SEMI JOIN test_function_facts USING (path))
        SELECT n.*, p.type AS ptype
        FROM n LEFT JOIN n p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
    """,
    "initializers": """
        SELECT i.fact_id,
               CASE WHEN i.ptype = 'decorated_definition' THEN p.parent_id ELSE p.node_id END
                 AS block
        FROM test_function_nodes i
        JOIN test_function_nodes p ON p.fact_id = i.fact_id AND p.node_id = i.parent_id
        WHERE i.type = 'function_definition' AND i.name = '__init__'
    """,
    "declared": """
        WITH placed AS (
          SELECT n.fact_id, n.path, n.node_id AS test, n.name,
                 n.node_id + n.descendant_count AS last,
                 list_contains(n.modifiers, 'async') AS is_async, n.end_line, n.end_column,
                 CASE WHEN n.ptype = 'decorated_definition' THEN p.start_line ELSE n.start_line END
                   AS start_line,
                 CASE WHEN n.ptype = 'decorated_definition' THEN p.start_column
                      ELSE n.start_column END AS start_column,
                 CASE WHEN n.ptype = 'decorated_definition' THEN p.parent_id ELSE p.node_id END
                   AS holder
          FROM test_function_nodes n
          JOIN test_function_nodes p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
          WHERE n.type = 'function_definition' AND starts_with(n.name, 'test'))
        SELECT d.*, b.node_id AS body,
               coalesce(h.type = 'module'
                 OR (c.node_id IS NOT NULL AND starts_with(c.name, 'Test')
                     AND (cp.type = 'module'
                          OR (cp.type = 'decorated_definition' AND cp.parent_id = 0))
                     AND NOT EXISTS (SELECT 1 FROM test_function_initializers i
                                     WHERE i.fact_id = d.fact_id AND i.block = h.node_id)),
                        false) AS collected,
               row_number() OVER (PARTITION BY d.fact_id ORDER BY d.test) - 1 AS ordinal
        FROM placed d
        JOIN test_function_nodes h ON h.fact_id = d.fact_id AND h.node_id = d.holder
        LEFT JOIN test_function_nodes c ON c.fact_id = d.fact_id AND c.node_id = h.parent_id
          AND h.type = 'block' AND c.type = 'class_definition'
        LEFT JOIN test_function_nodes cp ON cp.fact_id = d.fact_id AND cp.node_id = c.parent_id
        JOIN test_function_nodes b ON b.fact_id = d.fact_id AND b.parent_id = d.test
          AND b.type = 'block'
    """,
    "inside": """
        SELECT x.*, t.test AS in_test, t.body AS in_body
        FROM test_function_nodes x
        JOIN test_function_declared t ON x.fact_id = t.fact_id
          AND x.node_id BETWEEN t.test AND t.last
    """,
    "roots": """
        WITH impure AS (
          SELECT fact_id, parent_id AS node_id,
                 bool_or(regexp_matches(name, '^[a-zA-Z]*[fFbB]')) AS impure
          FROM test_function_nodes WHERE type = 'string' AND ptype = 'concatenated_string'
          GROUP BY fact_id, parent_id)
        SELECT i.*, CASE
                 WHEN i.type IN ('integer', 'float', 'true', 'false', 'none') THEN true
                 WHEN i.type = 'string' THEN i.ptype <> 'concatenated_string'
                      AND NOT regexp_matches(i.name, '^[a-zA-Z]*[fFbB]')
                 WHEN i.type = 'concatenated_string' THEN NOT coalesce(c.impure, false)
                 ELSE false END AS literal
        FROM test_function_inside i
        LEFT JOIN impure c ON c.fact_id = i.fact_id AND c.node_id = i.node_id
        WHERE i.type <> 'comment' AND i.node_id > i.in_body
    """,
    "shapes": """
        WITH kept AS (
          SELECT * FROM (
            SELECT r.*, max(CASE WHEN r.literal THEN r.node_id + r.descendant_count END)
                          OVER (PARTITION BY r.fact_id, r.in_test ORDER BY r.node_id
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS covered
            FROM test_function_roots r)
          WHERE covered IS NULL OR covered < node_id)
        SELECT fact_id, path, in_test AS test,
               string_agg(CASE WHEN literal THEN '?'
                               WHEN children_count = 0 THEN coalesce(nullif(name, ''), type)
                               ELSE type END, ' ' ORDER BY node_id) AS body_shape,
               list(name ORDER BY node_id) FILTER (WHERE literal) AS literals
        FROM kept GROUP BY fact_id, path, in_test
    """,
    "statements": """
        SELECT * FROM test_function_inside
        WHERE ptype = 'block' AND type NOT IN ('comment', 'case_clause')
    """,
    "docstrings": """
        SELECT DISTINCT s.fact_id, s.in_test AS test
        FROM test_function_statements s
        JOIN test_function_roots r ON r.fact_id = s.fact_id AND r.parent_id = s.node_id
          AND r.literal
          AND r.type IN ('string', 'concatenated_string')
        WHERE s.type = 'expression_statement' AND s.owner = s.in_test
          AND s.node_id = (SELECT min(f.node_id) FROM test_function_statements f
                           WHERE f.fact_id = s.fact_id AND f.in_test = s.in_test
                             AND f.parent_id = s.in_body)
    """,
    "modular": """
        WITH RECURSIVE assignments AS (
          SELECT fact_id, node_id, parent_id, children_count FROM test_function_nodes
          WHERE type = 'assignment' AND owner = -1
        ), walk AS (
          SELECT a.fact_id, a.node_id, a.children_count FROM assignments a
          JOIN test_function_nodes s ON s.fact_id = a.fact_id AND s.node_id = a.parent_id
            AND s.type = 'expression_statement' AND s.parent_id = 0
          UNION ALL
          SELECT c.fact_id, c.node_id, c.children_count FROM walk m
          JOIN assignments c ON c.fact_id = m.fact_id AND c.parent_id = m.node_id)
        SELECT m.fact_id, m.node_id, m.children_count, l.name,
               EXISTS (SELECT 1 FROM test_function_nodes r
                       WHERE r.fact_id = m.fact_id AND r.parent_id = m.node_id
                         AND r.sibling_index = m.children_count - 1
                         AND r.type IN ('list_comprehension', 'set_comprehension',
                                        'generator_expression')) AS generated
        FROM walk m
        JOIN test_function_nodes l ON l.fact_id = m.fact_id AND l.parent_id = m.node_id
          AND l.sibling_index = 0 AND l.type = 'identifier'
    """,
    "mutations": """
        SELECT g.fact_id, g.in_test AS test, count(i.node_id) AS n
        FROM test_function_inside g
        JOIN test_function_nodes i ON i.fact_id = g.fact_id AND i.parent_id = g.node_id
          AND i.type = 'identifier'
        WHERE g.type = 'global_statement' AND g.owner = g.in_test
        GROUP BY g.fact_id, g.in_test
        UNION ALL
        SELECT a.fact_id, a.in_test, count(*)
        FROM test_function_inside a
        JOIN test_function_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
          AND l.sibling_index = 0 AND l.type IN ('subscript', 'attribute')
        JOIN test_function_nodes b ON b.fact_id = a.fact_id AND b.parent_id = l.node_id
          AND b.sibling_index = 0 AND b.type = 'identifier'
        JOIN test_function_modular s ON s.fact_id = a.fact_id AND s.name = b.name
        WHERE a.type IN ('assignment', 'augmented_assignment') AND a.owner = a.in_test
        GROUP BY a.fact_id, a.in_test
        UNION ALL
        SELECT c.fact_id, c.in_test, count(*)
        FROM test_function_inside c
        JOIN test_function_modular s ON s.fact_id = c.fact_id AND s.name = c.receiver
        WHERE c.type = 'call' AND c.owner = c.in_test
          AND list_contains(['append', 'extend', 'insert', 'pop', 'remove', 'clear', 'update',
                             'add',
                             'sort', 'setdefault', 'popitem', 'discard'], c.name)
        GROUP BY c.fact_id, c.in_test
    """,
    "counts": """
        SELECT t.fact_id, t.test,
               (SELECT count(*) FROM test_function_statements s
                WHERE s.fact_id = t.fact_id AND s.in_test = t.test AND s.owner = t.test)
                 - (CASE WHEN EXISTS (SELECT 1 FROM test_function_docstrings d
                                      WHERE d.fact_id = t.fact_id AND d.test = t.test)
                         THEN 1 ELSE 0 END) AS owned_statement_count,
               (SELECT count(*) FROM test_function_statements s
                WHERE s.fact_id = t.fact_id AND s.in_test = t.test AND s.owner = t.test
                  AND s.type = 'if_statement') AS owned_conditional_count,
               coalesce((SELECT sum(m.n) FROM test_function_mutations m
                         WHERE m.fact_id = t.fact_id AND m.test = t.test), 0)
                 AS module_state_mutation_count
        FROM test_function_declared t
    """,
}
