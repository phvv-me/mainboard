"""What each method and class does: stubs, forwarding, aliases, receiver use.

Judgments mcmr's Rust passes made from ruff's AST, here read off sitting_duck's tree below each
member of `class_member_defs`. Follows `classes_syntax` in the family's relation order.
"""

STATEMENTS = """
    SELECT * EXCLUDE (head_type),
           pos = 1 AND stmt_type = 'expression_statement' AND head_type = 'string' AS is_docstring
    FROM (
      SELECT m.fact_id, m.class_node, m.method_node, s.node_id AS stmt, s.type AS stmt_type,
             e.type AS head_type,
             row_number() OVER (PARTITION BY m.fact_id, m.method_node ORDER BY s.node_id) AS pos
      FROM class_member_defs m
      JOIN class_nodes b ON b.fact_id = m.fact_id AND b.parent_id = m.method_node
                        AND b.type = 'block'
      JOIN class_nodes s ON s.fact_id = m.fact_id AND s.parent_id = b.node_id
                        AND s.type <> 'comment'
      LEFT JOIN class_nodes e ON e.fact_id = m.fact_id AND e.parent_id = s.node_id
                             AND e.sibling_index = 0 AND s.type = 'expression_statement')
"""

STUBS = """
    SELECT m.fact_id, m.method_node, bool_and(
             s.stmt_type = 'pass_statement'
             OR (s.stmt_type = 'expression_statement' AND e.type = 'ellipsis')
             OR (s.stmt_type = 'raise_statement'
                 AND coalesce(ends_with(r.name, 'NotImplementedError'), false))) AS is_stub
    FROM (SELECT DISTINCT fact_id, method_node FROM class_statements) m
    LEFT JOIN class_statements s ON s.fact_id = m.fact_id AND s.method_node = m.method_node
                                AND NOT s.is_docstring
    LEFT JOIN class_nodes e ON e.fact_id = s.fact_id AND e.parent_id = s.stmt
                           AND e.sibling_index = 0 AND s.stmt_type = 'expression_statement'
    LEFT JOIN class_nodes r ON r.fact_id = s.fact_id AND r.parent_id = s.stmt
                           AND r.sibling_index = 1 AND s.stmt_type = 'raise_statement'
    GROUP BY m.fact_id, m.method_node
"""

SUPER_DELEGATES = r"""
    SELECT DISTINCT m.fact_id, m.method_node
    FROM class_member_defs m
    JOIN class_nodes a ON a.fact_id = m.fact_id AND a.type = 'attribute'
                      AND a.node_id BETWEEN m.method_node AND m.method_node + m.descendant_count
                      AND starts_with(a.name, 'super(') AND ends_with(a.name, '.' || m.name)
"""

FORWARDS = r"""
    WITH single AS (
      SELECT s.fact_id, s.method_node, s.stmt
      FROM class_statements s
      WHERE NOT s.is_docstring AND s.stmt_type = 'return_statement'
        AND (SELECT count(*) FROM class_statements o
             WHERE o.fact_id = s.fact_id AND o.method_node = s.method_node
               AND NOT o.is_docstring) = 1
    ), calls AS (
      SELECT s.fact_id, s.method_node, c.node_id AS call_node
      FROM single s
      JOIN class_member_defs m ON m.fact_id = s.fact_id AND m.method_node = s.method_node
      JOIN class_nodes c ON c.fact_id = s.fact_id AND c.parent_id = s.stmt AND c.type = 'call'
      JOIN class_nodes f ON f.fact_id = c.fact_id AND f.parent_id = c.node_id
                        AND f.sibling_index = 0 AND f.type = 'attribute'
                        AND ends_with(f.name, '.' || m.name)
      JOIN class_nodes r ON r.fact_id = f.fact_id AND r.parent_id = f.node_id
                        AND r.sibling_index = 0 AND r.type = 'call' AND r.name = 'super'
      JOIN class_nodes ra ON ra.fact_id = r.fact_id AND ra.parent_id = r.node_id
                         AND ra.type = 'argument_list' AND ra.children_count = 2
      WHERE regexp_matches(f.name, '^super\(\)\.')
    ), passed AS (
      SELECT c.fact_id, c.method_node,
             coalesce(list(CASE a.type WHEN 'identifier' THEN a.name
                               WHEN 'list_splat' THEN '*' || coalesce(i.name, '')
                               ELSE '' END ORDER BY a.sibling_index)
                        FILTER (WHERE a.type NOT IN ('keyword_argument', 'dictionary_splat')),
                      []) AS positional,
             coalesce(list(DISTINCT CASE a.type
                               WHEN 'keyword_argument' THEN
                                 CASE WHEN v.type = 'identifier' AND v.name = a.name
                                      THEN a.name ELSE '' END
                               WHEN 'dictionary_splat' THEN '**' || coalesce(i.name, '') END)
                        FILTER (WHERE a.type IN ('keyword_argument', 'dictionary_splat')),
                      []) AS keywords
      FROM calls c
      JOIN class_nodes l ON l.fact_id = c.fact_id AND l.parent_id = c.call_node
                        AND l.type = 'argument_list'
      LEFT JOIN class_nodes a ON a.fact_id = c.fact_id AND a.parent_id = l.node_id
                             AND a.type NOT IN ('(', ')', ',', 'comment')
      LEFT JOIN class_nodes i ON i.fact_id = a.fact_id AND i.parent_id = a.node_id
                             AND i.type = 'identifier'
                             AND a.type IN ('list_splat', 'dictionary_splat')
      LEFT JOIN class_nodes v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
                             AND v.sibling_index = 2 AND a.type = 'keyword_argument'
      GROUP BY c.fact_id, c.method_node
    ), declared AS (
      SELECT p.fact_id, p.method_node,
             coalesce(list(p.name ORDER BY p.idx)
                      FILTER (WHERE p.kind IN ('positional', 'vararg')
                              AND p.idx > (SELECT min(q.idx) FROM class_member_params q
                                           WHERE q.fact_id = p.fact_id
                                             AND q.method_node = p.method_node
                                             AND q.kind = 'positional')), []) AS positional,
             coalesce(list(DISTINCT p.name) FILTER (WHERE p.kind IN ('kwonly', 'kwarg')), [])
               AS keywords
      FROM class_member_params p GROUP BY p.fact_id, p.method_node
    )
    SELECT c.fact_id, c.method_node
    FROM calls c
    JOIN passed p ON p.fact_id = c.fact_id AND p.method_node = c.method_node
    JOIN declared d ON d.fact_id = c.fact_id AND d.method_node = c.method_node
    WHERE d.positional = p.positional AND list_sort(d.keywords) = list_sort(p.keywords)
"""

BODIES = """
    SELECT s.fact_id, s.class_node, s.stmt, s.stmt_type, s.pos,
           coalesce(s.pos = 1 AND s.head_type = 'string', false) AS is_docstring,
           s.stmt_type = 'pass_statement' OR coalesce(s.head_type = 'ellipsis', false)
             AS is_placeholder,
           f.method_node IS NOT NULL AS forwards
    FROM (
      SELECT b.fact_id, b.class_node, t.node_id AS stmt, t.type AS stmt_type, e.type AS head_type,
             row_number() OVER (PARTITION BY b.fact_id, b.class_node ORDER BY t.node_id) AS pos,
             CASE t.type WHEN 'function_definition' THEN t.node_id ELSE i.node_id END AS def_node
      FROM class_blocks b
      JOIN class_nodes t ON t.fact_id = b.fact_id AND t.parent_id = b.block_node
                        AND t.type <> 'comment'
      LEFT JOIN class_nodes e ON e.fact_id = t.fact_id AND e.parent_id = t.node_id
                             AND e.sibling_index = 0 AND t.type = 'expression_statement'
      LEFT JOIN class_nodes i ON i.fact_id = t.fact_id AND i.parent_id = t.node_id
                             AND t.type = 'decorated_definition' AND i.type = 'function_definition'
    ) s
    LEFT JOIN class_forwards f ON f.fact_id = s.fact_id AND f.method_node = s.def_node
"""

PASS_THROUGH = """
    SELECT fact_id, class_node,
           bool_and(is_placeholder)
           OR (count(*) FILTER (WHERE NOT is_docstring) > 0
               AND bool_and(is_docstring OR (stmt_type IN ('function_definition',
                                                           'decorated_definition')
                                             AND forwards))) AS is_pass_through
    FROM class_bodies GROUP BY fact_id, class_node
"""

BEHAVIOUR = """
    SELECT m.fact_id, m.class_node,
           bool_or(NOT (starts_with(m.name, '__') AND ends_with(m.name, '__'))
                   AND m.name <> 'model_post_init' AND NOT coalesce(h.hooked, false))
             AS has_ordinary_behavior
    FROM class_member_defs m
    LEFT JOIN (SELECT fact_id, decorated_node,
                      bool_or(name IN ('computed_field', 'field_serializer', 'field_validator',
                                       'model_serializer', 'model_validator', 'root_validator',
                                       'validator')) AS hooked
               FROM class_decorations GROUP BY fact_id, decorated_node) h
      ON h.fact_id = m.fact_id AND h.decorated_node = m.statement_node
    GROUP BY m.fact_id, m.class_node
"""

INIT_ASSIGNS = r"""
    SELECT m.fact_id, m.class_node, m.method_node, a.node_id AS assign_node,
           r.type AS right_type, r.name AS right_name,
           CASE r.type WHEN 'identifier' THEN r.name ELSE ro.name END AS right_root
    FROM class_member_defs m
    JOIN class_nodes a ON a.fact_id = m.fact_id AND a.type = 'assignment'
                      AND a.node_id BETWEEN m.method_node AND m.method_node + m.descendant_count
    JOIN class_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
                      AND l.sibling_index = 0 AND l.type = 'attribute'
                      AND regexp_matches(l.name, '^self\.\w+$')
    JOIN class_nodes r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
                      AND r.sibling_index = a.children_count - 1
    LEFT JOIN class_nodes ro ON ro.fact_id = r.fact_id AND ro.parent_id = r.node_id
                            AND ro.sibling_index = 0 AND ro.type = 'identifier'
                            AND r.type = 'attribute'
    WHERE m.name IN ('__init__', 'model_post_init')
"""

ALIASES = """
    SELECT x.fact_id, x.class_node, count(*) AS alias_count
    FROM class_init_assigns x
    WHERE x.right_type = 'attribute'
      AND EXISTS (
        SELECT 1 FROM class_init_assigns y
        JOIN class_member_params p ON p.fact_id = y.fact_id AND p.method_node = y.method_node
                                  AND p.name = y.right_name
        WHERE y.fact_id = x.fact_id AND y.method_node = x.method_node
          AND y.right_type = 'identifier' AND y.right_name = x.right_root)
    GROUP BY x.fact_id, x.class_node
"""

RECEIVERS = """
    SELECT p.fact_id, p.class_node, p.method_node, m.class_id, p.name AS receiver
    FROM class_member_params p
    JOIN class_method_defs m ON m.fact_id = p.fact_id AND m.method_node = p.method_node
    WHERE p.kind = 'positional'
      AND p.idx = (SELECT min(q.idx) FROM class_member_params q
                   WHERE q.fact_id = p.fact_id AND q.method_node = p.method_node
                     AND q.kind = 'positional')
      AND NOT list_contains(m.decorator_names, 'staticmethod')
"""

ASSIGN_TARGETS = """
    SELECT t.fact_id, t.node_id AS first, t.node_id + t.descendant_count AS last
    FROM class_nodes a
    JOIN class_nodes t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id AND t.sibling_index = 0
    WHERE a.type IN ('assignment', 'augmented_assignment')
"""

RECEIVER_HITS = """
    SELECT r.fact_id, r.class_node, r.method_node, r.receiver, i.node_id AS hit
    FROM class_receivers r
    JOIN class_nodes b ON b.fact_id = r.fact_id AND b.parent_id = r.method_node AND
          b.type = 'block'
    JOIN class_nodes i ON i.fact_id = r.fact_id AND i.type = 'identifier' AND i.name = r.receiver
                      AND i.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
    JOIN class_nodes q ON q.fact_id = i.fact_id AND q.node_id = i.parent_id
    LEFT JOIN class_assign_targets t ON t.fact_id = i.fact_id
                                    AND i.node_id BETWEEN t.first AND t.last
    WHERE t.first IS NULL AND NOT (q.type = 'attribute' AND i.sibling_index > 0)
      AND NOT (q.type = 'keyword_argument' AND i.sibling_index = 0)
"""

RECEIVER_CHAINS = """
    WITH RECURSIVE chains AS (
      SELECT fact_id, class_node, method_node, receiver, hit, hit AS node, NULL::BIGINT AS top,
             0 AS depth
      FROM class_receiver_hits
      UNION ALL
      SELECT c.fact_id, c.class_node, c.method_node, c.receiver, c.hit, q.node_id,
             CASE WHEN q.type = 'attribute' THEN q.node_id ELSE c.top END, c.depth + 1
      FROM chains c
      JOIN class_nodes n ON n.fact_id = c.fact_id AND n.node_id = c.node
      JOIN class_nodes q ON q.fact_id = n.fact_id AND q.node_id = n.parent_id
                        AND q.type IN ('attribute', 'call', 'subscript') AND n.sibling_index = 0
    )
    SELECT * FROM chains
"""

RECEIVER_STATE = """
    SELECT DISTINCT c.fact_id, c.method_node
    FROM (SELECT fact_id, class_node, method_node, receiver, hit, arg_max(top, depth) AS top
          FROM class_receiver_chains GROUP BY fact_id, class_node, method_node, receiver, hit) c
    LEFT JOIN class_nodes a ON a.fact_id = c.fact_id AND a.parent_id = c.top AND
          a.sibling_index = 1
    WHERE CASE WHEN c.top IS NULL THEN c.receiver <> 'cls'
               ELSE NOT EXISTS (SELECT 1 FROM class_member_defs s
                                WHERE s.fact_id = c.fact_id AND s.class_node = c.class_node
                                  AND s.name = a.name) END
"""

OWNER_CALLS = """
    SELECT m.method_id, m.fact_id, f.name AS value,
           row_number() OVER (PARTITION BY m.method_id ORDER BY c.node_id) - 1 AS ordinal
    FROM class_method_defs m
    JOIN class_defs k ON k.fact_id = m.fact_id AND k.class_node = m.class_node
    JOIN class_nodes c ON c.fact_id = m.fact_id AND c.type = 'call'
                      AND c.node_id BETWEEN m.method_node AND m.method_node + m.descendant_count
    JOIN class_nodes f ON f.fact_id = c.fact_id AND f.parent_id = c.node_id
                      AND f.sibling_index = 0 AND f.type = 'attribute'
                      AND starts_with(f.name, k.name || '.')
    WHERE NOT EXISTS (
      SELECT 1 FROM class_nodes a
      WHERE a.fact_id = m.fact_id AND a.type = 'assignment' AND a.name = k.name
        AND a.node_id BETWEEN m.method_node AND m.method_node + m.descendant_count)
"""

RELATIONS = {
    "statements": STATEMENTS,
    "stubs": STUBS,
    "super_delegates": SUPER_DELEGATES,
    "forwards": FORWARDS,
    "bodies": BODIES,
    "pass_through": PASS_THROUGH,
    "behaviour": BEHAVIOUR,
    "init_assigns": INIT_ASSIGNS,
    "aliases": ALIASES,
    "receivers": RECEIVERS,
    "assign_targets": ASSIGN_TARGETS,
    "receiver_hits": RECEIVER_HITS,
    "receiver_chains": RECEIVER_CHAINS,
    "receiver_state": RECEIVER_STATE,
    "owner_calls": OWNER_CALLS,
}
