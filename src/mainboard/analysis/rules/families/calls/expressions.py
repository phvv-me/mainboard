# A call's expression trees: its arguments, receiver and nested values, walked from the syntax
# tree, as the CallFact family's SQL relations.

_SPINE = r"""
    WITH RECURSIVE walk(file, call_node, cur, parts, plain) AS (
      SELECT file, call_node, callee_node, []::VARCHAR[], true FROM call_sites
      UNION ALL
      SELECT w.file, w.call_node, c.node,
             CASE WHEN n.type = 'attribute' THEN list_prepend(a.name, w.parts) ELSE w.parts END,
             w.plain AND n.type IN ('attribute', 'list_splat')
      FROM walk w
      JOIN call_nodes n ON n.file = w.file AND n.node = w.cur
        AND n.type IN ('attribute', 'call', 'list_splat')
      JOIN call_nodes c ON c.file = n.file AND c.parent = n.node
        AND c.sibling_index = CASE n.type WHEN 'list_splat' THEN 1 ELSE 0 END
      LEFT JOIN call_nodes a ON a.file = n.file AND a.parent = n.node
        AND a.sibling_index = n.children_count - 1 AND a.type = 'identifier'
    )
    SELECT w.file, w.call_node, CASE WHEN r.type = 'identifier' THEN r.name END AS root_name,
           w.parts, w.plain
    FROM walk w JOIN call_nodes r ON r.file = w.file AND r.node = w.cur
    WHERE r.type NOT IN ('attribute', 'call', 'list_splat')
"""

_ROOTS = r"""
    WITH args AS (
      SELECT s.file, s.call_node, a.node, a.type, a.sibling_index, a.start_line, a.start_column,
             a.end_line, a.end_column
      FROM call_sites s
      JOIN call_nodes a ON a.file = s.file AND a.parent = s.arglist_node
      WHERE s.arglist_type = 'argument_list'
        AND a.type NOT IN ('keyword_argument', 'dictionary_splat', 'comment', '(', ')', ',')
      UNION ALL
      SELECT file, call_node, arglist_node, 'generator_expression', 0, arglist_start_line,
             arglist_start_column + 1, arglist_end_line, arglist_end_column - 1
      FROM call_sites WHERE arglist_type = 'generator_expression'
      UNION ALL
      SELECT s.file, s.call_node, coalesce(q.node, o.node), coalesce(q.type, o.type), -1,
             coalesce(q.start_line, o.start_line), coalesce(q.start_column, o.start_column),
             o.end_line, o.end_column
      FROM call_sites s
      JOIN call_nodes o ON o.file = s.file AND o.parent = s.callee_node AND o.sibling_index = 0
      LEFT JOIN call_nodes q ON q.file = o.file AND q.parent = o.node AND o.type = 'list_splat'
        AND q.sibling_index = 1
      WHERE s.callee_type = 'attribute'
    ), unwrapped AS (
      SELECT p.file, p.call_node, p.sibling_index,
             coalesce(i2.node, i1.node, p.node) AS node,
             coalesce(i2.type, i1.type, p.type) AS type,
             coalesce(i2.start_line, i1.start_line, p.start_line) AS start_line,
             coalesce(i2.start_column, i1.start_column, p.start_column) AS start_column,
             coalesce(i2.end_line, i1.end_line, p.end_line) AS end_line,
             coalesce(i2.end_column, i1.end_column, p.end_column) AS end_column
      FROM args p
      LEFT JOIN call_nodes i1 ON i1.file = p.file AND i1.parent = p.node
        AND p.type = 'parenthesized_expression' AND i1.sibling_index = 1
      LEFT JOIN call_nodes i2 ON i2.file = i1.file AND i2.parent = i1.node
        AND i1.type = 'parenthesized_expression' AND i2.sibling_index = 1
    )
    SELECT file, call_node,
           CASE WHEN sibling_index = -1 THEN 'receiver' ELSE 'argument' END AS relation,
           CASE WHEN sibling_index = -1 THEN 0
                ELSE row_number() OVER (PARTITION BY file, call_node, sibling_index = -1
                                        ORDER BY sibling_index) - 1 END::UBIGINT AS ordinal,
           node AS node, type, start_line, start_column, end_line, end_column
    FROM unwrapped
"""

_EDGES = r"""
    SELECT file, call_node AS parent, node AS child, relation, ordinal, type, start_line,
           start_column, end_line, end_column, NULL::INT AS entry
    FROM call_roots WHERE relation = 'argument'
    UNION ALL
    SELECT w.file, w.parent, coalesce(i2.node, i1.node, w.node) AS child,
           'mapping_value',
           (row_number() OVER (PARTITION BY w.file, w.parent ORDER BY w.entry) - 1)::UBIGINT,
           coalesce(i2.type, i1.type, w.type),
           coalesce(i2.start_line, i1.start_line, w.start_line),
           coalesce(i2.start_column, i1.start_column, w.start_column),
           coalesce(i2.end_line, i1.end_line, w.end_line),
           coalesce(i2.end_column, i1.end_column, w.end_column), w.entry_node
    FROM (
      SELECT d.file, d.node AS parent, e.sibling_index AS entry, e.node AS entry_node,
             v.node, v.type, v.start_line, v.start_column, v.end_line, v.end_column
      FROM call_nodes d
      JOIN call_nodes e ON e.file = d.file AND e.parent = d.node
        AND e.type IN ('pair', 'dictionary_splat')
      JOIN call_nodes v ON v.file = e.file AND v.parent = e.node
        AND v.sibling_index = (e.children_count - 1)
      WHERE d.type = 'dictionary') w
    LEFT JOIN call_nodes i1 ON i1.file = w.file AND i1.parent = w.node
      AND w.type = 'parenthesized_expression' AND i1.sibling_index = 1
    LEFT JOIN call_nodes i2 ON i2.file = i1.file AND i2.parent = i1.node
      AND i1.type = 'parenthesized_expression' AND i2.sibling_index = 1
"""

_TREE = r"""
    WITH RECURSIVE tree(file, call_node, node, parent, relation, ordinal, root_relation,
                        root_ordinal, depth, key, start_line, start_column, end_line,
                        end_column, type, entry) AS (
      SELECT file, call_node, node, NULL::INT, relation, ordinal, relation, ordinal, 0,
             [CASE relation WHEN 'argument' THEN 0 ELSE 1 END, ordinal::INT], start_line,
             start_column, end_line, end_column, type, NULL::INT
      FROM call_roots
      UNION ALL
      SELECT t.file, t.call_node, e.child, t.node, e.relation, e.ordinal, t.root_relation,
             t.root_ordinal, t.depth + 1, list_append(t.key, e.ordinal::INT), e.start_line,
             e.start_column, e.end_line, e.end_column, e.type, e.entry
      FROM tree t JOIN call_edges e ON e.file = t.file AND e.parent = t.node
    )
    SELECT file, call_node, node, parent, relation, ordinal, root_relation, root_ordinal, depth,
           (row_number() OVER (PARTITION BY file, call_node ORDER BY key) - 1)::UBIGINT AS idx,
           key, start_line, start_column, end_line, end_column, type, entry
    FROM tree
"""

_KEYS = r"""
    SELECT t.file, t.call_node, t.idx, k.start_line AS key_start_line,
           k.start_column AS key_start_column, k.end_line AS key_end_line,
           k.end_column AS key_end_column, e.type = 'dictionary_splat' AS is_spread
    FROM call_tree t
    JOIN call_nodes e ON e.file = t.file AND e.node = t.entry
    LEFT JOIN call_nodes k ON k.file = e.file AND k.parent = e.node AND k.sibling_index = 0
      AND e.type = 'pair'
    WHERE t.relation = 'mapping_value'
"""

_BOOLS = r"""
    WITH RECURSIVE base AS (
      SELECT file, node FROM call_nodes
      WHERE type IN ('true', 'false', 'comparison_operator', 'not_operator')
    ), operands AS (
      SELECT n.file, n.node, c.node AS operand
      FROM call_nodes n JOIN call_nodes c ON c.file = n.file AND c.parent = n.node
      WHERE (n.type = 'boolean_operator' AND c.sibling_index IN (0, 2))
         OR (n.type = 'conditional_expression' AND c.sibling_index IN (0, 4))
         OR (n.type = 'parenthesized_expression' AND c.sibling_index = 1)
    ), leaves(file, node, leaf) AS (
      SELECT DISTINCT file, node, node FROM operands
      UNION ALL
      SELECT l.file, l.node, o.operand
      FROM leaves l JOIN operands o ON o.file = l.file AND o.node = l.leaf
    )
    SELECT file, node FROM base
    UNION
    SELECT l.file, l.node
    FROM leaves l
    ANTI JOIN operands o ON o.file = l.file AND o.node = l.leaf
    LEFT JOIN base b ON b.file = l.file AND b.node = l.leaf
    GROUP BY l.file, l.node HAVING count(b.node) = count(*)
"""

_EXPRESSIONS = r"""
    SELECT 'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || t.idx AS expression_id,
           'calls:' || f.path || ':call:' || c.ordinal AS call_id,
           (row_number() OVER (ORDER BY f.path, c.ordinal, t.idx) - 1)::UBIGINT AS preorder,
           CASE WHEN t.parent IS NOT NULL THEN
             'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || p.idx END
             AS parent_expression_id,
           t.relation, t.ordinal, t.root_relation, t.root_ordinal, t.depth::UBIGINT AS depth,
           x.text, coalesce(r.qualified_name, '') AS qualified_name,
           CASE WHEN t.type = 'string' THEN
                  CASE WHEN regexp_matches(x.text, '^[A-Za-z]*[fFbB][A-Za-z]*[''"]')
                       THEN 'none' ELSE 'string' END
                WHEN t.type = 'concatenated_string' THEN
                  CASE WHEN regexp_matches(x.text, '(^|[''"\s])[A-Za-z]*[fFbB][A-Za-z]*[''"]')
                       THEN 'none' ELSE 'string' END
                WHEN t.type IN ('integer', 'float') THEN 'number'
                WHEN t.type IN ('true', 'false') THEN 'boolean'
                WHEN t.type = 'dictionary' THEN 'mapping'
                WHEN t.type IN ('list', 'tuple', 'set') THEN 'sequence'
                ELSE 'none' END AS literal_kind,
           CASE WHEN bl.node IS NOT NULL THEN 'bool' ELSE '' END AS resolved_type,
           f.path AS node_path, t.start_line AS node_start_line,
           t.start_column AS node_start_column,
           t.end_line AS node_end_line, t.end_column AS node_end_column, 'expression' AS node_kind,
           x.text AS node_text
    FROM call_tree t
    JOIN call_sites c ON c.file = t.file AND c.call_node = t.call_node
    JOIN call_files f ON f.file = t.file
    LEFT JOIN call_tree p ON p.file = t.file AND p.call_node = t.call_node AND p.node = t.parent
    JOIN call_texts x ON x.file = t.file AND x.start_line = t.start_line
      AND x.start_column = t.start_column AND x.end_line = t.end_line
      AND x.end_column = t.end_column
    LEFT JOIN call_resolved r ON r.file = t.file AND r.call_node = t.node AND t.type = 'call'
    LEFT JOIN call_bools bl ON bl.file = t.file AND bl.node = t.node
    ORDER BY f.path, c.ordinal, t.idx
"""

_ANCESTRY = r"""
    SELECT 'calls:' || f.path || ':call:' || c.ordinal AS call_id,
           'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || e.idx
             AS descendant_expression_id,
           a.depth::UBIGINT AS step,
           CASE WHEN a.depth = 0 THEN 'calls:' || f.path || ':call:' || c.ordinal
                ELSE 'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || p.idx END
             AS parent_id,
           CASE WHEN a.depth = 0 THEN 'call' ELSE 'expression' END AS parent_kind,
           'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || a.idx
             AS child_expression_id,
           a.relation, a.ordinal
    FROM call_tree e
    JOIN call_tree a ON a.file = e.file AND a.call_node = e.call_node
      AND list_slice(e.key, 1, len(a.key)) = a.key
    LEFT JOIN call_tree p ON p.file = a.file AND p.call_node = a.call_node AND p.node = a.parent
    JOIN call_sites c ON c.file = e.file AND c.call_node = e.call_node
    JOIN call_files f ON f.file = e.file
"""

_MAPPING_ENTRIES = r"""
    SELECT 'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || d.idx AS expression_id,
           t.ordinal, CASE WHEN k.is_spread THEN '' ELSE trim(x.text, chr(34) || chr(39)) END
             AS key,
           k.is_spread,
           NOT k.is_spread AND left(coalesce(x.text, ''), 1) IN (chr(34), chr(39))
             AS is_string_key,
           'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || t.idx
             AS value_expression_id
    FROM call_tree t
    JOIN call_keys k ON k.file = t.file AND k.call_node = t.call_node AND k.idx = t.idx
    JOIN call_tree d ON d.file = t.file AND d.call_node = t.call_node AND d.node = t.parent
    JOIN call_sites c ON c.file = t.file AND c.call_node = t.call_node
    JOIN call_files f ON f.file = t.file
    LEFT JOIN call_texts x ON x.file = k.file AND x.start_line = k.key_start_line
      AND x.start_column = k.key_start_column AND x.end_line = k.key_end_line
      AND x.end_column = k.key_end_column
    WHERE t.relation = 'mapping_value'
    ORDER BY f.path, c.ordinal, d.idx, t.ordinal
"""


def expressions() -> dict[str, str]:
    """The relations walking call expressions, by relation name."""
    return {
        "spine": _SPINE,
        "roots": _ROOTS,
        "edges": _EDGES,
        "tree": _TREE,
        "keys": _KEYS,
        "bools": _BOOLS,
        "expressions": _EXPRESSIONS,
        "expression_ancestry": _ANCESTRY,
        "mapping_entries": _MAPPING_ENTRIES,
    }
