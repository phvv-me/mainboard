"""`core_call` (calls and constructions) and `core_call_arg` (the arguments of a call)."""

from ..families.calls_sql import BUILTINS, listed

_PLAIN = r"'^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$'"

CALL = f"""
    WITH callees AS (
      SELECT c.file_id, c.node_id AS call_id, k.node_id AS callee_id, k.type AS callee_type,
             k.name AS callee
      FROM core_node c
      JOIN core_node k ON k.file_id = c.file_id AND k.parent_id = c.node_id
                      AND k.sibling_index < 2 AND k.flags % 2 = 0
      WHERE c.semantic_type = 'COMPUTATION_CALL' AND c.flags % 2 = 0 AND c.children_count > 0
      QUALIFY row_number() OVER (PARTITION BY c.file_id, c.node_id ORDER BY k.sibling_index) = 1
    ), bindings AS (
      SELECT file_id, name, min(target) AS target FROM core_import GROUP BY file_id, name
    )
    SELECT c.file_id, c.node_id AS call_id, f.path, f.language, f.parses, c.start_line,
           c.start_column, c.end_line, c.end_column, c.descendant_count AS size, c.parent_id,
           c.owner_id, c.function_id, c.class_id, k.callee_id, k.callee_type, k.callee,
           k.callee_type IN ('identifier', 'attribute', 'member_expression')
           AND regexp_matches(k.callee, {_PLAIN}) AS plain,
           CASE WHEN p1.type <> 'parenthesized_expression' THEN p1.type
                WHEN p2.type <> 'parenthesized_expression' THEN p2.type ELSE p3.type END
             AS top_type,
           CASE WHEN p1.type <> 'parenthesized_expression' THEN p1.children_count
                WHEN p2.type <> 'parenthesized_expression' THEN p2.children_count
                ELSE p3.children_count END AS top_children,
           CASE WHEN NOT plain THEN NULL
                WHEN b.target IS NOT NULL
                THEN b.target || substr(k.callee, length(split_part(k.callee, '.', 1)) + 1)
                WHEN f.language = 'python' AND NOT contains(k.callee, '.')
                     AND list_contains({listed(BUILTINS)}, k.callee)
                THEN 'builtins.' || k.callee END AS qualified_name
    FROM core_node c
    JOIN callees k ON k.file_id = c.file_id AND k.call_id = c.node_id
    JOIN core_file f ON f.file_id = c.file_id
    LEFT JOIN core_node p1 ON p1.file_id = c.file_id AND p1.node_id = c.parent_id
    LEFT JOIN core_node p2 ON p2.file_id = c.file_id AND p2.node_id = p1.parent_id
    LEFT JOIN core_node p3 ON p3.file_id = c.file_id AND p3.node_id = p2.parent_id
    LEFT JOIN bindings b ON b.file_id = c.file_id AND b.name = split_part(k.callee, '.', 1)
"""

CALL_ARG = """
    SELECT c.file_id, c.call_id, c.path, a.node_id, a.type, a.name, a.descendant_count AS size,
           row_number() OVER (PARTITION BY c.file_id, c.call_id ORDER BY a.sibling_index) - 1
             AS ordinal,
           a.type = 'keyword_argument' AS is_keyword, a.start_line, a.start_column, a.end_line,
           a.end_column
    FROM core_call c
    JOIN core_node l ON l.file_id = c.file_id AND l.parent_id = c.call_id
                    AND l.type IN ('argument_list', 'arguments')
    JOIN core_node a ON a.file_id = l.file_id AND a.parent_id = l.node_id
                    AND a.semantic_type NOT IN ('PARSER_DELIMITER', 'METADATA_COMMENT')
"""
