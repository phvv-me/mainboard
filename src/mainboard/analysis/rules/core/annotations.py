"""`core_annotation` (the annotations a Python file declares: parameters, returns, annotated
assignments and `type` aliases)."""

ANNOTATION = r"""
    SELECT t.file_id, f.path, f.language, t.node_id AS type_node, t.descendant_count AS size,
           coalesce(i.start_line, t.start_line) AS start_line,
           coalesce(i.start_column, t.start_column) AS start_column,
           coalesce(i.end_line, t.end_line) AS end_line,
           coalesce(i.end_column, t.end_column) AS end_column,
           coalesce(g.node_id, t.node_id) AS body,
           CASE p.type WHEN 'assignment' THEN 'variable' WHEN 'type_alias_statement' THEN 'alias'
                       WHEN 'function_definition' THEN 'return' ELSE 'parameter' END AS role,
           CASE WHEN p.type IN ('assignment', 'type_alias_statement', 'function_definition')
                THEN p.node_id ELSE q.parent_id END AS owner,
           CASE x.type WHEN 'identifier' THEN x.name
                       WHEN 'attribute' THEN regexp_extract(x.name, '[^.]*$')
                       WHEN 'string' THEN coalesce(s.name, '')
                       WHEN 'none' THEN 'None'
                       WHEN 'generic_type' THEN regexp_extract(coalesce(h.name, ''), '[^.]*$')
                       WHEN 'subscript' THEN regexp_extract(coalesce(h.name, ''), '[^.]*$')
                       ELSE '' END AS head
    FROM core_node t
    JOIN core_file f USING (file_id)
    JOIN core_node p ON p.file_id = t.file_id AND p.node_id = t.parent_id
    LEFT JOIN core_node q ON q.file_id = p.file_id AND q.node_id = p.parent_id
                         AND p.type IN ('typed_parameter', 'typed_default_parameter')
    LEFT JOIN core_node g ON g.file_id = t.file_id AND g.parent_id = t.node_id
                         AND g.type = 'parenthesized_expression' AND t.children_count = 1
    LEFT JOIN core_node i ON i.file_id = g.file_id AND i.parent_id = g.node_id
                         AND i.type NOT IN ('(', ')', 'comment')
    LEFT JOIN core_node x ON x.file_id = t.file_id AND x.parent_id = coalesce(g.node_id, t.node_id)
                         AND x.type NOT IN ('(', ')', 'comment')
    LEFT JOIN core_node h ON h.file_id = x.file_id AND h.parent_id = x.node_id
                         AND h.sibling_index = 0 AND x.type IN ('generic_type', 'subscript')
                         AND h.type IN ('identifier', 'attribute')
    LEFT JOIN core_node s ON s.file_id = x.file_id AND s.parent_id = x.node_id
                         AND s.type = 'string_content' AND x.type = 'string'
    WHERE t.type = 'type' AND f.language = 'python'
      AND ((p.type IN ('assignment', 'typed_parameter', 'typed_default_parameter')
            AND t.sibling_index = 1)
           OR p.type = 'function_definition'
           OR (p.type = 'type_alias_statement' AND t.sibling_index = 3))
"""
