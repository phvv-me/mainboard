"""`core_ref` (every identifier of a Python file and whether it reads a variable)."""

REF = r"""
    SELECT i.file_id, f.path, i.node_id AS ref_id, i.name, i.start_line, i.start_column,
           i.end_line, i.end_column, i.sibling_index AS place, i.owner_id, i.function_id,
           i.class_id, p.type AS parent_type, p.node_id AS parent, g.type AS grand_type,
           syntax_is_load(p.type, i.sibling_index, p.sibling_index, g.type) AS is_load
    FROM core_node i
    JOIN core_file f USING (file_id)
    JOIN core_node p ON p.file_id = i.file_id AND p.node_id = i.parent_id
    LEFT JOIN core_node g ON g.file_id = p.file_id AND g.node_id = p.parent_id
    WHERE i.type = 'identifier' AND f.language = 'python'
"""
