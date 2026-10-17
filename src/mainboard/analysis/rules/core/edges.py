"""`core_edge` (module and class edges, Python)."""

EDGE = r"""
    WITH modules AS (
      SELECT module, coalesce(min(file_id) FILTER (WHERE NOT is_stub), min(file_id)) AS file_id
      FROM core_file WHERE language = 'python' AND module IS NOT NULL GROUP BY module
    ), typed AS (
      SELECT DISTINCT i.file_id, i.statement_id
      FROM core_import i
      JOIN core_node f ON f.file_id = i.file_id AND f.type = 'if_statement'
                      AND i.statement_id > f.node_id
                      AND i.statement_id <= f.node_id + f.descendant_count
      JOIN core_node t ON t.file_id = f.file_id AND t.parent_id = f.node_id
                      AND t.sibling_index = 1 AND regexp_matches(t.name, '(^|\.)TYPE_CHECKING$')
    ), reached AS (
      SELECT i.file_id, i.statement_id, i.position, i.start_line, i.start_column,
             i.importer_module, max_by(p.module, length(p.module)) AS target,
             max_by(p.file_id, length(p.module)) AS target_file_id
      FROM core_import i
      CROSS JOIN LATERAL (
        SELECT CASE WHEN i.statement_type = 'import_statement' THEN i.imported_name
                    ELSE i.target END AS reached) r
      JOIN modules p ON r.reached = p.module OR starts_with(r.reached, p.module || '.')
      GROUP BY i.file_id, i.statement_id, i.position, i.start_line, i.start_column,
               i.importer_module
    )
    SELECT r.file_id, 'module' AS kind, r.importer_module AS source, r.target,
           r.position AS ordinal, r.target_file_id, t.statement_id IS NOT NULL AS type_checking,
           r.start_line, r.start_column
    FROM reached r LEFT JOIN typed t ON t.file_id = r.file_id AND t.statement_id = r.statement_id
    UNION ALL
    SELECT d.file_id, 'class', d.qualname, b.name, b.sibling_index, NULL, false, d.start_line,
           d.start_column
    FROM core_def d
    JOIN core_node l ON l.file_id = d.file_id AND l.parent_id = d.def_id
                    AND l.type = 'argument_list'
    JOIN core_node b ON b.file_id = l.file_id AND b.parent_id = l.node_id
                    AND b.type IN ('identifier', 'attribute', 'subscript', 'call')
    WHERE d.kind = 'class' AND d.language = 'python'
"""
