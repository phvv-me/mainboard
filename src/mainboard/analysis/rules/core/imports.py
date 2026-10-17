"""`core_import` (one row per name an import statement binds, Python)."""

IMPORT = r"""
    WITH statements AS MATERIALIZED (
      SELECT n.file_id, n.node_id AS statement_id, n.type AS statement_type,
             n.name AS statement_name, n.start_line, n.start_column, n.end_line, n.end_column,
             f.path, f.module AS importer_module, f.is_package
      FROM core_node n JOIN core_file f USING (file_id)
      WHERE n.type IN ('import_statement', 'import_from_statement', 'future_import_statement')
        AND f.language = 'python' AND f.parses
    ), parts AS MATERIALIZED (
      SELECT s.file_id, s.statement_id, c.node_id AS part, c.type AS part_type,
             c.name AS part_name, c.sibling_index AS position, c.start_line, c.start_column,
             c.end_line, c.end_column
      FROM statements s
      JOIN core_node c ON c.file_id = s.file_id AND c.parent_id = s.statement_id
      WHERE c.type IN ('dotted_name', 'aliased_import', 'wildcard_import', 'relative_import',
                       'import', '__future__')
    ), keywords AS (
      SELECT file_id, statement_id, min(position) AS keyword FROM parts
      WHERE part_type = 'import' GROUP BY ALL
    ), from_modules AS (
      SELECT p.file_id, p.statement_id,
             coalesce(d.node_id, CASE WHEN p.part_type <> 'relative_import' THEN p.part END)
               AS node
      FROM parts p
      JOIN keywords k ON k.file_id = p.file_id AND k.statement_id = p.statement_id
      LEFT JOIN core_node d ON d.file_id = p.file_id AND d.parent_id = p.part
        AND p.part_type = 'relative_import' AND d.type = 'dotted_name'
      WHERE p.part_type IN ('dotted_name', 'relative_import', '__future__')
        AND p.position < k.keyword
    ), bound AS (
      SELECT s.*, p.part AS binding_id, p.position,
             p.start_line AS binding_start_line, p.start_column AS binding_start_column,
             p.end_line AS binding_end_line, p.end_column AS binding_end_column,
             CASE p.part_type WHEN 'aliased_import' THEN a.name
                              WHEN 'wildcard_import' THEN '*' ELSE p.part_name END
               AS imported_name,
             CASE WHEN p.part_type = 'aliased_import' THEN p.part_name
                  WHEN p.part_type = 'wildcard_import' THEN '*'
                  WHEN s.statement_type = 'import_statement' THEN split_part(p.part_name, '.', 1)
                  ELSE p.part_name END AS name,
             p.part_type = 'aliased_import' AS aliased,
             CASE WHEN s.statement_type = 'import_statement' THEN coalesce(a.node_id, p.part)
                  ELSE m.node END AS module_id,
             count(*) OVER (PARTITION BY s.file_id, s.statement_id) AS bindings
      FROM statements s
      JOIN keywords k ON k.file_id = s.file_id AND k.statement_id = s.statement_id
      JOIN parts p ON p.file_id = s.file_id AND p.statement_id = s.statement_id
        AND p.part_type IN ('dotted_name', 'aliased_import', 'wildcard_import')
        AND p.position > k.keyword
      LEFT JOIN core_node a ON a.file_id = p.file_id AND a.parent_id = p.part
        AND p.part_type = 'aliased_import' AND a.type = 'dotted_name'
      LEFT JOIN from_modules m ON m.file_id = s.file_id AND m.statement_id = s.statement_id
    ), leveled AS (
      SELECT *,
             CASE statement_type WHEN 'import_statement' THEN imported_name
                                 WHEN 'future_import_statement' THEN '__future__'
                                 ELSE ltrim(statement_name, '.') END AS module,
             CASE WHEN statement_type = 'import_from_statement'
                  THEN length(statement_name) - length(ltrim(statement_name, '.')) ELSE 0
             END AS relative_level,
             CASE WHEN is_package THEN string_split(importer_module, '.')
                  ELSE list_slice(string_split(importer_module, '.'), 1, -2) END AS parents
      FROM bound
    ), resolved AS (
    SELECT file_id, path, statement_id, binding_id, position, statement_type, start_line,
           start_column, end_line, end_column, binding_start_line, binding_start_column,
           binding_end_line, binding_end_column, module_id, imported_name, name, aliased,
           module, relative_level, importer_module, bindings,
           relative_level > 0 OR split_part(module, '.', 1) = split_part(importer_module, '.', 1)
             AS is_project_owned,
           list_bool_or(list_transform(string_split(module, '.'),
             lambda part: starts_with(part, '_') AND NOT starts_with(part, '__')))
             AS has_private_module_component,
           CASE WHEN statement_type = 'import_statement' THEN NULL
                WHEN relative_level = 0 THEN module
                WHEN relative_level > len(parents) THEN repeat('.', relative_level) || module
                WHEN len(parents) + 1 - relative_level = 0 THEN module
                WHEN module = '' THEN array_to_string(
                       list_slice(parents, 1, len(parents) + 1 - relative_level), '.')
                ELSE array_to_string(
                       list_slice(parents, 1, len(parents) + 1 - relative_level), '.')
                     || '.' || module END AS source_module
    FROM leveled
    )
    SELECT *, CASE WHEN statement_type = 'import_statement'
                   THEN CASE WHEN aliased THEN imported_name
                             ELSE split_part(imported_name, '.', 1) END
                   ELSE source_module || '.' || imported_name END AS target
    FROM resolved
"""
