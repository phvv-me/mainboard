"""`core_def` (functions, classes, interfaces and type aliases), `core_param` (a function's
parameters) and `core_decor` (the decorators of a definition)."""

DEF = r"""
    WITH defs AS (
      SELECT f.file_id, x.node_id AS def_id, f.path, f.language, f.parses,
             CASE x.semantic_type WHEN 'DEFINITION_FUNCTION' THEN 'function' ELSE 'class' END
               AS kind,
             x.type, x.name,
             list_transform(x.qualified_name, lambda q: q.name) AS names,
             coalesce(nullif(array_to_string(
               CASE WHEN f.language = 'python' THEN names
                    ELSE list_filter(names, lambda n, i: i = 1 OR n <> names[i - 1]) END, '.'),
               ''), '<anonymous>') AS qualname,
             coalesce(w.start_line, x.start_line) AS start_line,
             coalesce(w.start_column, x.start_column) AS start_column,
             x.end_line, x.end_column, x.descendant_count AS size,
             greatest(x.scope."function", x.scope."class") AS owner_id,
             x.scope."function" AS function_id, x.scope."class" AS class_id,
             w.node_id AS wrapper_id, list_contains(x.modifiers, 'async') AS is_async,
             x.signature_type AS returns
      FROM ast_nodes x
      JOIN core_file f ON f.fact_id = x.fact_id
      LEFT JOIN ast_nodes w ON w.fact_id = x.fact_id AND w.node_id = x.parent_id
                           AND w.type = 'decorated_definition'
      WHERE x.semantic_type IN ('DEFINITION_FUNCTION', 'DEFINITION_CLASS') AND x.flags % 2 = 0
        AND x.children_count > 0 AND f.language <> 'toml'
    ), bodies AS (
      SELECT d.file_id, d.def_id, b.node_id AS body_id,
             count(s.node_id) FILTER (WHERE s.type NOT IN ('comment', '{', '}')) AS statements,
             arg_min(s.node_id, s.sibling_index) FILTER (WHERE s.type NOT IN ('comment', '{', '}'))
               AS first_id
      FROM defs d
      JOIN core_node b ON b.file_id = d.file_id AND b.parent_id = d.def_id
                      AND b.type IN ('block', 'statement_block', 'compound_statement')
      LEFT JOIN core_node s ON s.file_id = b.file_id AND s.parent_id = b.node_id
      GROUP BY d.file_id, d.def_id, b.node_id
    )
    SELECT d.*, b.body_id, coalesce(b.statements, 0) AS statements, l.name AS docstring
    FROM defs d
    LEFT JOIN bodies b ON b.file_id = d.file_id AND b.def_id = d.def_id
    LEFT JOIN core_node s ON s.file_id = b.file_id AND s.node_id = b.first_id
                         AND s.type = 'expression_statement' AND s.children_count = 1
    LEFT JOIN core_node l ON l.file_id = s.file_id AND l.parent_id = s.node_id
                         AND l.sibling_index = 0 AND l.type = 'string'
                         AND regexp_matches(l.name, '^[uUrR]*["'']')
"""

PARAM = r"""
    SELECT d.file_id, d.def_id, p.node_id, p.type, p.name,
           row_number() OVER w - 1 AS ordinal,
           CASE p.type WHEN 'list_splat_pattern' THEN 'star'
                       WHEN 'dictionary_splat_pattern' THEN 'double_star'
                       WHEN 'keyword_separator' THEN 'separator'
                       WHEN 'positional_separator' THEN 'separator'
                       WHEN '*' THEN 'separator' WHEN '/' THEN 'separator'
                       WHEN 'default_parameter' THEN 'default'
                       WHEN 'typed_default_parameter' THEN 'default' ELSE 'plain' END AS kind,
           regexp_replace(t.name, '\s', '', 'g') AS annotation, v.type AS default_type,
           coalesce(bool_or(p.type IN ('list_splat_pattern', 'keyword_separator', '*')) OVER (
             PARTITION BY d.file_id, d.def_id ORDER BY p.sibling_index
             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), false) AS keyword_only,
           p.start_line, p.start_column, p.end_line, p.end_column
    FROM core_def d
    JOIN core_node l ON l.file_id = d.file_id AND l.parent_id = d.def_id
                    AND l.type IN ('parameters', 'lambda_parameters', 'formal_parameters')
    JOIN core_node p ON p.file_id = l.file_id AND p.parent_id = l.node_id
                    AND p.semantic_type NOT IN ('PARSER_DELIMITER', 'METADATA_COMMENT')
    LEFT JOIN core_node t ON t.file_id = p.file_id AND t.parent_id = p.node_id
                         AND t.type = 'type'
    LEFT JOIN core_node v ON v.file_id = p.file_id AND v.parent_id = p.node_id
                         AND v.sibling_index = p.children_count::INTEGER - 1
                         AND p.type IN ('default_parameter', 'typed_default_parameter')
    WHERE d.kind = 'function'
    WINDOW w AS (PARTITION BY d.file_id, d.def_id ORDER BY p.sibling_index)
"""

DECOR = """
    SELECT w.file_id, t.node_id AS def_id, d.sibling_index AS ordinal, e.type AS expression_type,
           CASE WHEN e.type = 'call' THEN k.name ELSE e.name END AS name
    FROM core_node w
    JOIN core_node d ON d.file_id = w.file_id AND d.parent_id = w.node_id AND d.type = 'decorator'
    JOIN core_node t ON t.file_id = w.file_id AND t.parent_id = w.node_id
                    AND t.type IN ('function_definition', 'class_definition')
    JOIN core_node e ON e.file_id = d.file_id AND e.parent_id = d.node_id
                    AND e.flags % 2 = 0
    LEFT JOIN core_node k ON k.file_id = e.file_id AND k.parent_id = e.node_id
                         AND k.sibling_index = 0 AND e.type = 'call'
    WHERE w.type = 'decorated_definition'
"""
