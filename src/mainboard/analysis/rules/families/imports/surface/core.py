"""mcmr's ModuleSurfaceFact for TypeScript, derived from sitting_duck's syntax trees: what a module
exports, how far its relative imports climb, the constructs type stripping cannot erase and the
places it steps around its own types.
"""

from .....catalog import Family
from ...constants.imports_common import LINES

_FILES = (
    LINES
    + """,
  files AS (SELECT fact_id, path FROM ast_files WHERE language = 'typescript'),
  top AS (
    SELECT t.fact_id, t.node_id, t.type, t.start_line, t.depth, t.descendant_count
    FROM ast_nodes t JOIN files f ON f.fact_id = t.fact_id WHERE t.parent_id = 0
  ),
  statements AS (
    SELECT t.fact_id, t.node_id, t.type,
           min_by(trim(s.name, '"'''), s.sibling_index) FILTER (WHERE s.type = 'string') AS source,
           bool_or(s.type IN ('*', 'namespace_export')) AS is_star,
           bool_or(s.type = 'import_require_clause') AS requires,
           bool_or(s.type = '=') AS assigns
    FROM top t LEFT JOIN ast_nodes s ON s.fact_id = t.fact_id AND s.parent_id = t.node_id
    WHERE t.type IN ('export_statement', 'import_statement')
    GROUP BY t.fact_id, t.node_id, t.type
  ),
  specifiers AS (
    SELECT s.fact_id, s.node_id, count(e.node_id) AS specifiers
    FROM statements s
    JOIN ast_nodes c ON c.fact_id = s.fact_id AND c.parent_id = s.node_id
      AND c.type = 'export_clause'
    JOIN ast_nodes e ON e.fact_id = c.fact_id AND e.parent_id = c.node_id
      AND e.type = 'export_specifier'
    GROUP BY s.fact_id, s.node_id
  ),
  climbs AS (
    SELECT fact_id, node_id, source AS specifier, len(string_split(source, '../')) - 1 AS depth
    FROM statements WHERE source IS NOT NULL AND NOT requires
  ),
  deepest AS (
    SELECT fact_id, max(depth) AS depth,
           arg_max(specifier, depth * 1000000 - node_id) AS specifier
    FROM climbs GROUP BY fact_id
  ),
  surface AS (
    SELECT s.fact_id,
           count(*) FILTER (WHERE s.type = 'export_statement' AND NOT s.is_star AND NOT s.assigns)
             AS exports,
           coalesce(sum(p.specifiers) FILTER (WHERE s.source IS NOT NULL AND NOT s.is_star), 0)
             AS named_reexports
    FROM statements s LEFT JOIN specifiers p ON p.fact_id = s.fact_id AND p.node_id = s.node_id
    GROUP BY s.fact_id
  ),
  declarations AS (
    SELECT t.fact_id, t.node_id AS statement, t.start_line AS line, d.node_id, d.type,
           d.name, d.children_count
    FROM top t
    JOIN ast_nodes d ON d.fact_id = t.fact_id
      AND d.node_id BETWEEN t.node_id AND t.node_id + t.descendant_count
      AND d.depth - t.depth <= 2
    WHERE d.type IN ('enum_declaration', 'internal_module', 'module', 'import_require_clause')
      AND d.children_count > 0
  ),
  erasable AS (
    SELECT d.fact_id, d.line, 0 AS source_order, d.node_id,
           CASE d.type WHEN 'enum_declaration' THEN
                  CASE WHEN EXISTS (SELECT 1 FROM ast_nodes k WHERE k.fact_id = d.fact_id
                                      AND k.parent_id = d.node_id AND k.type = 'const')
                       THEN 'const_enum' ELSE 'enum' END
                WHEN 'import_require_clause' THEN 'import_equals' ELSE 'namespace' END AS kind,
           CASE d.type WHEN 'module' THEN coalesce(
                  (SELECT min_by(trim(c.name, '"'''), c.sibling_index) FROM ast_nodes c
                   WHERE c.fact_id = d.fact_id AND c.parent_id = d.node_id
                     AND c.type IN ('identifier', 'string')), '')
                ELSE d.name END AS name
    FROM declarations d
  ),
  constructors AS (
    SELECT m.fact_id, m.node_id
    FROM top t
    JOIN ast_nodes c ON c.fact_id = t.fact_id
      AND c.node_id BETWEEN t.node_id AND t.node_id + t.descendant_count
      AND c.type IN ('class_declaration', 'abstract_class_declaration') AND c.depth - t.depth <= 1
    JOIN ast_nodes b ON b.fact_id = c.fact_id AND b.parent_id = c.node_id AND b.type = 'class_body'
    JOIN ast_nodes m ON m.fact_id = b.fact_id AND m.parent_id = b.node_id
      AND m.type = 'method_definition' AND m.name = 'constructor'
  ),
  parameter_properties AS (
    SELECT p.fact_id, p.start_line AS line, 1 AS source_order, p.node_id,
           'parameter_property' AS kind, i.name
    FROM constructors c
    JOIN ast_nodes f ON f.fact_id = c.fact_id AND f.parent_id = c.node_id
      AND f.type = 'formal_parameters'
    JOIN ast_nodes p ON p.fact_id = f.fact_id AND p.parent_id = f.node_id
      AND p.type IN ('required_parameter', 'optional_parameter')
    JOIN ast_nodes i ON i.fact_id = p.fact_id AND i.parent_id = p.node_id AND i.type = 'identifier'
    WHERE EXISTS (SELECT 1 FROM ast_nodes m WHERE m.fact_id = p.fact_id AND m.parent_id = p.node_id
                    AND m.type IN ('accessibility_modifier', 'readonly'))
    QUALIFY row_number() OVER (PARTITION BY p.fact_id, p.node_id ORDER BY i.sibling_index) = 1
  ),
  hatches AS (
    SELECT n.fact_id, n.start_line AS line, n.start_column, n.node_id,
           CASE n.type WHEN 'predefined_type' THEN 'any' WHEN 'as_expression' THEN 'assertion'
                       WHEN 'type_assertion' THEN 'assertion'
                       WHEN 'non_null_expression' THEN 'non_null'
                       ELSE 'ignore_comment' END AS kind
    FROM ast_nodes n JOIN files f ON f.fact_id = n.fact_id
    WHERE (n.type = 'predefined_type' AND n.name = 'any')
       OR (n.type = 'as_expression' AND NOT EXISTS (
             SELECT 1 FROM ast_nodes k WHERE k.fact_id = n.fact_id AND k.parent_id = n.node_id
               AND k.sibling_index = 2 AND k.type = 'const'))
       OR n.type IN ('type_assertion', 'non_null_expression')
    UNION ALL
    SELECT c.fact_id, c.start_line, c.start_column, c.node_id, 'ignore_comment'
    FROM ast_nodes c
    JOIN files f ON f.fact_id = c.fact_id
    JOIN lines l ON l.path = f.path
    WHERE c.type = 'comment'
      AND regexp_matches(
            import_source_slice(l.lines, c.start_line, c.start_column, c.end_line, c.end_column),
            '@ts-ignore|@ts-expect-error')
  )
"""
)


class ModuleSurface(Family):
    family = "ModuleSurfaceFact"
    verified = False
    relations = {
        "facts": f"""
            WITH {_FILES}, erasures AS (
              SELECT fact_id, count(*) AS n
              FROM (SELECT fact_id FROM erasable
                    UNION ALL SELECT fact_id FROM parameter_properties)
              GROUP BY fact_id
            ), escapes AS (SELECT fact_id, count(*) AS n FROM hatches GROUP BY fact_id)
            SELECT (row_number() OVER (ORDER BY f.path) - 1)::UBIGINT AS fact_order,
                   'surface:' || f.path AS fact_id, f.path, 1::UBIGINT AS start_line,
                   0::UBIGINT AS start_column, 1::UBIGINT AS end_line, 0::UBIGINT AS end_column,
                   'typescript' AS language,
                   coalesce(d.depth, 0)::BIGINT AS deepest_relative_import,
                   CASE WHEN d.depth > 0 THEN d.specifier ELSE '' END
                     AS deepest_relative_specifier,
                   coalesce(e.n, 0)::BIGINT AS erasable_violations_length,
                   true AS erasable_violations_present,
                   coalesce(h.n, 0)::BIGINT AS escape_hatches_length,
                   true AS escape_hatches_present, 0::BIGINT AS evidence_length,
                   true AS evidence_present, coalesce(s.exports, 0)::BIGINT AS export_count,
                   (f.path = 'index.ts' OR ends_with(f.path, '/index.ts')) AS is_index_module,
                   coalesce(s.named_reexports, 0)::BIGINT AS named_reexport_count,
                   (length(l.content) - length(replace(l.content, chr(10), ''))
                    + CASE WHEN l.content <> '' AND NOT ends_with(l.content, chr(10)) THEN 1
                           ELSE 0 END)::BIGINT AS physical_line_count,
                   (SELECT count(*) FROM statements x WHERE x.fact_id = f.fact_id AND x.is_star)
                     ::BIGINT AS star_reexports_length,
                   true AS star_reexports_present
            FROM files f
            JOIN source_files l ON l.path = f.path
            LEFT JOIN deepest d ON d.fact_id = f.fact_id
            LEFT JOIN surface s ON s.fact_id = f.fact_id
            LEFT JOIN erasures e ON e.fact_id = f.fact_id
            LEFT JOIN escapes h ON h.fact_id = f.fact_id
        """,
        "erasable_violations": f"""
            WITH {_FILES}
            SELECT o.fact_order, o.fact_id,
                   (row_number() OVER (PARTITION BY v.fact_id ORDER BY v.source_order, v.node_id)
                    - 1)::UBIGINT AS ordinal, v.kind, v.line::BIGINT AS line, v.name
            FROM (SELECT fact_id, line, source_order, node_id, kind, name FROM erasable
                  UNION ALL
                  SELECT fact_id, line, source_order, node_id, kind, name
                  FROM parameter_properties) v
            JOIN files f ON f.fact_id = v.fact_id
            JOIN module_surface_facts o ON o.path = f.path
        """,
        "escape_hatches": f"""
            WITH {_FILES}
            SELECT o.fact_order, o.fact_id,
                   (row_number() OVER (PARTITION BY v.fact_id
                                       ORDER BY v.line, v.start_column, v.node_id) - 1)::UBIGINT
                     AS ordinal, v.kind, v.line::BIGINT AS line
            FROM hatches v
            JOIN files f ON f.fact_id = v.fact_id
            JOIN module_surface_facts o ON o.path = f.path
        """,
        "values": f"""
            WITH {_FILES}
            SELECT o.fact_order, o.fact_id, 'star_reexports' AS relation,
                   (row_number() OVER (PARTITION BY s.fact_id ORDER BY s.node_id) - 1)::UBIGINT
                     AS ordinal, s.source AS string_value
            FROM statements s
            JOIN files f ON f.fact_id = s.fact_id
            JOIN module_surface_facts o ON o.path = f.path
            WHERE s.is_star
        """,
    }
