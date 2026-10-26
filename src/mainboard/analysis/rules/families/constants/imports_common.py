"""The constants of `imports_common` other modules share."""

PARSED = """
  parsed AS (
    SELECT fact_id, path FROM ast_files
    WHERE language = 'python'
      AND fact_id NOT IN (SELECT fact_id FROM ast_nodes WHERE type = 'ERROR')
  )
"""

LINES = "lines AS (SELECT path, string_split(content, chr(10)) AS lines FROM source_files)"

MODULE_NAMES = r"""
  regular_packages AS (
    SELECT DISTINCT regexp_replace(path, '(^|/)[^/]*$', '') AS directory
    FROM ast_files WHERE language = 'python' AND regexp_matches(path, '(^|/)__init__\.py$')
      AND contains(path, '/')
  ),
  top_names AS (
    SELECT DISTINCT regexp_extract(directory, '[^/]*$') AS name FROM regular_packages p
    WHERE NOT EXISTS (SELECT 1 FROM regular_packages q
                      WHERE q.directory <> p.directory
                        AND starts_with(p.directory, q.directory || '/'))
  ),
  document_directories AS (
    SELECT DISTINCT unnest(list_transform(range(1, len(parts)),
                                          lambda k: array_to_string(parts[1:k], '/'))) AS directory
    FROM (SELECT string_split(path, '/') AS parts FROM ast_files)
  ),
  namespace_packages AS (
    SELECT d.directory FROM document_directories d
    WHERE regexp_extract(d.directory, '[^/]*$') IN (SELECT name FROM top_names)
      AND NOT EXISTS (SELECT 1 FROM regular_packages p
                      WHERE (p.directory = d.directory
                             OR starts_with(p.directory, d.directory || '/'))
                        AND regexp_extract(p.directory, '[^/]*$')
                          = regexp_extract(d.directory, '[^/]*$'))
  ),
  package_directories AS (
    SELECT directory FROM regular_packages UNION SELECT directory FROM namespace_packages
  ),
  outer_packages AS (
    SELECT p.directory FROM package_directories p
    WHERE NOT EXISTS (SELECT 1 FROM package_directories q
                      WHERE q.directory <> p.directory
                        AND starts_with(p.directory, q.directory || '/'))
  ),
  import_roots AS (
    SELECT DISTINCT regexp_replace(directory, '(^|/)[^/]*$', '') AS root
    FROM outer_packages WHERE contains(directory, '/')
  ),
  module_names AS (
    SELECT f.fact_id, f.path,
           replace(
             regexp_replace(
               regexp_replace(
                 CASE WHEN r.root IS NULL THEN f.path ELSE substr(f.path, length(r.root) + 2) END,
                 '\.pyi?$', ''),
               '/__init__$', ''),
             '/', '.') AS module
    FROM ast_files f
    LEFT JOIN LATERAL (
      SELECT max_by(root, length(root)) AS root FROM import_roots
      WHERE starts_with(f.path, root || '/')) r ON true
    WHERE f.language = 'python'
  )
"""

ALL_STRINGS = r"""
  all_statements AS (
    SELECT a.fact_id, a.path, a.node_id, a.type, a.start_line, a.start_column, a.end_line,
           a.end_column, a.descendant_count
    FROM ast_nodes a
    JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
      AND l.sibling_index = 0 AND l.type = 'identifier' AND l.name = '__all__'
    WHERE a.type IN ('assignment', 'augmented_assignment')
      AND a.scope.function IS NULL AND a.scope.class IS NULL
  ),
  all_strings AS (
    SELECT a.fact_id, a.path, a.node_id AS statement, s.node_id, s.start_line, s.start_column,
           s.end_line, s.end_column,
           trim(regexp_replace(s.name, '^[A-Za-z]{0,2}', ''), '"''') AS value
    FROM all_statements a
    JOIN ast_nodes s ON s.fact_id = a.fact_id AND s.type = 'string'
      AND s.node_id BETWEEN a.node_id + 1 AND a.node_id + a.descendant_count
  )
"""
