"""SQL the import-side families share: macros for source slices and test paths, and CTE
fragments (Python module names, the `__all__` strings of a module, which files are Python) that
every relation of a family spells again, since a relation cannot read a later one. A Python file
counts unless it holds a syntax error that mcmr's parser would also reject: tree-sitter lags the
language, and PEP 696 type parameter defaults leave a lone error node under a `type_parameter`
in a valid file."""

from ...catalog import Function

PARSED = """
  parsed AS (
    SELECT fact_id, path FROM ast_files
    WHERE language = 'python'
      AND fact_id NOT IN (SELECT fact_id FROM ast_nodes WHERE type = 'ERROR')
  )
"""


def python_files(name: str) -> str:
    """The common table expression `name` over the Python files mcmr's parser also accepts:
    those without an error node, except a lone one under a `type_parameter`."""
    return f"""
  {name} AS (
    SELECT fact_id, path FROM ast_files
    WHERE language = 'python'
      AND fact_id NOT IN (
        SELECT e.fact_id FROM ast_nodes e LEFT JOIN ast_nodes p
          ON p.fact_id = e.fact_id AND p.node_id = e.parent_id
        WHERE e.type = 'ERROR' AND coalesce(p.type, '') <> 'type_parameter')
  )
"""


PYTHON = python_files("python_files")


LINES = "lines AS (SELECT path, string_split(content, chr(10)) AS lines FROM source_files)"


class ImportSourceSlice(Function):
    """The source text between two positions of one file: `lines` is the file's list of lines
    and the columns count UTF-8 bytes."""

    lines: list[str]
    first_line: int
    first_column: int
    last_line: int
    last_column: int

    expression = """
        CASE WHEN first_line = last_line
             THEN decode(array_slice(encode(lines[first_line]), first_column + 1, last_column))
             ELSE array_to_string(list_concat(
                    [decode(array_slice(encode(lines[first_line]), first_column + 1,
                                        strlen(lines[first_line])))],
                    lines[first_line + 1:last_line - 1],
                    [decode(array_slice(encode(lines[last_line]), 0, last_column))]), chr(10))
             END
    """


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


class ImportIsTestPath(Function):
    """Whether a path follows mcmr's test-file convention: a test_ prefix, a conventional file
    name, a marker in the file name or a test directory."""

    path: str

    expression = r"""
        starts_with(regexp_extract(path, '[^/]*$'), 'test_')
        OR regexp_extract(path, '[^/]*$') IN ('test.py', 'tests.py', 'tests.rs', 'conftest.py')
        OR contains(regexp_extract(path, '[^/]*$'), '_test.')
        OR contains(regexp_extract(path, '[^/]*$'), '.test.')
        OR contains(regexp_extract(path, '[^/]*$'), '.spec.')
        OR list_has_any(string_split(path, '/')[:-2], ['test', 'tests', '__tests__'])
    """


def is_test_path(path: str) -> str:
    """The SQL expression for mcmr's test-file convention over the path expression `path`."""
    return f"import_is_test_path({path})"


NAME_READ_JOINS = """
    JOIN ast_nodes {parent} ON {parent}.fact_id = {identifier}.fact_id
      AND {parent}.node_id = {identifier}.parent_id
    LEFT JOIN ast_nodes {grandparent} ON {grandparent}.fact_id = {parent}.fact_id
      AND {grandparent}.node_id = {parent}.parent_id
      AND {parent}.type IN ('dotted_name', 'expression_list', 'list_splat_pattern',
                           'dictionary_splat_pattern')
"""

NAME_READ_FILTER = """
    {identifier}.type = 'identifier'
    AND {parent}.type NOT IN ('aliased_import', 'global_statement', 'nonlocal_statement',
                              'function_definition', 'class_definition', 'parameters',
                              'lambda_parameters')
    AND NOT ({parent}.type = 'dotted_name' AND {grandparent}.type IN ('import_statement',
         'import_from_statement', 'future_import_statement', 'aliased_import',
         'relative_import'))
    AND NOT ({parent}.type = 'dotted_name' AND {identifier}.sibling_index >= 1)
    AND NOT ({parent}.type IN ('list_splat_pattern', 'dictionary_splat_pattern')
             AND {grandparent}.type IN ('parameters', 'typed_parameter', 'lambda_parameters'))
    AND NOT ({parent}.type = 'attribute' AND {identifier}.sibling_index >= 1)
    AND NOT ({parent}.type = 'keyword_argument' AND {identifier}.sibling_index = 0)
    AND NOT ({parent}.type IN ('default_parameter', 'typed_parameter', 'typed_default_parameter')
             AND {identifier}.sibling_index = 0)
"""


def name_reads(
    identifier: str = "i", parent: str = "p", grandparent: str = "q"
) -> tuple[str, str]:
    """The joins (from the identifier node `identifier` to its parent and, within an import
    statement, grandparent) and the filter that keep the identifiers a Python file reads or
    stores as names, not as attributes, keywords, parameters or import parts."""
    names = {"identifier": identifier, "parent": parent, "grandparent": grandparent}
    return NAME_READ_JOINS.format(**names), NAME_READ_FILTER.format(**names)
