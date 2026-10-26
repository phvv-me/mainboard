"""SQL the import-side families share: macros for source slices and test paths, and CTE
fragments (Python module names, the `__all__` strings of a module, which files are Python) that
every relation of a family spells again, since a relation cannot read a later one. A Python file
counts unless it holds a syntax error that mcmr's parser would also reject: tree-sitter lags the
language, and PEP 696 type parameter defaults leave a lone error node under a `type_parameter`
in a valid file."""

from ....catalog import Function


def python_files(name: str) -> str:
    """The common table expression `name` over the Python files mcmr's parser accepts.

    Those are the files without an error node, except a lone one under a `type_parameter`.
    """
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


_NAME_READ_JOINS = """
    JOIN ast_nodes {parent} ON {parent}.fact_id = {identifier}.fact_id
      AND {parent}.node_id = {identifier}.parent_id
    LEFT JOIN ast_nodes {grandparent} ON {grandparent}.fact_id = {parent}.fact_id
      AND {grandparent}.node_id = {parent}.parent_id
      AND {parent}.type IN ('dotted_name', 'expression_list', 'list_splat_pattern',
                           'dictionary_splat_pattern')
"""

_NAME_READ_FILTER = """
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
    identifier: str = "i", *, parent: str = "p", grandparent: str = "q"
) -> tuple[str, str]:
    """The joins and filter keeping the identifiers a Python file uses as names.

    They join the identifier node `identifier` to its parent and, within an import statement, its
    grandparent, and drop identifiers used as attributes, keywords, parameters or import parts.
    """
    names = {"identifier": identifier, "parent": parent, "grandparent": grandparent}
    return _NAME_READ_JOINS.format(**names), _NAME_READ_FILTER.format(**names)
