"""Rules over how much one callable holds (mcmr's ALL-FUNC size rules)."""

from ...catalog import Example, Rule


class FunctionStatementCount(Rule, id="ALL-FUNC0001"):
    """Direct statements in one callable body, declarative bodies excepted. A test function is
    PY-TEST0012's, which counts a test's statements as a test reads."""

    maximum: int = 10

    query = """
        WITH owned AS (
          SELECT d.*, d.statements - (d.docstring IS NOT NULL)::INTEGER AS direct_statement_count,
                 EXISTS (SELECT 1 FROM core_decor x
                         WHERE x.file_id = d.file_id AND x.def_id = d.def_id
                           AND regexp_extract(trim(split_part(x.name, '(', 1)), '[^.]*$') = 'rule')
                 OR (NOT EXISTS (SELECT 1 FROM flow_controls() c
                                 WHERE c.file_id = d.file_id AND c.def_id = d.def_id)
                     AND EXISTS (SELECT 1 FROM core_node q
                                 JOIN core_node r ON r.file_id = q.file_id
                                   AND r.node_id BETWEEN q.node_id
                                     AND q.node_id + q.descendant_count
                                   AND r.type = 'identifier'
                                 WHERE q.file_id = d.file_id AND q.parent_id = d.def_id
                                   AND q.type = 'type'
                                   AND (r.name IN ('Expr', 'LazyFrame')
                                        OR ends_with(r.name, 'Query')))) AS is_declarative_body
          FROM core_def d WHERE d.kind = 'function' AND d.type <> 'lambda'
        )
        SELECT path, start_line, start_column,
               '`' || coalesce(nullif(name, ''), '<anonymous>') || '` owns '
                 || direct_statement_count || ' direct statement'
                 || CASE WHEN direct_statement_count = 1 THEN '' ELSE 's' END AS message
        FROM owned
        WHERE NOT is_declarative_body AND direct_statement_count > $maximum
          AND NOT (starts_with(name, 'test_') AND import_is_test_path(path))
    """


class ShallowCallable(Rule, id="ALL-FUNC0006", applicability="review"):
    """A one-line public module function with no behavior and no reuse."""

    minimum_references: int = 2
    minimum_operations: int = 1
    ignore_names: tuple[str, ...] = ()

    query = """
        SELECT * FROM function_functions
        WHERE implementation_lines = 1 AND behavior_operation_count < $minimum_operations
          AND reference_count < $minimum_references
          AND NOT (language <> 'python' OR visibility <> 'public' OR scope <> 'module'
                   OR docstring <> '' OR ends_with(path, '.pyi') OR starts_with(name, 'test_')
                   OR list_contains($ignore_names::VARCHAR[], name)
                   OR is_protocol_member OR is_abstract OR is_property OR is_overload
                   OR is_protocol_name OR is_framework_hook OR is_polymorphic
                   OR is_pass_body OR is_raise_body)
    """
    message = "`{fact_id}` satisfies shallow callable"
    examples = (
        Example(
            bad=(
                "def forward(value):\n"
                "    return value\n"
                "\n"
                "\n"
                "def total(items):\n"
                "    count = 0\n"
                "    for item in items:\n"
                "        count += item\n"
                "    return count\n"
            ),
            good=(
                "def total(items):\n"
                "    count = 0\n"
                "    for item in items:\n"
                "        count += item\n"
                "    return count\n"
            ),
        ),
    )
    # One call in the same file takes the body's expression with its arguments substituted
    # (`functions_inlined()`); one with no call at all is dropped. Either way the definition
    # goes with the blank lines after it, unless a string literal of the repository names the
    # function (a dispatch key, an entry point) or it is an entry point by convention.
    fix = """
        SELECT f.finding_key, i.path, i.ref_line AS start_line, i.ref_column AS start_column,
               i.ref_end_line AS end_line, i.ref_end_column AS end_column, i.inlined AS text
        FROM finding f JOIN functions_inlined() i ON i.entity_id = f.entity_id
        WHERE NOT starts_with(f.name, '__') AND f.name <> 'main'
          AND NOT starts_with(f.name, 'pytest_')
          AND NOT EXISTS (SELECT 1 FROM reach_quoted_words() w WHERE w.word = f.name)
        UNION ALL BY NAME
        SELECT f.finding_key, f.definition_path AS path,
               functions_delete_start(string_split(c.content, chr(10)), f.definition_start_line,
                                      f.definition_end_line) AS start_line,
               0 AS start_column,
               functions_delete_end(string_split(c.content, chr(10)), f.definition_end_line)
                 AS end_line,
               0 AS end_column, '' AS text
        FROM finding f
        LEFT JOIN functions_inlined() i ON i.entity_id = f.entity_id
        JOIN source_files c ON c.path = f.definition_path
        WHERE (i.entity_id IS NOT NULL OR f.reference_count = 0)
          AND NOT starts_with(f.name, '__') AND f.name <> 'main'
          AND NOT starts_with(f.name, 'pytest_')
          AND NOT EXISTS (SELECT 1 FROM reach_quoted_words() w WHERE w.word = f.name)
    """
