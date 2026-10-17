"""mcmr's ModuleFact, derived from sitting_duck's syntax trees: a module's size, its top-level
declarations, its `__all__` statements and where its public constants sit (Python), and the size
and top-level classes and functions of TypeScript, C, C++ and CUDA modules.

A Python module's top-level statements are the children of the tree's root, comments aside; a
decorated definition is one statement holding its class or function.
"""

from ...catalog import Family
from .imports_common import LINES, PYTHON, is_test_path, name_reads

_TOP = (
    PYTHON
    + r"""
  , top AS MATERIALIZED (
    SELECT t.fact_id, t.path, t.node_id, t.type, t.name, t.children_count, t.start_line,
           t.start_column, t.end_line, t.end_column,
           row_number() OVER (PARTITION BY t.fact_id ORDER BY t.node_id) - 1 AS idx
    FROM ast_nodes t JOIN python_files f ON f.fact_id = t.fact_id
    WHERE t.parent_id = 0 AND t.type <> 'comment'
  ),
  shape AS (
    SELECT t.*, d.type AS definition_type, d.name AS definition_name,
           e.type AS expression_type, e.node_id AS expression, e.name AS expression_name
    FROM top t
    LEFT JOIN ast_nodes d ON d.fact_id = t.fact_id AND d.parent_id = t.node_id
      AND t.type = 'decorated_definition' AND d.type IN ('function_definition', 'class_definition')
    LEFT JOIN ast_nodes e ON e.fact_id = t.fact_id AND e.parent_id = t.node_id
      AND t.type = 'expression_statement' AND e.sibling_index = 0
  ),
  targets AS (
    SELECT s.fact_id, s.node_id, l.name AS target, l.type AS target_type,
           a.type AS assignment_type, eq.node_id IS NOT NULL AS has_value,
           ty.node_id IS NOT NULL AS annotated,
           v.node_id AS value, v.type AS value_type, v.descendant_count AS value_size
    FROM shape s
    JOIN ast_nodes a ON a.fact_id = s.fact_id AND a.node_id = s.expression
      AND a.type IN ('assignment', 'augmented_assignment')
    JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id AND l.sibling_index = 0
    LEFT JOIN ast_nodes eq ON eq.fact_id = a.fact_id AND eq.parent_id = a.node_id
      AND eq.type = '='
    LEFT JOIN ast_nodes ty ON ty.fact_id = a.fact_id AND ty.parent_id = a.node_id
      AND ty.type = 'type'
    LEFT JOIN ast_nodes v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
      AND v.sibling_index = eq.sibling_index + 1
  ),
  classified AS (
    SELECT s.*, g.target, g.target_type, g.assignment_type,
           coalesce(g.annotated, false) AS annotated, g.value, g.value_type, g.value_size,
           s.type IN ('import_statement', 'import_from_statement', 'future_import_statement')
             AS is_import,
           coalesce(g.assignment_type = 'assignment' AND g.target = '__all__', false)
             AS states_all,
           coalesce(g.assignment_type = 'augmented_assignment' AND g.target = '__all__', false)
             AS extends_all,
           s.type = 'expression_statement' AND s.children_count = 1
             AND s.expression_type IN ('string', 'concatenated_string')
             AND NOT regexp_matches(s.expression_name, '^[A-Za-z]*[fF][A-Za-z]*["'']')
             AS is_docstring,
           coalesce(g.assignment_type = 'assignment' AND g.target_type = 'identifier'
             AND g.has_value AND g.value_type <> 'assignment' AND NOT starts_with(g.target, '_')
             AND regexp_matches(g.target, '[A-Za-z]') AND NOT regexp_matches(g.target, '[a-z]'),
             false) AS is_constant
    FROM shape s LEFT JOIN targets g ON g.fact_id = s.fact_id AND g.node_id = s.node_id
  )
"""
)

_PLACEMENTS = (
    _TOP
    + r""",
  typed_guard AS (
    SELECT DISTINCT c.fact_id, c.node_id FROM classified c
    JOIN ast_nodes x ON x.fact_id = c.fact_id AND x.parent_id = c.node_id
      AND x.sibling_index = 1 AND x.type = 'identifier' AND x.name = 'TYPE_CHECKING'
    WHERE c.type = 'if_statement'
  ),
  scaffolding AS (
    SELECT c.fact_id, c.idx,
           NOT (c.is_import OR c.is_docstring OR c.is_constant OR t.node_id IS NOT NULL
                OR coalesce(c.assignment_type = 'assignment' AND NOT c.annotated
                    AND (c.target IN ('pytestmark', 'pytest_plugins')
                         OR (c.target_type = 'identifier' AND starts_with(c.target, '__'))),
                    false))
             AS intervenes
    FROM classified c LEFT JOIN typed_guard t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
  ),
  declared_names AS (
    SELECT fact_id, idx,
           CASE WHEN definition_type IS NOT NULL THEN definition_name
                WHEN type IN ('function_definition', 'class_definition') THEN name
                WHEN assignment_type = 'assignment' AND target_type IN ('identifier', 'attribute')
                     AND coalesce(value_type, '') <> 'assignment' THEN target END AS name
    FROM classified
  ),
  anchored AS (
    SELECT c.*, max(CASE WHEN c.is_import THEN c.idx END) OVER (
             PARTITION BY c.fact_id ORDER BY c.idx ROWS BETWEEN UNBOUNDED PRECEDING
             AND 1 PRECEDING) AS latest_import
    FROM classified c
  ),
  dependencies AS (
    SELECT c.fact_id, c.idx, max(d.idx) AS anchor
    FROM anchored c
    JOIN ast_nodes i ON i.fact_id = c.fact_id
      AND i.node_id BETWEEN c.value AND c.value + c.value_size
    """
    + name_reads()[0]
    + r"""
    JOIN declared_names d ON d.fact_id = c.fact_id AND d.name = i.name AND d.idx < c.idx
    WHERE c.is_constant AND """
    + name_reads()[1]
    + r"""
    GROUP BY ALL
  ),
  placements AS (
    SELECT c.fact_id, c.path, c.idx, c.target AS name,
           (SELECT count(*) FROM scaffolding s
            WHERE s.fact_id = c.fact_id AND s.intervenes
              AND s.idx > greatest(coalesce(d.anchor, -1), coalesce(c.latest_import, -1))
              AND s.idx < c.idx) AS intervening
    FROM anchored c LEFT JOIN dependencies d ON d.fact_id = c.fact_id AND d.idx = c.idx
    WHERE c.is_constant
  )
"""
)

_FOREIGN = r"""
  foreign_files AS (
    SELECT fact_id, path, language FROM ast_files
    WHERE language IN ('typescript', 'c', 'cpp', 'cuda', 'rust')
  ),
  declared AS (
    SELECT fact_id, node_id, type FROM ast_nodes
    WHERE parent_id = 0 AND type <> 'namespace_definition'
    UNION ALL
    SELECT c.fact_id, c.node_id, c.type
    FROM ast_nodes t JOIN ast_nodes c ON c.fact_id = t.fact_id AND c.parent_id = t.node_id
    WHERE t.parent_id = 0 AND t.type IN ('export_statement', 'ambient_declaration')
      AND NOT EXISTS (SELECT 1 FROM ast_nodes d WHERE d.fact_id = t.fact_id
                        AND d.parent_id = t.node_id AND d.type = 'default')
    UNION ALL
    SELECT c.fact_id, c.node_id, c.type
    FROM ast_nodes t
    JOIN ast_nodes a ON a.fact_id = t.fact_id AND a.parent_id = t.node_id
      AND a.type = 'ambient_declaration'
    JOIN ast_nodes c ON c.fact_id = a.fact_id AND c.parent_id = a.node_id
    WHERE t.parent_id = 0 AND t.type = 'export_statement'
    UNION ALL
    SELECT c.fact_id, c.node_id, c.type
    FROM ast_nodes t
    JOIN ast_nodes b ON b.fact_id = t.fact_id AND b.parent_id = t.node_id
      AND b.type = 'declaration_list'
    JOIN ast_nodes c ON c.fact_id = b.fact_id AND c.parent_id = b.node_id
    WHERE t.parent_id = 0 AND t.type = 'namespace_definition'
  ),
  foreign_sizes AS (
    SELECT f.fact_id,
           count(*) FILTER (WHERE d.type IN ('class_declaration', 'abstract_class_declaration',
                                             'class', 'class_specifier', 'struct_specifier',
                                             'union_specifier', 'enum_specifier', 'struct_item',
                                             'enum_item', 'union_item', 'trait_item'))
             AS class_count,
           count(*) FILTER (WHERE d.type IN ('function_declaration', 'function_signature',
                                             'generator_function_declaration',
                                             'function_definition', 'function_item'))
             AS function_count
    FROM foreign_files f JOIN declared d ON d.fact_id = f.fact_id GROUP BY f.fact_id
  ),
  statement_nodes AS (
    SELECT f.fact_id, n.node_id
    FROM foreign_files f JOIN ast_nodes n ON n.fact_id = f.fact_id
    WHERE f.language <> 'rust' AND CASE WHEN f.language = 'typescript' THEN
            n.type IN ('expression_statement', 'return_statement', 'if_statement', 'do_statement',
                       'while_statement', 'for_statement', 'for_in_statement', 'with_statement',
                       'switch_statement', 'try_statement', 'continue_statement',
                       'break_statement', 'throw_statement', 'debugger_statement',
                       'labeled_statement', 'empty_statement', 'lexical_declaration',
                       'variable_declaration', 'function_declaration', 'function_signature',
                       'generator_function_declaration', 'function_expression',
                       'generator_function', 'method_definition', 'class_declaration',
                       'abstract_class_declaration', 'import_statement', 'enum_declaration',
                       'type_alias_declaration', 'interface_declaration', 'internal_module')
            OR (n.type IN ('class', 'module') AND n.descendant_count > 0)
          ELSE n.type LIKE '%\_statement' ESCAPE '\' OR n.type IN ('declaration',
                 'function_definition', 'namespace_definition', 'preproc_call', 'preproc_def',
                 'preproc_function_def', 'preproc_include') END
  ),
  rust_statements AS (
    SELECT f.fact_id, c.node_id
    FROM foreign_files f
    JOIN ast_nodes c ON c.fact_id = f.fact_id
    LEFT JOIN ast_nodes b ON b.fact_id = c.fact_id AND b.node_id = c.parent_id
    WHERE f.language = 'rust' AND c.type NOT IN ('{', '}', 'line_comment', 'block_comment',
                                                 'attribute_item', 'inner_attribute_item',
                                                 'shebang', 'empty_statement')
      AND (c.parent_id = 0 OR b.type = 'block')
  ),
  statement_extras AS (
    SELECT f.fact_id, n.node_id
    FROM foreign_files f JOIN ast_nodes n ON n.fact_id = f.fact_id
    JOIN ast_nodes k ON k.fact_id = n.fact_id AND k.parent_id = n.node_id
    WHERE f.language = 'typescript'
      AND ((n.type = 'for_in_statement' AND k.type IN ('const', 'let', 'var'))
           OR (n.type = 'export_statement'
               AND k.type IN ('*', 'namespace_export', '=', 'namespace')))
    UNION ALL
    SELECT f.fact_id, n.node_id
    FROM foreign_files f JOIN ast_nodes n ON n.fact_id = f.fact_id
    WHERE f.language = 'typescript' AND n.type = 'arrow_function'
      AND NOT EXISTS (SELECT 1 FROM ast_nodes k WHERE k.fact_id = n.fact_id
                        AND k.parent_id = n.node_id AND k.type = 'statement_block')
  ),
  foreign_statements AS (
    SELECT fact_id, count(*) AS statement_count FROM (
      SELECT s.fact_id, s.node_id FROM statement_nodes s
      JOIN ast_nodes x ON x.fact_id = s.fact_id AND x.node_id = s.node_id
      JOIN ast_nodes up ON up.fact_id = x.fact_id AND up.node_id = x.parent_id
      WHERE NOT (x.type = 'expression_statement' AND EXISTS (
                   SELECT 1 FROM ast_nodes k
                   WHERE k.fact_id = x.fact_id AND k.parent_id = x.node_id
                     AND k.type = 'internal_module'))
        AND NOT (x.type = 'empty_statement' AND up.type = 'for_statement')
      UNION ALL SELECT fact_id, node_id FROM statement_extras
      UNION ALL SELECT fact_id, node_id FROM rust_statements
    ) GROUP BY fact_id
  ),
  foreign_modules AS (
    SELECT f.fact_id, f.path, f.language, f.language IN ('typescript', 'rust') AS flat,
           coalesce(s.class_count, 0) AS class_count,
           coalesce(s.function_count, 0) AS function_count,
           coalesce(z.statement_count, 0) AS statement_count
    FROM foreign_files f
    LEFT JOIN foreign_sizes s ON s.fact_id = f.fact_id
    LEFT JOIN foreign_statements z ON z.fact_id = f.fact_id
  )
"""


class Module(Family):
    family = "ModuleFact"
    verified = True
    relations = {
        "facts": rf"""
            WITH {_TOP}, {_FOREIGN}, sizes AS (
              SELECT n.fact_id, count(*) AS statement_count
              FROM ast_nodes n JOIN python_files f ON f.fact_id = n.fact_id
              WHERE n.type LIKE '%\_statement' ESCAPE '\' OR n.type IN ('function_definition',
                                                                       'class_definition')
              GROUP BY n.fact_id
            ), ends AS (
              SELECT fact_id, start_line, start_column, end_line, end_column FROM ast_nodes
              WHERE node_id = 0
            ), summary AS (
              SELECT c.fact_id,
                     count(*) FILTER (WHERE coalesce(c.definition_type, c.type)
                                            = 'class_definition') AS class_count,
                     count(*) FILTER (WHERE coalesce(c.definition_type, c.type)
                                            = 'function_definition') AS function_count,
                     count(*) - count(*) FILTER (WHERE c.idx = 0 AND c.is_docstring)
                       AS executable_statement_count,
                     bool_and(c.is_import OR c.states_all OR c.extends_all)
                       AS has_only_imports_and_all,
                     count(*) FILTER (WHERE c.states_all OR c.extends_all) AS all_declarations
              FROM classified c GROUP BY c.fact_id
            ), declares AS (
              SELECT DISTINCT a.fact_id FROM ast_nodes a
              JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
                AND l.sibling_index = 0 AND l.type = 'identifier' AND l.name = '__all__'
              LEFT JOIN ast_nodes eq ON eq.fact_id = a.fact_id AND eq.parent_id = a.node_id
                AND eq.type = '='
              WHERE a.type IN ('assignment', 'augmented_assignment')
                AND (a.type = 'augmented_assignment' OR eq.node_id IS NOT NULL)
                AND a.scope.function IS NULL AND a.scope.class IS NULL
            ), python_facts AS (
            SELECT 'module:' || f.path AS fact_id, f.path, 1::UBIGINT AS start_line,
                   0::UBIGINT AS start_column, e.end_line::UBIGINT AS end_line,
                   e.end_column::UBIGINT AS end_column, 'python' AS language,
                   coalesce(s.all_declarations, 0)::BIGINT AS all_declarations_length,
                   true AS all_declarations_present,
                   coalesce(s.class_count, 0)::BIGINT AS class_count,
                   d.fact_id IS NOT NULL AS declares_all, 0::BIGINT AS evidence_length,
                   true AS evidence_present,
                   coalesce(s.executable_statement_count, 0)::BIGINT
                     AS executable_statement_count,
                   coalesce(s.function_count, 0)::BIGINT AS function_count,
                   coalesce(s.has_only_imports_and_all, true) AS has_only_imports_and_all,
                   ends_with(f.path, '__init__.py') AS is_package_initializer,
                   {is_test_path("f.path")} AS is_test,
                   (coalesce(s.class_count, 0) + coalesce(s.function_count, 0))::BIGINT
                     AS members_length,
                   true AS members_present,
                   (length(l.content) - length(replace(l.content, chr(10), ''))
                    + CASE WHEN l.content <> '' AND NOT ends_with(l.content, chr(10)) THEN 1
                           ELSE 0 END)::BIGINT AS physical_line_count,
                   coalesce(z.statement_count, 0)::BIGINT AS statement_count
            FROM python_files f
            JOIN source_files l ON l.path = f.path
            LEFT JOIN ends e ON e.fact_id = f.fact_id
            LEFT JOIN summary s ON s.fact_id = f.fact_id
            LEFT JOIN declares d ON d.fact_id = f.fact_id
            LEFT JOIN sizes z ON z.fact_id = f.fact_id
            ), other_facts AS (
            SELECT 'module:' || m.path AS fact_id, m.path,
                   CASE WHEN m.flat THEN 1 ELSE r.start_line END::UBIGINT AS start_line,
                   CASE WHEN m.flat THEN 0 ELSE r.start_column END::UBIGINT AS start_column,
                   CASE WHEN m.flat THEN 1 ELSE r.end_line END::UBIGINT AS end_line,
                   CASE WHEN m.flat THEN 0 ELSE r.end_column END::UBIGINT AS end_column,
                   m.language, 0::BIGINT AS all_declarations_length,
                   true AS all_declarations_present, m.class_count::BIGINT AS class_count,
                   false AS declares_all, 0::BIGINT AS evidence_length, true AS evidence_present,
                   0::BIGINT AS executable_statement_count,
                   m.function_count::BIGINT AS function_count, false AS has_only_imports_and_all,
                   CASE m.language WHEN 'typescript' THEN ends_with(m.path, '/index.ts')
                                    WHEN 'rust' THEN ends_with(m.path, '/mod.rs')
                                                  OR ends_with(m.path, '/lib.rs')
                                    ELSE false END
                     AS is_package_initializer,
                   {is_test_path("m.path")} AS is_test,
                   (m.class_count + m.function_count)::BIGINT AS members_length,
                   true AS members_present,
                   (length(l.content) - length(replace(l.content, chr(10), ''))
                    + CASE WHEN l.content <> '' AND NOT ends_with(l.content, chr(10)) THEN 1
                           ELSE 0 END)::BIGINT AS physical_line_count,
                   m.statement_count::BIGINT AS statement_count
            FROM foreign_modules m JOIN source_files l ON l.path = m.path
            JOIN ends r ON r.fact_id = m.fact_id
            )
            SELECT (row_number() OVER (ORDER BY path) - 1)::UBIGINT AS fact_order, *
            FROM (SELECT * FROM python_facts UNION ALL BY NAME SELECT * FROM other_facts)
        """,
        "all_declarations": rf"""
            WITH {_TOP}, {LINES}
            SELECT 'module:' || c.path AS fact_id,
                   (row_number() OVER (PARTITION BY c.fact_id ORDER BY c.idx) - 1)::UBIGINT
                     AS ordinal,
                   c.path || ':' || c.node_id || ':statement' AS id, c.path AS span_path,
                   c.start_line::BIGINT AS span_start_line,
                   c.start_column::BIGINT AS span_start_column,
                   c.end_line::BIGINT AS span_end_line, c.end_column::BIGINT AS span_end_column,
                   import_source_slice(l.lines, c.start_line, c.start_column, c.end_line,
                                       c.end_column)
                     AS text
            FROM classified c JOIN lines l ON l.path = c.path
            WHERE c.states_all OR c.extends_all
        """,
        "constant_placements": rf"""
            WITH {_PLACEMENTS}
            SELECT 'module:' || p.path AS fact_id,
                   (row_number() OVER (PARTITION BY p.fact_id ORDER BY p.idx) - 1)::UBIGINT
                     AS ordinal,
                   p.name, p.intervening::BIGINT AS intervening_statement_count
            FROM placements p
        """,
        "members": rf"""
            WITH {_TOP}
            SELECT 'module:' || path AS fact_id,
                   (row_number() OVER (PARTITION BY fact_id ORDER BY idx) - 1)::UBIGINT AS ordinal,
                   CASE coalesce(definition_type, type) WHEN 'class_definition' THEN 'class'
                        ELSE 'function' END AS kind,
                   coalesce(definition_name, name) AS name
            FROM classified
            WHERE coalesce(definition_type, type) IN ('class_definition', 'function_definition')
        """,
    }
