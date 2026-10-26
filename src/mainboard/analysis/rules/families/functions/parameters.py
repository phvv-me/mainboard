"""mcmr's ParameterFact, derived from sitting_duck's syntax trees (Python only).

For every annotated parameter of every function, mcmr read the body for what it does with the
name: the methods called on it, the builtins that read it (`len`, `iter`, `sorted` ...), subscripts
(`getitem`, `setitem`, `delitem`), membership tests, arithmetic, iteration, the attributes read off
it, whether it is returned and whether some other use was not recognized. It walks the
statements of the body top down; the relations here port that walk bottom up, one occurrence of
the name at a time, in three steps:

- `tree` is the expression tree as mcmr's parser has it: no parentheses (a lone `(x)` target
  included), and without the `*` tree-sitter hangs on the head of a lone starred element
  (`[*x.keys()]` parses as `(*x).keys()`), so every use sits under the node it has in mcmr;
- `sites` places each load of the name by the node that holds it and the one above;
- `uses` classifies the load from that table, the same rule mcmr applies:

  - a use that is the whole value of a place mcmr evaluates (`y = x`, `if x:`, `@x`, a default,
    an annotation, a base class ...) is dropped, and `return x` states the parameter is returned;
  - `x.attr` is an attribute read, `x.method(...)` an operation named for the method;
  - `x[...]` is `getitem`, `setitem` as the target of an assignment, `delitem` as a target of
    `del`;
  - a positional argument of a reading builtin is an operation named for the builtin;
  - the last operand of one `in` / `not in` is `contains`, an operand of arithmetic is
    `arithmetic`, an iterated name (`for`, comprehension) is `iter`;
  - any other use is unrecognized, so the parameter's uses are not all known;
  - the operand a binary operation does not hold the name in is never read.

mcmr never reaches some positions, so a name there is not a use at all: a `yield from`, a `match`
subject and its patterns, a `case` guard, a slice, the format specifier of an f-string field, the
message of an `assert`, the cause of a `raise ... from`, the type of an `except`, the keywords of
a class, and the parameters of a lambda.

Difference from the kernel, kept on purpose: the test of an `elif` is read as the use it is (the
kernel skips it).
"""

from ....catalog import Family

_FACTS = r"""
    SELECT (row_number() OVER (ORDER BY f.path) - 1)::UBIGINT AS fact_order,
           'parameterfact:' || f.path AS fact_id, f.path, 1 AS start_line, 0 AS start_column,
           coalesce(m.end_line, 1) AS end_line, coalesce(m.end_column, 0) AS end_column,
           'python' AS language
    FROM ast_files f
    LEFT JOIN (SELECT fact_id, end_line, end_column FROM ast_nodes WHERE parent_id IS NULL) m
      USING (fact_id)
    WHERE f.language = 'python'
"""

_DECLARED = r"""
    WITH defs AS (
      SELECT n.fact_id AS ast, n.node_id AS def, n.name AS owner, f.path
      FROM ast_nodes n JOIN ast_files f ON f.fact_id = n.fact_id AND f.language = 'python'
      WHERE n.type = 'function_definition'
    ), kids AS (
      SELECT d.ast, d.def, d.owner, d.path, k.node_id AS node, k.type, k.descendant_count AS size
      FROM defs d
      JOIN ast_nodes k ON k.fact_id = d.ast AND k.parent_id = d.def
        AND k.type IN ('block', 'parameters')
    ), declared AS (
      SELECT p.fact_id AS ast, b.def, b.owner, b.path, b.node AS body, b.size, p.node_id AS param,
             p.sibling_index, p.start_line, p.start_column, p.end_line, p.end_column,
             coalesce(nullif(p.name, ''), ltrim(c.name, '*')) AS name
      FROM kids b
      JOIN kids l ON l.ast = b.ast AND l.def = b.def AND l.type = 'parameters'
      JOIN ast_nodes p ON p.fact_id = l.ast AND p.parent_id = l.node
        AND p.type IN ('typed_parameter', 'typed_default_parameter')
      LEFT JOIN ast_nodes c ON c.fact_id = p.fact_id AND c.parent_id = p.node_id
        AND c.sibling_index = 0
      WHERE b.type = 'block'
    ), annotated AS (
      SELECT d.*, e.type AS shape, e.name AS head, e.node_id AS expression
      FROM declared d
      JOIN ast_nodes a ON a.fact_id = d.ast AND a.parent_id = d.param AND a.type = 'type'
      JOIN ast_nodes e ON e.fact_id = a.fact_id AND e.parent_id = a.node_id AND e.sibling_index = 0
    )
    SELECT a.ast, a.path, a.def, a.body, a.size, a.param, a.name, a.owner, a.start_line,
           a.start_column, a.end_line, a.end_column,
           row_number() OVER (PARTITION BY a.ast ORDER BY a.def, a.sibling_index) - 1 AS position,
           CASE WHEN a.shape = 'identifier' THEN a.head
                WHEN a.shape = 'attribute' THEN regexp_extract(a.head, '[^.]*$')
                WHEN a.shape IN ('generic_type', 'subscript')
                THEN regexp_extract(coalesce(v.name, ''), '[^.]*$')
                WHEN a.shape = 'string' THEN coalesce(t.name, '')
                WHEN a.shape = 'none' THEN 'None' ELSE '' END AS annotation
    FROM annotated a
    LEFT JOIN ast_nodes v ON v.fact_id = a.ast AND v.parent_id = a.expression
      AND v.sibling_index = 0 AND a.shape IN ('generic_type', 'subscript')
    LEFT JOIN ast_nodes t ON t.fact_id = a.ast AND t.parent_id = a.expression
      AND t.type = 'string_content' AND t.sibling_index = 0 AND a.shape = 'string'
"""

_TREE = r"""
    WITH RECURSIVE wrappers AS (
      SELECT n.fact_id, n.node_id, n.parent_id, n.sibling_index
      FROM ast_nodes n LEFT JOIN ast_nodes p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
      WHERE n.scope.function IS NOT NULL
        AND (n.type = 'parenthesized_expression'
             OR (n.type = 'list_splat' AND n.sibling_index = 0
                 AND p.type IN ('attribute', 'subscript', 'call'))
             OR (n.type = 'tuple_pattern' AND n.children_count = 3 AND EXISTS (
                   SELECT 1 FROM ast_nodes k
                   WHERE k.fact_id = n.fact_id AND k.parent_id = n.node_id
                     AND k.type NOT IN ('(', ')') AND k.end_line = n.end_line
                     AND k.end_column + 1 = n.end_column)))
    ), standing(fact_id, node_id, parent_id, sibling_index) AS (
      SELECT w.* FROM wrappers w
      ANTI JOIN wrappers p ON p.fact_id = w.fact_id AND p.node_id = w.parent_id
      UNION ALL
      SELECT w.fact_id, w.node_id, s.parent_id, s.sibling_index
      FROM wrappers w JOIN standing s ON s.fact_id = w.fact_id AND s.node_id = w.parent_id
    ), held AS (
      SELECT n.fact_id, n.node_id, s.parent_id, s.sibling_index,
             n.type IN ('(', ')', '*', 'comment') AS is_token
      FROM ast_nodes n JOIN standing s ON s.fact_id = n.fact_id AND s.node_id = n.parent_id
    )
    SELECT n.fact_id, n.node_id, n.type, n.name,
           coalesce(h.parent_id, n.parent_id) AS parent_id,
           coalesce(h.sibling_index, n.sibling_index) AS sibling_index,
           n.children_count, n.descendant_count
    FROM ast_nodes n
    LEFT JOIN held h ON h.fact_id = n.fact_id AND h.node_id = n.node_id
    WHERE n.scope.function IS NOT NULL AND NOT coalesce(h.is_token, false)
      AND NOT EXISTS (SELECT 1 FROM wrappers w
                      WHERE w.fact_id = n.fact_id AND w.node_id = n.node_id)
"""

_SITES = r"""
    WITH RECURSIVE patterns(fact_id, node_id, owner) AS (
      SELECT t.fact_id, t.node_id, p.type
      FROM parameter_tree t
      JOIN parameter_tree p ON p.fact_id = t.fact_id AND p.node_id = t.parent_id
      WHERE (t.type IN ('pattern_list', 'tuple_pattern', 'list_pattern', 'list_splat_pattern')
             AND p.type NOT IN ('pattern_list', 'tuple_pattern', 'list_pattern',
                                'list_splat_pattern'))
         OR (t.type IN ('tuple', 'list') AND p.type = 'as_pattern_target')
      UNION ALL
      SELECT t.fact_id, t.node_id, r.owner
      FROM parameter_tree t JOIN patterns r ON r.fact_id = t.fact_id AND t.parent_id = r.node_id
      WHERE t.type IN ('pattern_list', 'tuple_pattern', 'list_pattern', 'list_splat_pattern',
                       'tuple', 'list')
    ), occurrences AS (
      SELECT d.ast, d.param, d.name, o.node_id AS node, o.sibling_index AS sibling,
             a.node_id AS above, a.type AS above_type, a.sibling_index AS above_sibling,
             a.children_count AS above_kids, b.node_id AS beyond,
             coalesce(b.type, '') AS beyond_type, coalesce(c.type, '') AS outer_type,
             i.sibling_index AS later, r.owner AS pattern_owner
      FROM parameter_declared d
      JOIN parameter_tree o ON o.fact_id = d.ast AND o.type = 'identifier' AND o.name = d.name
        AND o.node_id BETWEEN d.body AND d.body + d.size
      JOIN parameter_tree a ON a.fact_id = o.fact_id AND a.node_id = o.parent_id
      LEFT JOIN parameter_tree b ON b.fact_id = a.fact_id AND b.node_id = a.parent_id
      LEFT JOIN parameter_tree c ON c.fact_id = b.fact_id AND c.node_id = b.parent_id
      LEFT JOIN parameter_tree i ON i.fact_id = a.fact_id AND i.parent_id = a.node_id
        AND i.type = 'in' AND a.type IN ('for_statement', 'for_in_clause')
      LEFT JOIN patterns r ON r.fact_id = a.fact_id AND r.node_id = a.node_id
    )
    SELECT x.*, t.name AS attribute, f.type AS callee_type, f.name AS callee, q.type AS operator
    FROM occurrences x
    LEFT JOIN parameter_tree t ON t.fact_id = x.ast AND t.parent_id = x.above
      AND t.sibling_index = 1 AND x.above_type = 'attribute'
    LEFT JOIN parameter_tree f ON f.fact_id = x.ast AND f.parent_id = x.beyond
      AND f.sibling_index = 0 AND x.above_type = 'argument_list' AND x.beyond_type = 'call'
    LEFT JOIN parameter_tree q ON q.fact_id = x.ast AND q.parent_id = x.above
      AND q.sibling_index = 1 AND x.above_type = 'comparison_operator'
    WHERE NOT functions_store_position(x.above_type, x.sibling, x.later, x.beyond_type)
       OR x.pattern_owner IN ('assignment', 'for_statement', 'as_pattern_target')
"""

_USES = r"""
    WITH binaries AS (
      SELECT b.fact_id AS ast, b.node_id AS node, b.descendant_count AS size, k.name
      FROM parameter_tree b
      JOIN parameter_tree k ON k.fact_id = b.fact_id AND k.parent_id = b.node_id
        AND k.type = 'identifier'
      WHERE b.type = 'binary_operator'
    ), unwalked AS (
      SELECT n.fact_id AS ast, n.node_id AS node, n.descendant_count AS size
      FROM parameter_tree n
      LEFT JOIN parameter_tree p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
      LEFT JOIN parameter_tree g ON g.fact_id = p.fact_id AND g.node_id = p.parent_id
      WHERE n.type IN ('slice', 'format_specifier', 'lambda_parameters', 'case_pattern')
         OR (p.type = 'assert_statement' AND n.sibling_index = 2)
         OR (p.type = 'raise_statement' AND n.sibling_index = 3)
         OR (p.type IN ('except_clause', 'except_group_clause') AND n.type <> 'block')
         OR (n.type IN ('keyword_argument', 'dictionary_splat') AND p.type = 'argument_list'
             AND g.type = 'class_definition')
         OR (n.type = 'yield' AND EXISTS (SELECT 1 FROM parameter_tree k
                                          WHERE k.fact_id = n.fact_id AND k.parent_id = n.node_id
                                            AND k.type = 'from'))
         OR (p.type = 'match_statement' AND n.sibling_index > 0 AND n.type NOT IN ('block', ','))
         OR (p.type = 'case_clause' AND n.type = 'if_clause')
    ), classified AS (
      SELECT s.ast, s.param, s.node,
             CASE
               WHEN s.above_type = 'attribute' AND s.sibling = 0
               THEN CASE WHEN s.beyond_type = 'call' AND s.above_sibling = 0
                         THEN 'operation:' || s.attribute ELSE 'attribute:' || s.attribute END
               WHEN s.above_type IN ('subscript', 'generic_type') AND s.sibling = 0
               THEN CASE WHEN s.beyond_type IN ('assignment', 'augmented_assignment')
                              AND s.above_sibling = 0 THEN 'operation:setitem'
                         WHEN s.beyond_type = 'delete_statement'
                              OR (s.beyond_type = 'expression_list'
                                  AND s.outer_type = 'delete_statement')
                         THEN 'operation:delitem'
                         ELSE 'operation:getitem' END
               WHEN s.above_type = 'argument_list' AND s.callee_type = 'identifier'
                    AND functions_reading_builtin(s.callee) AND s.sibling > 0
               THEN 'operation:' || s.callee
               WHEN s.above_type = 'comparison_operator' AND s.above_kids = 3
                    AND s.operator IN ('in', 'not in') AND s.sibling = 2
               THEN 'operation:contains'
               WHEN s.above_type = 'binary_operator' THEN 'operation:arithmetic'
               WHEN s.above_type IN ('for_statement', 'for_in_clause') THEN 'operation:iter'
               WHEN s.above_type = 'return_statement' THEN 'returned'
               -- A value assigned somewhere (`self.kept = param`) escapes the uses walked here.
               WHEN s.above_type = 'assignment' AND s.sibling > 0 THEN 'unrecognized'
               WHEN s.above_type IN ('expression_statement', 'assignment', 'augmented_assignment',
                                     'if_statement', 'elif_clause', 'while_statement',
                                     'raise_statement', 'assert_statement', 'delete_statement',
                                     'with_item', 'as_pattern', 'decorator',
                                     'default_parameter', 'typed_default_parameter',
                                     'type_alias_statement')
                    OR (s.above_type = 'type' AND s.beyond_type <> 'type_parameter')
                    OR (s.above_type = 'expression_list' AND s.beyond_type = 'delete_statement')
                    OR (s.above_type = 'argument_list' AND s.beyond_type = 'class_definition')
               THEN 'dropped'
               ELSE 'unrecognized' END AS kind
      FROM parameter_sites s
      WHERE NOT EXISTS (
        SELECT 1 FROM binaries b
        WHERE b.ast = s.ast AND b.name = s.name
          AND s.node > b.node AND s.node <= b.node + b.size AND s.above <> b.node)
        AND NOT EXISTS (
          SELECT 1 FROM unwalked w
          WHERE w.ast = s.ast AND s.node BETWEEN w.node AND w.node + w.size)
    )
    SELECT ast, param, node, kind FROM classified
"""

_PARAMETERS = r"""
    WITH summed AS (
      SELECT ast, param,
             count(*) FILTER (WHERE kind = 'unrecognized') AS unrecognized,
             bool_or(kind = 'returned') AS returned,
             count(DISTINCT kind) FILTER (WHERE starts_with(kind, 'operation:')) AS operations,
             count(*) FILTER (WHERE starts_with(kind, 'attribute:')) AS attributes
      FROM parameter_uses GROUP BY ast, param
    )
    SELECT f.fact_order, f.fact_id, f.fact_id AS parent_id, d.ast, d.param,
           f.fact_id || '/parameters:' || d.position AS record_id, d.position::UBIGINT AS ordinal,
           coalesce(s.unrecognized, 0) = 0 AS all_uses_known, d.annotation,
           coalesce(s.attributes, 0) AS attribute_reads_length, true AS attribute_reads_present,
           coalesce(s.returned, false) AS is_return_value, d.name,
           coalesce(s.operations, 0) AS operations_length, true AS operations_present, d.owner,
           d.end_column AS span_end_column, d.end_line AS span_end_line, d.path AS span_path,
           d.start_column AS span_start_column, d.start_line AS span_start_line
    FROM parameter_declared d
    JOIN parameter_facts f ON f.path = d.path
    LEFT JOIN summed s ON s.ast = d.ast AND s.param = d.param
"""

_VALUES = r"""
    SELECT p.fact_order, p.fact_id, 'parameters.operations' AS relation,
           p.record_id AS parent_id, v.value AS string_value,
           row_number() OVER (PARTITION BY p.record_id ORDER BY v.value) - 1 AS ordinal
    FROM parameter_parameters p
    JOIN (SELECT DISTINCT ast, param, substr(kind, 11) AS value FROM parameter_uses
          WHERE starts_with(kind, 'operation:')) v ON v.ast = p.ast AND v.param = p.param
    UNION ALL
    SELECT p.fact_order, p.fact_id, 'parameters.attribute_reads', p.record_id,
           substr(u.kind, 11), row_number() OVER (PARTITION BY p.record_id ORDER BY u.node) - 1
    FROM parameter_parameters p
    JOIN parameter_uses u ON u.ast = p.ast AND u.param = p.param
    WHERE starts_with(u.kind, 'attribute:')
"""


class Parameters(Family):
    family = "ParameterFact"
    relations = {
        "facts": _FACTS,
        "declared": _DECLARED,
        "tree": _TREE,
        "sites": _SITES,
        "uses": _USES,
        "parameters": _PARAMETERS,
        "values": _VALUES,
    }
