"""The relations of FunctionFact that read a function's body: which function owns a node, the
nodes inside each function (nested functions included, as mcmr walks them), their measures and the
asyncio evidence."""

OWNERS = r"""
    WITH lam AS (
      SELECT ast, node AS lam, function_scope AS above FROM function_nodes WHERE type = 'lambda'
    )
    SELECT c.ast, c.node,
           CASE WHEN l1.lam IS NULL THEN c.function_scope WHEN l2.lam IS NULL THEN l1.above
                ELSE l2.above END AS def
    FROM function_calls c
    LEFT JOIN lam l1 ON l1.ast = c.ast AND l1.lam = c.function_scope
    LEFT JOIN lam l2 ON l2.ast = c.ast AND l2.lam = l1.above
"""

CHAIN = r"""
    WITH RECURSIVE chain(ast, below, above) AS (
      SELECT ast, node, node FROM function_defs
      UNION ALL
      SELECT c.ast, c.below, d.enclosing
      FROM chain c JOIN function_defs d ON d.ast = c.ast AND d.node = c.above
      WHERE d.enclosing IS NOT NULL
    )
    SELECT * FROM chain
"""

RECEIVERS = r"""
    SELECT d.ast, d.node AS def, p.name AS receiver
    FROM function_defs d
    JOIN function_parameters p ON p.function_id = d.entity_id AND p.ordinal = 0
      AND NOT p.is_keyword_only AND p.name IN ('self', 'cls')
    WHERE d.scope = 'method'
      AND NOT EXISTS (SELECT 1 FROM function_decorators x
                      WHERE x.function_id = d.entity_id
                        AND functions_decorator_name(x.decorator) = 'staticmethod')
"""

TAKEN = r"""
    SELECT b.ast, b.def, coalesce(i.name, regexp_extract(k.name, '[^*]+$')) AS name
    FROM function_bodies b
    JOIN function_nodes k ON k.ast = b.ast AND k.parent = b.params
    LEFT JOIN function_nodes i ON i.ast = k.ast AND i.parent = k.node AND i.sibling = 0
      AND i.type = 'identifier' AND k.type IN ('list_splat_pattern', 'dictionary_splat_pattern')
    WHERE k.type IN ('identifier', 'typed_parameter', 'default_parameter',
                     'typed_default_parameter', 'list_splat_pattern', 'dictionary_splat_pattern')
"""

INSIDE = r"""
    WITH lam AS (
      SELECT ast, node AS lam, function_scope AS above FROM function_nodes WHERE type = 'lambda'
    ), targets AS (
      SELECT a.ast, l.node AS lo, l.node + l.size AS hi
      FROM function_nodes a
      JOIN function_nodes l ON l.ast = a.ast AND l.parent = a.node AND l.sibling = 0
      WHERE a.type IN ('assignment', 'augmented_assignment')
    ), interest AS (
      SELECT n.ast, n.node, n.type, n.name, n.kids,
             CASE WHEN l1.lam IS NULL THEN n.function_scope WHEN l2.lam IS NULL THEN l1.above
                  ELSE l2.above END AS owner,
             n.type = 'boolean_operator' AND p.node IS NOT NULL AND o1.type = o2.type AS merged
      FROM function_nodes n
      LEFT JOIN lam l1 ON l1.ast = n.ast AND l1.lam = n.function_scope
      LEFT JOIN lam l2 ON l2.ast = n.ast AND l2.lam = l1.above
      LEFT JOIN function_nodes p ON p.ast = n.ast AND p.node = n.parent
        AND n.type = 'boolean_operator' AND p.type = 'boolean_operator' AND n.sibling = 0
      LEFT JOIN function_nodes o1 ON o1.ast = n.ast AND o1.parent = n.node AND o1.sibling = 1
        AND n.type = 'boolean_operator'
      LEFT JOIN function_nodes o2 ON o2.ast = p.ast AND o2.parent = p.node AND o2.sibling = 1
      WHERE functions_behavior_type(n.type) OR n.type = 'raise_statement'
         OR (n.type IN ('await', 'yield') AND n.kids > 0)
         OR (n.type = 'identifier' AND n.name IN ('self', 'cls'))
         OR (n.type IN ('attribute', 'identifier') AND n.name LIKE '%TaskGroup')
    )
    SELECT i.ast, c.above AS def, i.node, i.type, i.name, i.kids, i.merged,
           t.lo IS NOT NULL AS targeted
    FROM interest i
    JOIN function_chain c ON c.ast = i.ast AND c.below = i.owner
    JOIN function_bodies b ON b.ast = c.ast AND b.def = c.above AND i.node > b.body
    LEFT JOIN targets t ON t.ast = i.ast AND i.node BETWEEN t.lo AND t.hi
      AND i.type = 'identifier'
"""

ROOTS = r"""
    WITH wanted AS (
      SELECT a.ast, a.node, a.type, a.name
      FROM function_calls c
      JOIN function_nodes al ON al.ast = c.ast AND al.parent = c.node AND al.sibling = 1
      JOIN function_nodes a ON a.ast = al.ast AND a.parent = al.node AND a.sibling >= 1
      WHERE regexp_extract(c.callee, '[^.]*$') IN ('isinstance', 'issubclass', 'gather')
      UNION
      SELECT l.ast, l.node, l.type, l.name
      FROM function_nodes a
      JOIN function_nodes l ON l.ast = a.ast AND l.parent = a.node AND l.sibling = 0
      WHERE a.type = 'assignment'
    )
    SELECT w.ast, w.node,
           coalesce(functions_root_name(w.type, w.name), functions_root_name(h1.type, h1.name),
                    functions_root_name(h2.type, h2.name),
                    functions_root_name(h3.type, h3.name)) AS root
    FROM wanted w
    LEFT JOIN function_nodes h1 ON h1.ast = w.ast AND h1.parent = w.node
      AND h1.sibling = functions_operand(w.type)
    LEFT JOIN function_nodes h2 ON h2.ast = h1.ast AND h2.parent = h1.node
      AND h2.sibling = functions_operand(h1.type)
    LEFT JOIN function_nodes h3 ON h3.ast = h2.ast AND h3.parent = h2.node
      AND h3.sibling = functions_operand(h2.type)
"""

MEASURES = r"""
    WITH checks AS (
      SELECT DISTINCT i.ast, i.def
      FROM function_inside i
      JOIN function_calls c ON c.ast = i.ast AND c.node = i.node
      JOIN function_nodes al ON al.ast = c.ast AND al.parent = c.node AND al.sibling = 1
      JOIN function_nodes a ON a.ast = al.ast AND a.parent = al.node AND a.sibling = 1
      JOIN function_roots r ON r.ast = a.ast AND r.node = a.node
      JOIN function_taken p ON p.ast = i.ast AND p.def = i.def AND p.name = r.root
      WHERE i.type = 'call' AND regexp_extract(c.callee, '[^.]*$') IN ('isinstance', 'issubclass')
    ), raised AS (
      SELECT DISTINCT i.ast, i.def
      FROM function_inside i
      JOIN function_nodes e ON e.ast = i.ast AND e.parent = i.node AND e.sibling = 1
      LEFT JOIN function_calls c ON c.ast = e.ast AND c.node = e.node
      WHERE i.type = 'raise_statement'
        AND functions_validation_exception(regexp_extract(coalesce(c.callee, e.name), '[^.]*$'))
    ), classes AS (
      SELECT ast, node, name FROM function_nodes WHERE type = 'class_definition'
    ), summed AS (
      SELECT i.ast, i.def,
             count(*) FILTER (WHERE (functions_behavior_type(i.type)
                                    AND i.type <> 'boolean_operator')
                              OR (i.type = 'boolean_operator' AND NOT i.merged)
                              OR i.type IN ('await', 'yield')) AS behavior_operation_count,
             coalesce(bool_or(c.callee = d.name
                              OR (r.receiver IS NOT NULL
                                  AND c.callee = r.receiver || '.' || d.name))
                        FILTER (WHERE i.type = 'call'), false) AS is_recursive,
             coalesce(bool_or(c.callee = k.name OR (r.receiver = 'cls' AND c.callee = 'cls'))
                        FILTER (WHERE i.type = 'call'), false) AS constructs_owner_model,
             coalesce(bool_or(i.name = r.receiver AND NOT i.targeted)
                        FILTER (WHERE i.type = 'identifier'), false) AS reads_receiver
      FROM function_inside i
      JOIN function_defs d ON d.ast = i.ast AND d.node = i.def
      LEFT JOIN function_calls c ON c.ast = i.ast AND c.node = i.node AND i.type = 'call'
      LEFT JOIN function_receivers r ON r.ast = i.ast AND r.def = i.def
      LEFT JOIN classes k ON k.ast = d.ast AND k.node = d.owner
      GROUP BY i.ast, i.def
    )
    SELECT d.ast, d.node AS def,
           coalesce(s.behavior_operation_count, 0) AS behavior_operation_count,
           coalesce(s.is_recursive, false) AS is_recursive,
           coalesce(s.constructs_owner_model, false) AS constructs_owner_model,
           coalesce(s.reads_receiver, false) AS reads_receiver,
           ck.def IS NOT NULL AS checks_raw_input_type,
           rs.def IS NOT NULL AS raises_validation_exception
    FROM function_defs d
    LEFT JOIN summed s ON s.ast = d.ast AND s.def = d.node
    LEFT JOIN checks ck ON ck.ast = d.ast AND ck.def = d.node
    LEFT JOIN raised rs ON rs.ast = d.ast AND rs.def = d.node
"""

TASKS = r"""
    WITH vocabulary(name, kind) AS (
      VALUES ('create_task', 'create'), ('ensure_future', 'create'), ('gather', 'gather'),
             ('TaskGroup', 'group')
    ), spelled AS (
      SELECT i.ast, i.bound || '.' || v.name AS spelling, v.kind
      FROM function_imports i, vocabulary v WHERE i.origin = 'asyncio' AND i.target = 'asyncio'
      UNION ALL
      SELECT i.ast, i.bound, v.kind
      FROM function_imports i JOIN vocabulary v ON i.target = 'asyncio.' || v.name
      WHERE i.origin = 'asyncio'
    ), files AS (
      SELECT DISTINCT ast FROM spelled
    ), tagged AS (
      SELECT i.ast, i.def, i.node, i.type, s.kind, c.callee
      FROM function_inside i SEMI JOIN files f ON f.ast = i.ast
      LEFT JOIN function_calls c ON c.ast = i.ast AND c.node = i.node AND i.type = 'call'
      JOIN spelled s ON s.ast = i.ast AND s.spelling = coalesce(c.callee, i.name)
      WHERE i.type IN ('call', 'attribute', 'identifier')
    ), made AS (
      SELECT ast, def, node FROM tagged WHERE type = 'call' AND kind = 'create'
    ), gathered AS (
      SELECT t.ast, t.def, t.node, al.node AS arguments,
             coalesce(bool_or(k.node IS NOT NULL), false) AS returns_exceptions
      FROM tagged t
      JOIN function_nodes al ON al.ast = t.ast AND al.parent = t.node AND al.sibling = 1
      LEFT JOIN function_nodes k ON k.ast = al.ast AND k.parent = al.node
        AND k.type = 'keyword_argument' AND k.name = 'return_exceptions'
        AND EXISTS (SELECT 1 FROM function_nodes v WHERE v.ast = k.ast AND v.parent = k.node
                    AND v.type = 'true')
      WHERE t.type = 'call' AND t.kind = 'gather'
      GROUP BY t.ast, t.def, t.node, al.node
    ), bindings AS (
      SELECT b.ast, b.def, lr.root AS name
      FROM function_bodies b SEMI JOIN files f ON f.ast = b.ast
      JOIN function_nodes blk ON blk.ast = b.ast AND blk.node = b.body
      JOIN function_nodes a ON a.ast = b.ast AND a.node BETWEEN blk.node AND blk.node + blk.size
        AND a.type = 'assignment'
      JOIN function_nodes l ON l.ast = a.ast AND l.parent = a.node AND l.sibling = 0
      JOIN function_roots lr ON lr.ast = l.ast AND lr.node = l.node
      JOIN function_nodes r ON r.ast = a.ast AND r.parent = a.node AND r.sibling = a.kids - 1
      WHERE EXISTS (SELECT 1 FROM made m WHERE m.ast = b.ast AND m.def = b.def
                    AND m.node BETWEEN r.node AND r.node + r.size)
      UNION
      SELECT t.ast, t.def, split_part(c.callee, '.', 1)
      FROM function_inside t SEMI JOIN files f ON f.ast = t.ast
      JOIN function_calls c ON c.ast = t.ast AND c.node = t.node AND t.type = 'call'
      JOIN function_nodes al ON al.ast = c.ast AND al.parent = c.node AND al.sibling = 1
      JOIN function_nodes a ON a.ast = al.ast AND a.parent = al.node
      WHERE regexp_matches(c.callee, '^[^.]+(\.[^.]+)*\.append$')
        AND EXISTS (SELECT 1 FROM made m WHERE m.ast = t.ast AND m.def = t.def AND m.node = a.node)
    ), consuming AS (
      SELECT DISTINCT g.ast, g.def
      FROM gathered g
      JOIN function_nodes a ON a.ast = g.ast AND a.parent = g.arguments
        AND a.type NOT IN ('(', ')', ',', 'keyword_argument', 'comment')
      LEFT JOIN function_roots r ON r.ast = a.ast AND r.node = a.node
      WHERE EXISTS (SELECT 1 FROM made m WHERE m.ast = g.ast AND m.def = g.def
                    AND m.node BETWEEN a.node AND a.node + a.size)
         OR EXISTS (SELECT 1 FROM bindings b WHERE b.ast = g.ast AND b.def = g.def
                    AND b.name = r.root)
    )
    SELECT d.ast, d.node AS def,
           (SELECT count(*) FROM made m WHERE m.ast = d.ast AND m.def = d.node)
             AS created_task_count,
           EXISTS (SELECT 1 FROM tagged t WHERE t.ast = d.ast AND t.def = d.node
                   AND t.kind = 'group') AS has_task_group,
           EXISTS (SELECT 1 FROM gathered g WHERE g.ast = d.ast AND g.def = d.node
                   AND g.returns_exceptions) AS gather_returns_exceptions,
           EXISTS (SELECT 1 FROM consuming c WHERE c.ast = d.ast AND c.def = d.node)
             AS gather_consumes_created_tasks
    FROM function_defs d SEMI JOIN files f ON f.ast = d.ast
"""

TENSORS = r"""
    WITH annotations AS (
      SELECT b.ast, b.def, p.name AS role, a.node, a.size
      FROM function_bodies b
      JOIN function_nodes p ON p.ast = b.ast AND p.parent = b.params
        AND p.type IN ('typed_parameter', 'typed_default_parameter')
      JOIN function_nodes a ON a.ast = p.ast AND a.parent = p.node AND a.type = 'type'
      UNION ALL
      SELECT b.ast, b.def, 'return', a.node, a.size
      FROM function_bodies b JOIN function_nodes a ON a.ast = b.ast AND a.node = b.returns
    ), named AS (
      SELECT a.ast, a.def, a.role, a.node AS annotation,
             coalesce(functions_root_name(p.type, p.name), t.name) AS root
      FROM annotations a
      JOIN function_nodes t ON t.ast = a.ast AND t.node BETWEEN a.node AND a.node + a.size
        AND t.type = 'identifier' AND functions_tensor_type(t.name)
      JOIN function_nodes p ON p.ast = t.ast AND p.node = t.parent
    ), wrapped AS (
      SELECT a.ast, a.def, a.role, a.node AS annotation,
             regexp_extract(v.name, '[^.]*$') AS wrapper,
             functions_root_name(v.type, v.name) AS root
      FROM annotations a
      JOIN function_nodes s ON s.ast = a.ast AND s.parent = a.node AND s.sibling = 0
        AND s.type = 'subscript'
      JOIN function_nodes v ON v.ast = s.ast AND v.parent = s.node AND v.sibling = 0
        AND v.type IN ('identifier', 'attribute')
      WHERE functions_tensor_wrapper(regexp_extract(v.name, '[^.]*$'))
        AND EXISTS (SELECT 1 FROM ast_nodes q WHERE q.fact_id = s.ast AND q.type = 'string'
                    AND q.node_id > v.node + v.size AND q.node_id <= s.node + s.size)
    ), tensors AS (
      SELECT n.ast, n.def, n.role, n.annotation, NULL AS wrapper, n.root FROM named n
      UNION ALL
      SELECT w.ast, w.def, w.role, w.annotation, w.wrapper, w.root FROM wrapped w
    ), resolved AS (
      SELECT t.ast, t.def, t.role, t.wrapper
      FROM tensors t
      WHERE functions_tensor_library(split_part(coalesce(
              (SELECT any_value(i.origin) FROM function_imports i
               WHERE i.ast = t.ast AND i.binding = t.root), t.root), '.', 1))
    )
    SELECT DISTINCT ast, def, role, wrapper FROM resolved
"""
