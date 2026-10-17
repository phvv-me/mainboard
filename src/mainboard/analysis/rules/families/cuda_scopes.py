"""The names of Python code as the PY-CUDA facts follow them: every call, the names a function or
module binds, and which binding each use of a name reads.

A scope is a function node (or the module, `-1`); a class body is not a scope a function reads
through, as in Python. Keys name a thing in the type tables: `e:<fact>:<node>` an expression,
`v:<fact>:<scope>:<name>` a variable.
"""

from ...catalog import Function
from .cuda_base import CudaRelation


class CudaShown(Function):
    """A call as a message shows it: on one line, and when longer than 80 characters cut after
    its last whole top-level argument that fits, or `callee(...)` where none does."""

    text: str
    callee: str

    expression = r"""
        list_transform([regexp_replace(regexp_replace(regexp_replace(text, '\s+', ' ', 'g'),
                                                      '([(\[{]) ', '\1', 'g'),
                                       ' ([)\]}])', '\1', 'g')], lambda n:
          CASE WHEN length(n) <= 80 THEN n
               ELSE coalesce(
                 list_transform([list_max(list_filter(range(1, 70), lambda i:
                     substr(n, i, 1) = ','
                     AND length(regexp_replace(left(n, i), '[^(\[{]', '', 'g'))
                         - length(regexp_replace(left(n, i), '[^)\]}]', '', 'g')) = 1))],
                   lambda k: left(n, k - 1) || ', ...)')[1],
                 callee || '(...)') END)[1]
    """


class CudaCalls(CudaRelation):
    """Every call, once: its node, the callee's resolved name and target (`call_calls`), its
    callee node and, for `receiver.method(...)`, the receiver node and the method's name, the
    function and class it sits in and its argument list."""

    stored = True
    query = r"""
        SELECT n.fact_id, c.node_path AS path, n.node_id, c.call_id, n.parent_id,
               c.node_start_line AS start_line, c.node_start_column AS start_column,
               c.node_end_line AS end_line, c.node_end_column AS end_column,
               c.qualified_name, c.target_id, c.assigned_target, c.callee_text,
               callee.node_id AS callee_id, callee.type AS callee_type,
               CASE WHEN callee.type = 'attribute'
                    THEN regexp_extract(c.callee_text, '\.([A-Za-z_][A-Za-z0-9_]*)$', 1) END
                 AS method,
               receiver.node_id AS receiver_id, coalesce(n.scope."function", -1) AS scope_id,
               n.scope."class" AS class_id, arguments.node_id AS arguments_id,
               regexp_matches(c.node_text, '\(\s*\)$') AS bare,
               cuda_shown(c.node_text, c.callee_text) AS shown,
               c.node_text
        FROM call_calls c
        JOIN cuda_nodes() n ON n.path = c.node_path AND n.type = 'call'
          AND n.start_line = c.node_start_line AND n.start_column = c.node_start_column
          AND n.end_line = c.node_end_line AND n.end_column = c.node_end_column
        LEFT JOIN cuda_nodes() callee ON callee.fact_id = n.fact_id AND callee.parent_id = n.node_id
          AND callee.sibling_index = 0
        LEFT JOIN cuda_nodes() receiver ON receiver.fact_id = n.fact_id
          AND receiver.parent_id = callee.node_id AND receiver.sibling_index = 0
          AND callee.type = 'attribute'
        LEFT JOIN cuda_nodes() arguments ON arguments.fact_id = n.fact_id
          AND arguments.parent_id = n.node_id AND arguments.type IN ('argument_list',
                                                                     'generator_expression')
    """


class CudaStores(CudaRelation):
    """The names a statement binds, not reads, one row per identifier: an assignment's targets, a
    loop's, a `with ... as`, a parameter. `kind` says how the name is bound: `assignment` (a
    plain `name = value`, `source` the value's node), `loop` (`source` the iterable), `unpack`
    (a tuple target: `source` the value, `position` its place in the tuple) or `other`. `scope_id` is the function (or `-1`) the name is local to."""

    stored = True
    query = r"""
        WITH names AS (
          SELECT i.fact_id, i.node_id, i.name, coalesce(i.scope."function", -1) AS scope_id,
                 i.sibling_index, p.node_id AS holder_id, p.type AS holder, p.parent_id AS up_id,
                 p.children_count AS holder_count
          FROM cuda_nodes() i JOIN cuda_nodes() p ON p.fact_id = i.fact_id AND p.node_id = i.parent_id
          WHERE i.type = 'identifier'
            AND NOT (i.scope."class" IS NOT NULL AND i.scope."function" IS NULL)
        ), owners AS (
          SELECT n.*, CASE WHEN n.holder IN ('pattern_list', 'tuple_pattern', 'list_pattern',
                                             'list_splat_pattern')
                           THEN u.type ELSE n.holder END AS owner,
                 CASE WHEN n.holder IN ('pattern_list', 'tuple_pattern', 'list_pattern',
                                        'list_splat_pattern')
                      THEN u.node_id ELSE n.holder_id END AS owner_id,
                 n.holder IN ('pattern_list', 'tuple_pattern', 'list_pattern',
                              'list_splat_pattern') AS nested
          FROM names n LEFT JOIN cuda_nodes() u ON u.fact_id = n.fact_id AND u.node_id = n.up_id
        ), outer_owners AS (
          SELECT o.*, CASE WHEN o.owner IN ('pattern_list', 'tuple_pattern', 'list_pattern')
                           THEN w.type ELSE o.owner END AS final_owner,
                 CASE WHEN o.owner IN ('pattern_list', 'tuple_pattern', 'list_pattern')
                      THEN w.node_id ELSE o.owner_id END AS final_id
          FROM owners o LEFT JOIN cuda_nodes() w ON w.fact_id = o.fact_id
            AND w.node_id = (SELECT x.parent_id FROM cuda_nodes() x
                             WHERE x.fact_id = o.fact_id AND x.node_id = o.owner_id)
        ), assigned AS (
          SELECT a.fact_id, a.node_id AS assignment_id, r.node_id AS rhs_id, r.type AS rhs_type,
                 r.children_count AS rhs_count
          FROM cuda_nodes() a JOIN cuda_nodes() r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
          WHERE a.type = 'assignment'
          QUALIFY row_number() OVER (PARTITION BY a.fact_id, a.node_id
                                     ORDER BY r.sibling_index DESC) = 1
        ), placed AS (
          SELECT o.*, x.rhs_id, x.rhs_type,
                 o.nested AND x.rhs_type IN ('expression_list', 'tuple')
                   AND x.rhs_count = o.holder_count AS matched
          FROM outer_owners o
          LEFT JOIN assigned x ON x.fact_id = o.fact_id AND x.assignment_id = o.final_id
        )
        SELECT o.fact_id, o.node_id, o.name, o.scope_id,
               CASE WHEN o.final_owner = 'assignment' AND (o.matched OR (NOT o.nested AND o.sibling_index = 0))
                    THEN 'assignment'
                    WHEN o.final_owner IN ('for_statement', 'for_in_clause') AND NOT o.nested
                    THEN 'loop'
                    WHEN o.nested THEN 'unpack' ELSE 'other' END AS kind,
               o.final_id AS statement_id,
               CASE WHEN o.final_owner = 'assignment' AND o.matched
                    THEN (SELECT e.node_id FROM cuda_nodes() e
                          WHERE e.fact_id = o.fact_id AND e.parent_id = o.rhs_id
                            AND e.sibling_index = o.sibling_index)
                    WHEN o.final_owner = 'assignment' AND o.rhs_type <> 'type' THEN o.rhs_id
                    WHEN o.final_owner IN ('for_statement', 'for_in_clause') AND NOT o.nested
                    THEN (SELECT r.node_id FROM cuda_nodes() r
                          WHERE r.fact_id = o.fact_id AND r.parent_id = o.final_id
                            AND r.sibling_index = 3) END AS source_id,
               CASE WHEN o.nested AND NOT o.matched THEN o.sibling_index END AS position
        FROM placed o
        WHERE (o.final_owner = 'assignment' AND (o.nested OR o.sibling_index = 0))
           OR (o.final_owner IN ('for_statement', 'for_in_clause')
               AND (o.nested OR o.sibling_index = 1))
           OR (o.final_owner IN ('augmented_assignment', 'named_expression', 'as_pattern_target')
               AND o.sibling_index = 0)
           OR o.final_owner = 'parameters'
           OR (o.final_owner IN ('typed_parameter', 'default_parameter',
                                 'typed_default_parameter', 'list_splat_pattern',
                                 'dictionary_splat_pattern') AND o.sibling_index = 0)
    """


class CudaScopes(CudaRelation):
    """The scopes a function reads names through: itself (`depth` 0), the function around it, and
    so on to the module (`-1`)."""

    stored = True
    query = """
        WITH RECURSIVE chain(fact_id, scope_id, visible_id, depth) AS (
          SELECT fact_id, node_id, node_id, 0 FROM cuda_definitions() WHERE kind = 'function'
          UNION ALL
          SELECT c.fact_id, c.scope_id, coalesce(d.outer_id, -1), c.depth + 1
          FROM chain c JOIN cuda_definitions() d ON d.fact_id = c.fact_id
            AND d.node_id = c.visible_id AND d.kind = 'function'
          WHERE c.visible_id <> -1 AND c.depth < 6
        )
        SELECT * FROM chain
        UNION ALL
        SELECT fact_id, -1, -1, 0 FROM cuda_modules()
    """


class CudaIdentifierReads(CudaRelation):
    """Each identifier that reads a plain name: not an attribute's member, a keyword's name, a
    definition's or import's name, nor a binding."""

    stored = True
    query = """
        SELECT i.fact_id, i.node_id, i.name, coalesce(i.scope."function", -1) AS scope_id
        FROM cuda_nodes() i JOIN cuda_nodes() p ON p.fact_id = i.fact_id AND p.node_id = i.parent_id
        WHERE i.type = 'identifier'
          AND NOT (p.type = 'attribute' AND i.sibling_index > 0)
          AND NOT (p.type = 'keyword_argument' AND i.sibling_index = 0)
          AND p.type NOT IN ('function_definition', 'class_definition', 'dotted_name',
                             'aliased_import', 'import_from_statement', 'import_statement',
                             'relative_import', 'global_statement', 'nonlocal_statement')
          AND NOT EXISTS (SELECT 1 FROM cuda_stores() s
                          WHERE s.fact_id = i.fact_id AND s.node_id = i.node_id)
    """


class CudaNameUses(CudaRelation):
    """Each read of a plain name that a store in the same function, an enclosing one or the module
    binds: the identifier and the variable (`v:<fact>:<scope>:<name>`) it reads."""

    stored = True
    query = """
        WITH bound AS (
          SELECT DISTINCT fact_id, scope_id, name FROM cuda_stores()
        )
        SELECT r.fact_id, r.node_id, r.name,
               'v:' || r.fact_id || ':' || b.scope_id || ':' || r.name AS variable
        FROM cuda_identifier_reads() r
        JOIN cuda_scopes() s ON s.fact_id = r.fact_id AND s.scope_id = r.scope_id
        JOIN bound b ON b.fact_id = r.fact_id AND b.scope_id = s.visible_id AND b.name = r.name
        QUALIFY row_number() OVER (PARTITION BY r.fact_id, r.node_id ORDER BY s.depth) = 1
    """


class CudaArguments(CudaRelation):
    """The arguments of every call, in order: a positional one has its `position` (from 0), a
    keyword one its `keyword`; `value_id` is the node of the value. Calls with a `*args` or
    `**kwargs` argument before the argument have no positions after it; keywords always count."""

    stored = True
    query = """
        WITH written AS (
          SELECT c.fact_id, c.node_id AS call_id, a.node_id, a.sibling_index, a.type,
                 row_number() OVER (PARTITION BY c.fact_id, c.node_id ORDER BY a.sibling_index) - 1
                   AS slot
          FROM cuda_calls() c
          JOIN cuda_nodes() a ON a.fact_id = c.fact_id AND a.parent_id = c.arguments_id
          WHERE a.type NOT IN ('(', ')', ',', 'comment')
        ), splatted AS (
          SELECT fact_id, call_id, min(slot) AS first_splat FROM written
          WHERE type IN ('list_splat', 'dictionary_splat') GROUP BY ALL
        )
        SELECT w.fact_id, w.call_id,
               CASE WHEN w.type = 'keyword_argument' THEN NULL ELSE w.slot END AS position,
               CASE WHEN w.type = 'keyword_argument' THEN k.name END AS keyword,
               coalesce(v.node_id, w.node_id) AS value_id
        FROM written w
        LEFT JOIN cuda_nodes() k ON k.fact_id = w.fact_id AND k.parent_id = w.node_id
          AND k.sibling_index = 0 AND w.type = 'keyword_argument'
        LEFT JOIN cuda_nodes() v ON v.fact_id = w.fact_id AND v.parent_id = w.node_id
          AND v.sibling_index = 2 AND w.type = 'keyword_argument'
        LEFT JOIN splatted s ON s.fact_id = w.fact_id AND s.call_id = w.call_id
        WHERE w.type NOT IN ('list_splat', 'dictionary_splat')
          AND (w.type = 'keyword_argument' OR s.first_splat IS NULL OR w.slot < s.first_splat)
    """
