"""How much the threads of a block agree on a value, and where a barrier is exposed to it.

A value has a `level`: 0 when every thread of the block holds the same (a scalar parameter, a
block index, a constant), 1 when the threads of one warp agree but the warps may not (`threadIdx.x
// 32`, the result of a warp vote), 2 when the lanes of a warp may differ (`threadIdx.x`,
`cuda.grid()`, `lane()`). A block barrier needs a condition of level 0 around it, a warp
collective a level of at most 1.

Levels start at atoms (thread indices, collectives with a uniform result) and flow, to a fixpoint,
along stores (each store is its own key, and a use reads the stores that can reach it), function
returns, arguments to parameters and, for a store made under a branch, the branch's condition.
Keys name what carries a level: `s:<fact>:<node>` the value one store binds, `r:<fact>:<function
node>` what a function returns (`#<position>` one element of a tuple or record, `.<field>` one
field of a record), `b:<fact>:<node>` a branch's condition. A field read off a record a call just
built takes the level of that field alone (`cuda_field_reads`).
"""

from ...constants.cuda_base import NUMBA_CUDA
from ..base import CudaRelation


class CudaLevelStores(CudaRelation):
    """The stores of device code, each a key of its own: the `variable` it binds, `done` the node
    after which its value exists (the end of its statement), and whether it `kills` earlier
    stores (a plain `name = value` statement of a block, which every later use in the block
    sees) with the block's span."""

    stored = True
    query = """
        SELECT s.fact_id, s.node_id, s.name, s.scope_id, s.kind,
               's:' || s.fact_id || ':' || s.node_id AS key,
               'v:' || s.fact_id || ':' || s.scope_id || ':' || s.name AS variable,
               CASE WHEN s.kind IN ('assignment', 'unpack') THEN a.node_id + a.descendant_count
                    ELSE s.node_id END AS done,
               coalesce(s.kind IN ('assignment', 'unpack') AND es.type = 'expression_statement'
                        AND bl.type = 'block', false) AS kills,
               bl.node_id AS block_first, bl.node_id + bl.descendant_count AS block_last,
               a.type = 'augmented_assignment' AS augmented,
               a.type IN ('parameters', 'typed_parameter', 'default_parameter',
                          'typed_default_parameter', 'list_splat_pattern',
                          'dictionary_splat_pattern') AS parameter
        FROM cuda_stores() s
        JOIN cuda_nodes() a ON a.fact_id = s.fact_id AND a.node_id = s.statement_id
        LEFT JOIN cuda_nodes() es ON es.fact_id = a.fact_id AND es.node_id = a.parent_id
        LEFT JOIN cuda_nodes() bl ON bl.fact_id = es.fact_id AND bl.node_id = es.parent_id
        WHERE EXISTS (SELECT 1 FROM cuda_device_code() d
                      WHERE d.fact_id = s.fact_id AND d.function_id = s.scope_id)
    """


class CudaReachingStores(CudaRelation):
    """For each use of a variable in device code, the stores whose value it can read: in its own
    function the stores before it that no later unconditional store replaced (a plain assignment
    that is a statement of a block holding the use, or the earlier store), and, in a loop, the
    stores later in that loop; across functions every store."""

    stored = True
    query = """
        WITH uses AS (
          SELECT u.fact_id, u.node_id, u.variable, coalesce(n.scope."function", -1) AS scope_id
          FROM cuda_name_uses() u
          JOIN cuda_nodes() n ON n.fact_id = u.fact_id AND n.node_id = u.node_id
          WHERE EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = n.fact_id AND d.function_id = n.scope."function")
          UNION ALL
          SELECT fact_id, node_id, variable, scope_id FROM cuda_level_stores() WHERE augmented
        ), loops AS (
          SELECT fact_id, node_id AS first, node_id + descendant_count AS last
          FROM cuda_nodes() WHERE type IN ('for_statement', 'while_statement')
        )
        SELECT DISTINCT u.fact_id, u.node_id AS use_id, s.key
        FROM uses u
        JOIN cuda_level_stores() s ON s.fact_id = u.fact_id AND s.variable = u.variable
        WHERE s.scope_id <> u.scope_id
           OR (s.done < u.node_id AND NOT EXISTS (
                 SELECT 1 FROM cuda_level_stores() k
                 WHERE k.fact_id = u.fact_id AND k.variable = u.variable AND k.kills
                   AND k.node_id > s.node_id AND k.done < u.node_id
                   AND (u.node_id BETWEEN k.block_first AND k.block_last
                        OR s.node_id BETWEEN k.block_first AND k.block_last)))
           OR (s.node_id >= u.node_id AND EXISTS (
                 SELECT 1 FROM loops l
                 WHERE l.fact_id = u.fact_id AND u.node_id BETWEEN l.first AND l.last
                   AND s.node_id BETWEEN l.first AND l.last
                   AND NOT EXISTS (
                     SELECT 1 FROM cuda_level_stores() k
                     WHERE k.fact_id = u.fact_id AND k.variable = u.variable AND k.kills
                       AND k.node_id > l.first AND k.done < u.node_id
                       AND u.node_id BETWEEN k.block_first AND k.block_last)))
    """


class CudaConstantValues(CudaRelation):
    """The integer value of each right operand of an arithmetic or comparison operator that is a
    literal, a name the file binds once to a literal at module or class level (`WARP_SIZE = 32`,
    `WARP_BITS = 5`), or a name or attribute spelling the warp size (`WARP_SIZE`,
    `cuda.warpsize`: 32)."""

    stored = True
    query = r"""
        WITH defined AS (
          SELECT a.fact_id, l.name, try_cast(v.name AS BIGINT) AS value
          FROM cuda_nodes() a
          JOIN cuda_nodes() l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'identifier'
          JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
            AND v.type = 'integer'
          WHERE a.type = 'assignment' AND a.scope."function" IS NULL
        ), constants AS (
          SELECT fact_id, name, min(value) AS value FROM defined
          GROUP BY fact_id, name HAVING min(value) = max(value)
        )
        SELECT k.fact_id, k.node_id, k.parent_id, k.sibling_index,
               CASE WHEN k.type = 'integer' THEN try_cast(k.name AS BIGINT)
                    WHEN c.value IS NOT NULL THEN c.value
                    ELSE 32 END AS value
        FROM cuda_nodes() k
        JOIN cuda_nodes() p ON p.fact_id = k.fact_id AND p.node_id = k.parent_id
        LEFT JOIN constants c ON c.fact_id = k.fact_id AND c.name = k.name
        WHERE p.type IN ('binary_operator', 'comparison_operator')
          AND (k.type = 'integer'
               OR (k.type IN ('identifier', 'attribute')
                   AND (c.value IS NOT NULL
                        OR regexp_matches(k.name, '(?i)warp_?size|WARP$'))))
    """


class CudaLaneOffsets(CudaRelation):
    """The expression `tid - tid % 32` (or any multiple of the warp size), which every lane of a
    warp computes to the same first lane of the warp: `whole_id` is its left operand and `part_id`
    the one inside the remainder, both the same name or attribute."""

    stored = True
    query = """
        SELECT d.fact_id, w.node_id AS whole_id, q.node_id AS part_id
        FROM cuda_nodes() d
        JOIN cuda_nodes() op ON op.fact_id = d.fact_id AND op.parent_id = d.node_id
          AND op.sibling_index = 1 AND op.type = '-'
        JOIN cuda_nodes() w ON w.fact_id = d.fact_id AND w.parent_id = d.node_id
          AND w.sibling_index = 0 AND w.type IN ('identifier', 'attribute')
        JOIN cuda_nodes() m ON m.fact_id = d.fact_id AND m.parent_id = d.node_id
          AND m.sibling_index = 2 AND m.type = 'binary_operator'
        JOIN cuda_nodes() mop ON mop.fact_id = d.fact_id AND mop.parent_id = m.node_id
          AND mop.sibling_index = 1 AND mop.type = '%'
        JOIN cuda_nodes() q ON q.fact_id = d.fact_id AND q.parent_id = m.node_id
          AND q.sibling_index = 0 AND q.type = w.type AND q.name = w.name
        JOIN cuda_constant_values() k ON k.fact_id = d.fact_id AND k.parent_id = m.node_id
          AND k.sibling_index = 2
        WHERE d.type = 'binary_operator' AND k.value >= 32 AND k.value % 32 = 0
    """


class CudaLevelAtoms(CudaRelation):
    """What gives an expression its level, by node: a thread index (`level` 2 or 1, no `ref`), a
    collective whose result every thread shares (its `level`, hiding the nodes of its arguments
    from `mask_first` to `mask_last`), a use of a variable (`ref` a store that reaches it) or a
    call of a project function (`ref` its return, hiding the arguments, which reach it as its
    parameters). `lowers` marks an atom that a division by 32, a shift by 5 or a comparison with
    a multiple of 32 makes warp-wide: `threadIdx.x // 32`, `threadIdx.x < 32`."""

    stored = True
    query = rf"""
        WITH device_nodes AS (
          SELECT n.* FROM cuda_nodes() n
          WHERE EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = n.fact_id AND d.function_id = n.scope."function")
        ), thread_fields AS (
          SELECT a.fact_id, a.node_id,
                 CASE WHEN regexp_matches(nm.qualified || substr(a.name, length(split_part(a.name, '.', 1)) + 1),
                        '^{NUMBA_CUDA}\.(threadIdx(\.[xyz])?|laneid)$') THEN 2 END AS level,
                 a.name AS label
          FROM device_nodes a
          JOIN cuda_names() nm ON nm.fact_id = a.fact_id AND nm.bound = split_part(a.name, '.', 1)
          JOIN cuda_nodes() p ON p.fact_id = a.fact_id AND p.node_id = a.parent_id
          WHERE a.type = 'attribute' AND NOT (p.type = 'attribute' AND a.sibling_index = 0)
        ), identity_calls AS (
          SELECT c.fact_id, c.node_id,
                 CASE WHEN regexp_matches(c.qualified_name, '^{NUMBA_CUDA}\.grid$')
                        OR regexp_matches(c.qualified_name,
                             '^patos\.cuda\.typed(\.\w+)*\.(thread_index|thread_in_block|lane)$')
                      THEN 2
                      WHEN regexp_matches(c.qualified_name,
                             '^patos\.cuda\.typed(\.\w+)*\.(warp_index|warp_in_block)$')
                      THEN 1
                      WHEN regexp_matches(c.qualified_name, '^patos\.cuda\.typed(\.\w+)*\.items(_through)?$')
                      THEN CASE coalesce(k.per, 'THREAD') WHEN 'BLOCK' THEN 0 WHEN 'WARP' THEN 1
                                                          ELSE 2 END END AS level,
                 c.callee_text || '()' AS label
          FROM cuda_calls() c
          LEFT JOIN cuda_kernel_items() k ON k.fact_id = c.fact_id AND k.function_id = c.scope_id
          WHERE EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = c.fact_id AND d.function_id = c.scope_id)
        ), raw AS (
          SELECT fact_id, node_id, level, CAST(NULL AS VARCHAR) AS ref, label,
                 CAST(NULL AS BIGINT) AS mask_first, CAST(NULL AS BIGINT) AS mask_last
          FROM thread_fields WHERE level IS NOT NULL
          UNION ALL
          SELECT fact_id, node_id, level, NULL, label, NULL, NULL
          FROM identity_calls WHERE level IS NOT NULL
          UNION ALL
          SELECT r.fact_id, r.use_id, 0, r.key, n.name, NULL, NULL
          FROM cuda_reaching_stores() r
          JOIN cuda_nodes() n ON n.fact_id = r.fact_id AND n.node_id = r.use_id
          UNION ALL
          SELECT c.fact_id, c.node_id, 0, 'r:' || t.definition_fact || ':' || t.definition_id,
                 c.callee_text || '()', NULL, NULL
          FROM cuda_calls() c
          JOIN cuda_resolved_calls() t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
            AND t.kind = 'function' AND t.via <> 'override'
          WHERE EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = c.fact_id AND d.function_id = c.scope_id)
            AND NOT EXISTS (SELECT 1 FROM cuda_collective_calls() k
                            WHERE k.fact_id = c.fact_id AND k.node_id = c.node_id
                              AND k.result_level IS NOT NULL)
          UNION ALL
          SELECT k.fact_id, k.node_id, k.result_level, NULL, k.shown || ' (a vote or reduction)',
                 a.node_id, a.node_id + a.descendant_count
          FROM cuda_collective_calls() k
          JOIN cuda_nodes() a ON a.fact_id = k.fact_id AND a.node_id = k.arguments_id
          WHERE k.result_level IS NOT NULL
          UNION ALL
          SELECT fact_id, node_id, level, ref, label, mask_first, mask_last
          FROM cuda_field_reads()
        )
        SELECT r.*,
               coalesce(
                 (p.type = 'binary_operator' AND n.sibling_index = 0
                  AND EXISTS (SELECT 1 FROM cuda_nodes() o
                              JOIN cuda_constant_values() k ON k.fact_id = o.fact_id
                                AND k.parent_id = o.parent_id AND k.sibling_index = 2
                              WHERE o.fact_id = p.fact_id AND o.parent_id = p.node_id
                                AND o.sibling_index = 1
                                AND ((o.type = '//' AND k.value >= 32 AND k.value % 32 = 0)
                                     OR (o.type = '>>' AND k.value >= 5))))
                 OR (p.type = 'comparison_operator' AND p.children_count = 3
                     AND n.sibling_index = 0
                     AND EXISTS (SELECT 1 FROM cuda_nodes() o
                                 JOIN cuda_constant_values() k ON k.fact_id = o.fact_id
                                   AND k.parent_id = o.parent_id AND k.sibling_index = 2
                                 WHERE o.fact_id = p.fact_id AND o.parent_id = p.node_id
                                   AND o.sibling_index = 1 AND o.type IN ('<', '>=')
                                   AND k.value >= 32 AND k.value % 32 = 0))
                 OR EXISTS (SELECT 1 FROM cuda_lane_offsets() w
                            WHERE w.fact_id = n.fact_id AND n.node_id IN (w.whole_id, w.part_id)),
                 false) AS lowers
        FROM raw r
        JOIN cuda_nodes() n ON n.fact_id = r.fact_id AND n.node_id = r.node_id
        LEFT JOIN cuda_nodes() p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
    """


class CudaParamSites(CudaRelation):
    """Each argument of a call of a device function, as the `site` that binds the parameter it
    reaches (`key`, a store), the argument's nodes `first`..`last`, and how many `sites` bind
    that parameter in all."""

    stored = True
    query = """
        WITH bound AS (
          SELECT DISTINCT a.fact_id,
                 'a:' || a.fact_id || ':' || c.node_id || ':' || ps.node_id AS site, ps.key,
                 x.node_id AS first, x.node_id + x.descendant_count AS last
          FROM cuda_resolved_calls() c
          JOIN cuda_definitions() d ON d.fact_id = c.definition_fact AND d.node_id = c.definition_id
            AND d.kind = 'function'
          JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
          JOIN cuda_nodes() x ON x.fact_id = a.fact_id AND x.node_id = a.value_id
          JOIN cuda_parameters() p ON p.fact_id = d.fact_id AND p.function_id = d.node_id
            AND (p.name = a.keyword
                 OR p.position = a.position + 1
                    + CASE WHEN d.class_id IS NOT NULL AND d.outer_id IS NULL
                            AND NOT EXISTS (SELECT 1 FROM cuda_decorators() z
                                            WHERE z.fact_id = d.fact_id AND z.definition_id = d.node_id
                                              AND z.written = 'staticmethod')
                           THEN 1 ELSE 0 END)
          JOIN cuda_level_stores() ps ON ps.fact_id = d.fact_id AND ps.scope_id = d.node_id
            AND ps.name = p.name AND ps.parameter
          WHERE c.kind = 'function' AND c.via <> 'override'
        )
        SELECT *, count(*) OVER (PARTITION BY key) AS sites FROM bound
    """


class CudaReturnElements(CudaRelation):
    """The values a function returns one by one when it returns a tuple (`return a, b`) or builds
    a record from arguments (`return Pair(a, b)`): `key` is `r:<fact>:<function node>#<position>`."""

    stored = True
    query = """
        WITH returns AS (
          SELECT r.fact_id, r.node_id, r.scope."function" AS function_id, e.node_id AS value_id,
                 e.type
          FROM cuda_nodes() r
          JOIN cuda_nodes() e ON e.fact_id = r.fact_id AND e.parent_id = r.node_id
            AND e.sibling_index = 1
          WHERE r.type = 'return_statement' AND r.scope."function" IS NOT NULL
            AND EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = r.fact_id AND d.function_id = r.scope."function")
        )
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.function_id || '#'
                          || (row_number() OVER (PARTITION BY r.fact_id, r.node_id
                                                 ORDER BY x.sibling_index) - 1) AS key,
               x.node_id AS first, x.node_id + x.descendant_count AS last
        FROM returns r
        JOIN cuda_nodes() x ON x.fact_id = r.fact_id AND x.parent_id = r.value_id
        WHERE r.type IN ('tuple', 'expression_list') AND x.type NOT IN ('(', ')', ',', 'comment')
        UNION ALL
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.function_id || '#' || a.position,
               x.node_id, x.node_id + x.descendant_count
        FROM returns r
        JOIN cuda_arguments() a ON a.fact_id = r.fact_id AND a.call_id = r.value_id
          AND a.position IS NOT NULL
        JOIN cuda_nodes() x ON x.fact_id = a.fact_id AND x.node_id = a.value_id
        WHERE r.type = 'call'
        UNION ALL
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.function_id || '.' || coalesce(f.field, a.keyword),
               x.node_id, x.node_id + x.descendant_count
        FROM returns r
        JOIN cuda_calls() c ON c.fact_id = r.fact_id AND c.node_id = r.value_id
        LEFT JOIN cuda_call_targets() t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
          AND t.kind = 'class'
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
        JOIN cuda_nodes() x ON x.fact_id = a.fact_id AND x.node_id = a.value_id
        LEFT JOIN cuda_record_fields() f ON f.record = coalesce(t.key, c.qualified_name)
          AND f.position = a.position
        WHERE r.type = 'call' AND coalesce(f.field, a.keyword) IS NOT NULL
    """


class CudaRecordFields(CudaRelation):
    """The fields of the records a call can build, by `record` (the key of a class the project
    defines, else the dotted name of one a library does) and the `position` a positional argument
    fills: the annotated fields of the class in the order its body writes them."""

    stored = True
    query = """
        SELECT o.class_key AS record, f.name AS field, o.position
        FROM cuda_field_order() o JOIN cuda_fields() f ON f.field = o.field
        UNION
        SELECT record, field, position FROM cuda_library_results()
    """


class CudaFieldReads(CudaRelation):
    """The attributes read off a record that a call has just built, as the atoms they are:
    `warps.total` after `warps = self._prefixed(x)` takes the level of what `_prefixed` returns as
    `total` (`ref`), and `emitted.before` after `emitted = warp.ballot_prefix(f)` the library's
    level of that field (`level`), not the level of the whole record, which is the highest of its
    fields. A read is listed only when every store its name can see is such a call, so
    it hides the nodes of its object (`mask_first`..`mask_last`) and everything else reads the
    whole record."""

    stored = True
    query = """
        WITH reads AS (
          SELECT x.fact_id, x.node_id, x.name AS label, o.node_id AS object_id,
                 o.node_id + o.descendant_count AS object_last, o.type AS object_type, f.name AS field
          FROM cuda_nodes() x
          JOIN cuda_nodes() o ON o.fact_id = x.fact_id AND o.parent_id = x.node_id
            AND o.sibling_index = 0
          JOIN cuda_nodes() f ON f.fact_id = x.fact_id AND f.parent_id = x.node_id
            AND f.sibling_index = 1
          WHERE x.type = 'attribute' AND o.type IN ('identifier', 'call')
            AND EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = x.fact_id AND d.function_id = x.scope."function")
        ), sources AS (
          SELECT r.*, st.source_id AS call_id, st.kind = 'assignment' AS assigned
          FROM reads r
          JOIN cuda_reaching_stores() g ON g.fact_id = r.fact_id AND g.use_id = r.object_id
          JOIN cuda_level_stores() ls ON ls.key = g.key
          JOIN cuda_stores() st ON st.fact_id = ls.fact_id AND st.node_id = ls.node_id
          WHERE r.object_type = 'identifier'
          UNION ALL
          SELECT r.*, r.object_id, true FROM reads r WHERE r.object_type = 'call'
        ), told AS (
          SELECT s.*, l.level,
                 'r:' || t.definition_fact || ':' || t.definition_id || '.' || s.field AS ref
          FROM sources s
          LEFT JOIN cuda_calls() c ON c.fact_id = s.fact_id AND c.node_id = s.call_id AND s.assigned
          LEFT JOIN cuda_library_results() l ON l.function = c.qualified_name AND l.field = s.field
          LEFT JOIN cuda_resolved_calls() t ON t.fact_id = s.fact_id AND t.node_id = s.call_id
            AND s.assigned AND t.kind = 'function' AND t.via <> 'override'
        ), told_known AS (
          SELECT t.*, t.level IS NOT NULL
                      OR EXISTS (SELECT 1 FROM cuda_return_elements() e WHERE e.key = t.ref) AS known
          FROM told t
        ), every AS (
          SELECT *, bool_and(known) OVER (PARTITION BY fact_id, node_id) AS all_known
          FROM told_known
        )
        SELECT DISTINCT fact_id, node_id, coalesce(level, 0) AS level,
               CASE WHEN level IS NULL THEN ref END AS ref, label, object_id AS mask_first,
               object_last AS mask_last
        FROM every
        WHERE all_known
    """


class CudaLevelLinks(CudaRelation):
    """A name bound by unpacking a call (`before, total = self._prefixed(x)`) takes the level of
    the returned element at its position (`ref`), not of the whole tuple, where the callee
    returns its elements one by one."""

    stored = True
    query = """
        SELECT s.key, 'r:' || t.definition_fact || ':' || t.definition_id || '#' || o.position AS ref,
               s.name AS label
        FROM cuda_level_stores() s
        JOIN cuda_stores() o ON o.fact_id = s.fact_id AND o.node_id = s.node_id
        JOIN cuda_calls() c ON c.fact_id = o.fact_id AND c.node_id = o.source_id
        JOIN cuda_resolved_calls() t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
          AND t.kind = 'function' AND t.via <> 'override'
        WHERE s.kind = 'unpack' AND o.position IS NOT NULL
          AND EXISTS (SELECT 1 FROM cuda_return_elements() e
                      WHERE e.key = 'r:' || t.definition_fact || ':' || t.definition_id
                                    || '#' || o.position)
    """


class CudaLevelSinks(CudaRelation):
    """Where a level lands: `key` takes the level of the atoms in `first`..`last` (nodes of
    `fact_id`). A store takes its value and, under a branch, the branch's condition; a function
    its returned values (and each element it returns one by one); an argument its expression
    (the parameter takes the lowest level of its arguments); a branch its condition."""

    stored = True
    query = """
        SELECT s.fact_id, s.key, x.node_id AS first, x.node_id + x.descendant_count AS last
        FROM cuda_level_stores() s
        JOIN cuda_stores() o ON o.fact_id = s.fact_id AND o.node_id = s.node_id
        JOIN cuda_nodes() x ON x.fact_id = o.fact_id AND x.node_id = o.source_id
        WHERE s.kind IN ('assignment', 'loop', 'unpack')
          AND NOT EXISTS (SELECT 1 FROM cuda_level_links() l WHERE l.key = s.key)
        UNION ALL
        SELECT fact_id, key, first, last FROM cuda_return_elements()
        UNION ALL
        SELECT s.fact_id, s.key, r.node_id, r.node_id + r.descendant_count
        FROM cuda_level_stores() s
        JOIN cuda_nodes() a ON a.fact_id = s.fact_id AND a.node_id = (
          SELECT q.parent_id FROM cuda_nodes() q WHERE q.fact_id = s.fact_id AND q.node_id = s.node_id)
        JOIN cuda_nodes() r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
          AND r.sibling_index = 2
        WHERE s.augmented AND a.type = 'augmented_assignment'
        UNION ALL
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.scope."function", e.node_id,
               e.node_id + e.descendant_count
        FROM cuda_nodes() r
        JOIN cuda_nodes() e ON e.fact_id = r.fact_id AND e.parent_id = r.node_id
          AND e.sibling_index = 1
        WHERE r.type = 'return_statement' AND r.scope."function" IS NOT NULL
          AND EXISTS (SELECT 1 FROM cuda_device_code() d
                      WHERE d.fact_id = r.fact_id AND d.function_id = r.scope."function")
        UNION ALL
        SELECT fact_id, site, first, last FROM cuda_param_sites()
        UNION ALL
        SELECT s.fact_id, s.key, b.cond_first, b.cond_last
        FROM cuda_level_stores() s
        JOIN cuda_branches() b ON b.fact_id = s.fact_id AND b.scope_id = s.scope_id
          AND s.node_id BETWEEN b.first AND b.last
          AND NOT s.node_id BETWEEN b.cond_first AND b.cond_last
        WHERE NOT s.parameter
        UNION ALL
        SELECT fact_id, 'b:' || fact_id || ':' || branch_id, cond_first, cond_last
        FROM cuda_branches()
    """


class CudaLevelFlows(CudaRelation):
    """Each atom a sink's expression reads that a collective's or call's result does not hide
    (`label`, its `level` or the `ref` it takes, `lowers` as in the atoms)."""

    stored = True
    query = """
        SELECT s.key, a.level, a.ref, a.lowers, a.label
        FROM cuda_level_sinks() s
        JOIN cuda_level_atoms() a ON a.fact_id = s.fact_id AND a.node_id BETWEEN s.first AND s.last
        WHERE NOT EXISTS (
          SELECT 1 FROM cuda_level_atoms() m
          WHERE m.fact_id = a.fact_id AND m.mask_first IS NOT NULL AND m.node_id <> a.node_id
            AND m.node_id BETWEEN s.first AND s.last
            AND a.node_id BETWEEN m.mask_first AND m.mask_last)
        UNION ALL
        SELECT key, 0, ref, false, label FROM cuda_level_links()
    """


class CudaLevels(CudaRelation):
    """The level, 1 or 2, of every key the atoms reach, with `why` the atom that gave it. A
    parameter takes the lowest level among the arguments of all the calls that bind it, so one
    call with a uniform argument keeps it uniform."""

    stored = True
    query = """
        WITH RECURSIVE lv(key, level, why) USING KEY (key) AS (
          SELECT key, max(CASE WHEN lowers THEN least(level, 1) ELSE level END),
                 arg_max(label, (CASE WHEN lowers THEN least(level, 1) ELSE level END, label))
          FROM cuda_level_flows() WHERE ref IS NULL
          GROUP BY key
          HAVING max(CASE WHEN lowers THEN least(level, 1) ELSE level END) > 0
          UNION
          (SELECT f.key,
                  max(CASE WHEN f.lowers THEN least(d.level, 1) ELSE d.level END),
                  arg_max(d.why, (CASE WHEN f.lowers THEN least(d.level, 1) ELSE d.level END, d.why))
           FROM lv d JOIN cuda_level_flows() f ON f.ref = d.key
           GROUP BY f.key
           HAVING max(CASE WHEN f.lowers THEN least(d.level, 1) ELSE d.level END)
                  > coalesce((SELECT x.level FROM recurring.lv x WHERE x.key = f.key), 0)
           UNION ALL
           SELECT p.key, min(x.level), arg_max(x.why, (x.level, x.why))
           FROM (SELECT DISTINCT q.key FROM lv d JOIN cuda_param_sites() q ON q.site = d.key) k
           JOIN cuda_param_sites() p ON p.key = k.key
           JOIN recurring.lv x ON x.key = p.site
           GROUP BY p.key, p.sites
           HAVING count(*) = p.sites
              AND min(x.level) > coalesce((SELECT y.level FROM recurring.lv y WHERE y.key = p.key), 0))
        )
        SELECT * FROM lv
    """
