"""How much the threads of a block agree on a value, and the atoms its level starts from.

A value has a `level`: 0 when every thread of the block holds the same (a scalar parameter, a
block index, a constant), 1 when the threads of one warp agree but the warps may not (`threadIdx.x
// 32`, the result of a warp vote), 2 when the lanes of a warp may differ (`threadIdx.x`,
`cuda.grid()`, `lane()`). A block barrier needs a condition of level 0 around it, a warp
collective a level of at most 1.

Levels start at atoms (thread indices, collectives with a uniform result), which this module
finds with the stores they read and the sites that bind arguments to parameters. `flows.py`
carries them from there.
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
