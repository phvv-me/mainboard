"""Where code allocates device memory, and which files deal with the device at all."""

from .cuda_base import CudaRelation


class CudaDeviceFiles(CudaRelation):
    """The files that handle device memory: they import a GPU framework, launch a kernel or hold
    a device value."""

    stored = True
    query = """
        SELECT DISTINCT fact_id FROM cuda_names()
        WHERE regexp_matches(qualified, '^(cupyx?|numba\\.cuda|torch\\.cuda|cuda\\.|triton|helion|warp|patos\\.cuda)')
        UNION
        SELECT DISTINCT fact_id FROM cuda_launches()
        UNION
        SELECT DISTINCT fact_id FROM cuda_device_values()
    """


class CudaCapacityGuarded(CudaRelation):
    """The calls that store a buffer in the body of an `if` or `elif` that lets the allocation
    happen once, or once per key: a size compared with the buffer's own capacity (`if need >
    self.buffer.shape[0]: self.buffer = make(need)`), a buffer that `is None` (`if self.buf is
    None: self.buf = make()`), a key `not in` the dict that stores it (`if n not in self.cache:
    self.cache[n] = make(n)`), or a comparison whose operand the body then assigns (`if n >
    self.capacity: ...; self.capacity = n`)."""

    stored = True
    query = r"""
        WITH guards AS (
          SELECT DISTINCT i.fact_id, blk.node_id AS first, blk.node_id + blk.descendant_count AS last,
                 syntax_slice(s.content, cmp.start_line, cmp.start_column, cmp.end_line,
                              cmp.end_column) AS compared,
                 EXISTS (SELECT 1 FROM cuda_nodes() op
                         WHERE op.fact_id = cmp.fact_id AND op.parent_id = cmp.node_id
                           AND op.type IN ('<', '<=', '>', '>=')) AS ordered,
                 syntax_slice(s.content, l.start_line, l.start_column, l.end_line, l.end_column)
                   AS left_side,
                 syntax_slice(s.content, r.start_line, r.start_column, r.end_line, r.end_column)
                   AS right_side
          FROM cuda_nodes() i
          JOIN cuda_nodes() cond ON cond.fact_id = i.fact_id AND cond.parent_id = i.node_id
            AND cond.sibling_index = 1
          JOIN cuda_nodes() blk ON blk.fact_id = i.fact_id AND blk.parent_id = i.node_id
            AND blk.type = 'block' AND blk.sibling_index = (
              SELECT min(b.sibling_index) FROM cuda_nodes() b
              WHERE b.fact_id = i.fact_id AND b.parent_id = i.node_id AND b.type = 'block')
          JOIN cuda_nodes() cmp ON cmp.fact_id = i.fact_id AND cmp.type = 'comparison_operator'
            AND cmp.node_id BETWEEN cond.node_id AND cond.node_id + cond.descendant_count
          LEFT JOIN cuda_nodes() l ON l.fact_id = cmp.fact_id AND l.parent_id = cmp.node_id
            AND l.sibling_index = 0 AND cmp.children_count = 3
          LEFT JOIN cuda_nodes() r ON r.fact_id = cmp.fact_id AND r.parent_id = cmp.node_id
            AND r.sibling_index = 2 AND cmp.children_count = 3
          JOIN source_files s ON s.path = i.path
          WHERE i.type IN ('if_statement', 'elif_clause')
        )
        SELECT DISTINCT c.fact_id, c.node_id
        FROM guards g
        JOIN cuda_calls() c ON c.fact_id = g.fact_id AND c.node_id BETWEEN g.first AND g.last
        JOIN cuda_nodes() a ON a.fact_id = c.fact_id AND a.type = 'assignment'
          AND c.node_id BETWEEN a.node_id + 1 AND a.node_id + a.descendant_count
          AND a.node_id BETWEEN g.first AND g.last
        JOIN cuda_nodes() t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id
          AND t.sibling_index = 0
        JOIN source_files s ON s.path = c.path
        CROSS JOIN LATERAL (
          SELECT syntax_slice(s.content, t.start_line, t.start_column, t.end_line, t.end_column)
                   AS target
        ) x
        WHERE (g.ordered AND contains(g.compared, x.target)
               AND regexp_matches(g.compared, '(shape|size|capacity|nbytes|numel|length|len\()'))
           OR (contains(g.compared, x.target) AND regexp_matches(g.compared, '\bis\s+None\b'))
           OR (regexp_matches(g.compared, '\bnot\s+in\s+\S')
               AND starts_with(x.target,
                               trim(regexp_extract(g.compared, '\bnot\s+in\s+(.+)$', 1)) || '['))
           OR EXISTS (SELECT 1 FROM cuda_nodes() u
                      JOIN cuda_nodes() ut ON ut.fact_id = u.fact_id AND ut.parent_id = u.node_id
                        AND ut.sibling_index = 0
                      WHERE u.fact_id = g.fact_id AND u.node_id BETWEEN g.first AND g.last
                        AND u.type IN ('assignment', 'augmented_assignment')
                        AND u.node_id <> a.node_id
                        AND syntax_slice(s.content, ut.start_line, ut.start_column, ut.end_line,
                                         ut.end_column) IN (g.left_side, g.right_side))
    """


class CudaAllocations(CudaRelation):
    """Calls that create device memory: a constructor of a device array (`cupy.empty`,
    `torch.zeros(..., device="cuda")`, `numba.cuda.device_array`, `arrays.empty` where `arrays`
    is a GPU module) and the framework it comes from, but not one of zero elements. `grows` marks a buffer that is built only
    when it must grow: the call sits in a capacity check, or the function holding it is called
    only from capacity checks (and constructors, which start the buffer empty). `output` marks a buffer the function returns (or yields), the
    allocation of a result and not a scratch buffer."""

    stored = True
    query = """
        WITH made AS (
          SELECT c.fact_id, c.path, c.node_id, c.scope_id, c.start_line, c.start_column,
                 c.end_line, c.end_column, c.shown, t.fw AS framework
          FROM cuda_calls() c
          JOIN cuda_types() t ON t.key = 'e:' || c.fact_id || ':' || c.node_id AND t.ty = 'device'
          WHERE regexp_matches(coalesce(c.method, regexp_extract(c.callee_text, '[^.]+$')),
                '^(empty|zeros|ones|full|empty_like|zeros_like|ones_like|full_like|arange|array|asarray|device_array|device_array_like|to_device|alloc|malloc|linspace|eye|rand|randn|randint|copy)$')
            AND NOT EXISTS (SELECT 1 FROM cuda_arguments() a
                            JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.node_id = a.value_id
                            WHERE a.fact_id = c.fact_id AND a.call_id = c.node_id AND a.position = 0
                              AND v.type = 'integer' AND v.name = '0')
        ), exits AS (
          SELECT r.fact_id, r.node_id AS first, r.node_id + r.descendant_count AS last,
                 coalesce(r.scope."function", -1) AS scope_id
          FROM cuda_nodes() r WHERE r.type IN ('return_statement', 'yield')
        )
        SELECT m.*,
               EXISTS (SELECT 1 FROM cuda_capacity_guarded() g
                       WHERE g.fact_id = m.fact_id AND g.node_id = m.node_id)
               OR (EXISTS (SELECT 1 FROM cuda_resolved_calls() r
                           WHERE r.definition_fact = m.fact_id AND r.definition_id = m.scope_id
                             AND r.kind = 'function')
                   AND NOT EXISTS (SELECT 1 FROM cuda_resolved_calls() r
                                   JOIN cuda_calls() caller ON caller.fact_id = r.fact_id
                                     AND caller.node_id = r.node_id
                                   LEFT JOIN cuda_definitions() home ON home.fact_id = caller.fact_id
                                     AND home.node_id = caller.scope_id AND home.kind = 'function'
                                   WHERE r.definition_fact = m.fact_id
                                     AND r.definition_id = m.scope_id AND r.kind = 'function'
                                     AND coalesce(home.name, '') <> '__init__'
                                     AND NOT EXISTS (SELECT 1 FROM cuda_capacity_guarded() g
                                                     WHERE g.fact_id = r.fact_id
                                                       AND g.node_id = r.node_id))) AS grows,
               EXISTS (SELECT 1 FROM exits e
                       WHERE e.fact_id = m.fact_id AND e.scope_id = m.scope_id
                         AND m.node_id BETWEEN e.first AND e.last)
               OR EXISTS (SELECT 1 FROM cuda_stores() s
                          JOIN cuda_identifier_reads() u ON u.fact_id = s.fact_id
                            AND u.scope_id = s.scope_id AND u.name = s.name
                          JOIN exits e ON e.fact_id = u.fact_id AND e.scope_id = u.scope_id
                            AND u.node_id BETWEEN e.first AND e.last
                          WHERE s.fact_id = m.fact_id AND s.source_id = m.node_id
                            AND s.kind = 'assignment') AS output
        FROM made m
    """
