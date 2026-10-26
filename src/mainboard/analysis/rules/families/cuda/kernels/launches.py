"""Kernel launches under any framework, with the stream each names (or does not).

A launch is a call that runs a device kernel from the host: `kernel[grid, block, stream](...)`
(numba-cuda, patos, Triton and Helion spell it with a subscript or a plain call of the decorated
function), `cuda.core.launch(stream, config, kernel, ...)`, `warp.launch(kernel, ..., stream=)`.
`stream_kind` is `explicit` when a stream is passed (numba's third subscript item, or the `stream=`
keyword of the call, which is how patos, Triton and Helion take one), `none` when the framework
then uses the default stream (numba's `kernel[grid, block](...)`), `ambient` when it reads the
current one (patos, Triton and Helion without a `stream=`).
"""

from ..base import CudaRelation


class CudaLaunches(CudaRelation):
    """Each kernel launch: the call, the framework, the kernel it runs when the project defines
    it (`kernel_key`), and the stream."""

    stored = True
    query = """
        WITH subscripted AS (
          SELECT c.fact_id, c.node_id AS call_id, o.name AS object_text, c.callee_id
          FROM cuda_calls() c
          JOIN cuda_nodes() o ON o.fact_id = c.fact_id AND o.parent_id = c.callee_id
            AND o.sibling_index = 0
          WHERE c.callee_type = 'subscript' AND o.type IN ('identifier', 'attribute')
        ), configured AS (
          SELECT s.fact_id, s.call_id, s.object_text,
                 list(i.node_id ORDER BY i.sibling_index) AS items
          FROM subscripted s
          JOIN cuda_nodes() i ON i.fact_id = s.fact_id AND i.parent_id = s.callee_id
            AND i.sibling_index >= 2 AND i.type <> ']'
          GROUP BY ALL
        ), via_subscript AS (
          SELECT g.fact_id, g.call_id, k.key AS kernel_key, d.framework,
                 CASE WHEN d.framework = 'numba' AND len(g.items) >= 3 THEN 'explicit'
                      WHEN d.framework = 'numba' THEN 'none' ELSE 'ambient' END AS stream_kind,
                 CASE WHEN d.framework = 'numba' AND len(g.items) >= 3 THEN g.items[3] END
                   AS stream_id
          FROM configured g
          JOIN cuda_chains() h ON h.fact_id = g.fact_id AND h.chain = g.object_text
          JOIN cuda_definitions() k ON k.key = h.definition AND k.kind = 'function'
          JOIN cuda_device_code() d ON d.fact_id = k.fact_id AND d.function_id = k.node_id
            AND d.role = 'kernel'
        ), via_member AS (
          SELECT g.fact_id, g.call_id, k.key AS kernel_key, d.framework,
                 CASE WHEN d.framework = 'numba' AND len(g.items) >= 3 THEN 'explicit'
                      WHEN d.framework = 'numba' THEN 'none' ELSE 'ambient' END AS stream_kind,
                 CASE WHEN d.framework = 'numba' AND len(g.items) >= 3 THEN g.items[3] END
                   AS stream_id
          FROM configured g
          JOIN subscripted s ON s.fact_id = g.fact_id AND s.call_id = g.call_id
          JOIN cuda_nodes() o ON o.fact_id = s.fact_id AND o.parent_id = s.callee_id
            AND o.sibling_index = 0 AND o.type = 'attribute'
          JOIN cuda_nodes() r ON r.fact_id = o.fact_id AND r.parent_id = o.node_id
            AND r.sibling_index = 0
          JOIN cuda_nodes() m ON m.fact_id = o.fact_id AND m.parent_id = o.node_id
            AND m.sibling_index = 1
          JOIN cuda_types() t ON t.key = 'e:' || r.fact_id || ':' || r.node_id
            AND t.ty LIKE 'obj:%'
          JOIN cuda_members() mm ON mm.receiver = substr(t.ty, 5) AND mm.name = m.name
            AND mm.kind = 'method'
          JOIN cuda_methods() meth ON meth.ret = mm.member
          JOIN cuda_definitions() k ON k.fact_id = meth.fact_id AND k.node_id = meth.node_id
          JOIN cuda_device_code() d ON d.fact_id = k.fact_id AND d.function_id = k.node_id
            AND d.role = 'kernel'
        ), via_call AS (
          SELECT t.fact_id, t.node_id AS call_id, t.key AS kernel_key, d.framework,
                 'ambient' AS stream_kind, CAST(NULL AS BIGINT) AS stream_id
          FROM cuda_resolved_calls() t
          JOIN cuda_device_code() d ON d.fact_id = t.definition_fact
            AND d.function_id = t.definition_id AND d.role = 'kernel'
          WHERE t.kind = 'function' AND d.framework IN ('helion', 'triton')
        ), via_api AS (
          SELECT c.fact_id, c.node_id AS call_id, CAST(NULL AS VARCHAR) AS kernel_key,
                 CASE WHEN c.qualified_name LIKE 'warp.%' THEN 'warp' ELSE 'cuda.core' END
                   AS framework,
                 'explicit' AS stream_kind, a.value_id AS stream_id
          FROM cuda_calls() c
          LEFT JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
            AND (a.keyword = 'stream' OR (a.position = 0 AND c.qualified_name LIKE 'cuda.core%'))
          WHERE c.qualified_name IN ('cuda.core.launch', 'cuda.core.experimental.launch',
                                     'warp.launch')
        ), named AS (
          SELECT fact_id, call_id, value_id FROM cuda_arguments() WHERE keyword = 'stream'
        ), launched AS (
          SELECT * FROM via_subscript UNION ALL SELECT * FROM via_member
          UNION ALL SELECT * FROM via_call
          UNION ALL SELECT * FROM via_api
        )
        SELECT l.fact_id, c.path, l.call_id AS node_id, c.start_line, c.start_column, c.end_line,
               c.end_column, c.scope_id, c.shown, l.kernel_key, l.framework,
               CASE WHEN n.value_id IS NOT NULL THEN 'explicit'
                    WHEN l.stream_kind = 'explicit' AND l.stream_id IS NULL THEN 'none'
                    ELSE l.stream_kind END AS stream_kind,
               coalesce(l.stream_id, n.value_id) AS stream_id
        FROM launched l JOIN cuda_calls() c ON c.fact_id = l.fact_id AND c.node_id = l.call_id
        LEFT JOIN named n ON n.fact_id = l.fact_id AND n.call_id = l.call_id
    """
