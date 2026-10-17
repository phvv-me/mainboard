"""Functions compiled for the device, under any framework: numba-cuda and patos kernels and
device functions, Triton, Helion, Warp, CuPy's `rawkernel`, cuTile and TileLang kernels.

A function is device code when a decorator resolves to one of those, or when it is defined inside
device code. Host rules leave such functions alone: `if x:` inside a kernel is a branch on the
device, not a wait for it.
"""

from .cuda_base import CudaRelation


class CudaDecorators(CudaRelation):
    """Every decorator of a function or class: the name as written (`cuda.jit`, `kernel`), and
    whether it is a call with `device=True`."""

    stored = True
    query = """
        SELECT t.fact_id, t.node_id AS definition_id, d.sibling_index AS position,
               CASE WHEN e.type = 'call' THEN k.name ELSE e.name END AS written,
               e.type AS expression_type,
               EXISTS (
                 SELECT 1 FROM cuda_nodes() a
                 JOIN cuda_nodes() kw ON kw.fact_id = a.fact_id AND kw.parent_id = a.node_id
                   AND kw.type = 'keyword_argument'
                 JOIN cuda_nodes() kn ON kn.fact_id = kw.fact_id AND kn.parent_id = kw.node_id
                   AND kn.sibling_index = 0 AND kn.name = 'device'
                 JOIN cuda_nodes() kv ON kv.fact_id = kw.fact_id AND kv.parent_id = kw.node_id
                   AND kv.sibling_index = 2 AND kv.type = 'true'
                 WHERE a.fact_id = e.fact_id AND a.parent_id = e.node_id
                   AND a.type = 'argument_list') AS device_flag
        FROM cuda_nodes() w
        JOIN cuda_nodes() d ON d.fact_id = w.fact_id AND d.parent_id = w.node_id
          AND d.type = 'decorator'
        JOIN cuda_nodes() t ON t.fact_id = w.fact_id AND t.parent_id = w.node_id
          AND t.type IN ('function_definition', 'class_definition')
        JOIN cuda_nodes() e ON e.fact_id = d.fact_id AND e.parent_id = d.node_id
          AND e.sibling_index = 1
        LEFT JOIN cuda_nodes() k ON k.fact_id = e.fact_id AND k.parent_id = e.node_id
          AND k.sibling_index = 0 AND e.type = 'call'
        WHERE w.type = 'decorated_definition'
    """


class CudaDeviceRoles(CudaRelation):
    """What each decorator that compiles for the device makes of a function: a `kernel` the host
    launches or a `device` function other device code calls, and the framework."""

    query = r"""
        SELECT * FROM (VALUES
          ('^(numba\.)?cuda\.jit$', 'kernel', 'numba'),
          ('^patos\.cuda\.typed(\.\w+)*\.kernel$', 'kernel', 'patos'),
          ('^patos\.cuda\.typed(\.\w+)*\.(device|ptx)$', 'device', 'patos'),
          ('^triton\.(jit|autotune|heuristics)$', 'kernel', 'triton'),
          ('^triton\.experimental\.gluon\.jit$', 'kernel', 'triton'),
          ('^helion\.(kernel|jit)$', 'kernel', 'helion'),
          ('^warp\.kernel$', 'kernel', 'warp'), ('^warp\.func$', 'device', 'warp'),
          ('^cupyx\.jit\.rawkernel$', 'kernel', 'cupy'),
          ('^cuda\.tile\.(kernel|jit)$', 'kernel', 'cutile'),
          ('^tilelang\.(jit|prim_func)$', 'kernel', 'tilelang')
        ) t(pattern, role, framework)
    """


class CudaDeviceCode(CudaRelation):
    """Functions compiled for the device: `role` is `kernel`, `device` or `nested` (defined
    inside device code), `framework` the compiler. Calls and reads inside belong to the device."""

    stored = True
    query = """
        WITH decorated AS (
          SELECT d.fact_id, d.definition_id, r.framework,
                 CASE WHEN r.framework = 'numba' AND d.device_flag THEN 'device' ELSE r.role END
                   AS role
          FROM cuda_decorators() d
          JOIN cuda_chains() c ON c.fact_id = d.fact_id AND c.chain = d.written
          JOIN cuda_device_roles() r ON regexp_matches(c.qualified, r.pattern)
            OR list_bool_or(list_transform(c.seen, lambda name: regexp_matches(name, r.pattern)))
          WHERE d.expression_type <> 'class_definition'
          QUALIFY row_number() OVER (PARTITION BY d.fact_id, d.definition_id
                                     ORDER BY r.role = 'kernel' DESC) = 1
        ), own AS (
          SELECT f.fact_id, f.path, f.node_id AS function_id, f.name, f.qualname, x.role,
                 x.framework, f.start_line, f.start_column, f.end_line, f.end_column
          FROM decorated x JOIN cuda_definitions() f ON f.fact_id = x.fact_id
            AND f.node_id = x.definition_id AND f.kind = 'function'
        )
        SELECT * FROM own
        UNION ALL
        SELECT f.fact_id, f.path, f.node_id, f.name, f.qualname, 'nested', o.framework,
               f.start_line, f.start_column, f.end_line, f.end_column
        FROM cuda_definitions() f
        JOIN cuda_scopes() s ON s.fact_id = f.fact_id AND s.scope_id = f.node_id AND s.depth > 0
        JOIN own o ON o.fact_id = s.fact_id AND o.function_id = s.visible_id
        WHERE f.kind = 'function'
    """
