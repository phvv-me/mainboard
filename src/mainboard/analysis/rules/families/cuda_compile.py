"""What decides when a device function compiles: memoized factories, and reads of compile state."""

from .cuda_base import CudaRelation


class CudaMemoized(CudaRelation):
    """The functions that build a device function once per argument: those a decorator memoizes
    (`cache`, `lru_cache`, `cached_property`), those wrapped by `cache(make)` or
    `lru_cache(...)(make)`, and those that store what they define in a container that outlives
    the call (`KERNELS[width] = fill`, `KERNELS.setdefault(width, fill)`, `self.fill = fill`)."""

    stored = True
    query = """
        SELECT DISTINCT fact_id, definition_id AS function_id FROM cuda_decorators()
        WHERE regexp_matches(written, '(^|\\.)(cache|lru_cache|cached_property)$')
        UNION
        SELECT d.fact_id, d.node_id
        FROM cuda_calls() c
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id AND a.position = 0
        JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.node_id = a.value_id
          AND v.type = 'identifier'
        JOIN cuda_definitions() d ON d.fact_id = c.fact_id AND d.kind = 'function'
          AND d.name = v.name AND d.outer_id IS NULL
        WHERE regexp_matches(c.callee_text, '(^|\\.)(cache|lru_cache)(\\(.*\\))?$')
        UNION
        SELECT k.fact_id, k.scope."function"
        FROM cuda_nodes() k
        WHERE k.type = 'function_definition' AND k.scope."function" IS NOT NULL
          AND (EXISTS (
                 SELECT 1 FROM cuda_nodes() s
                 JOIN cuda_nodes() t ON t.fact_id = s.fact_id AND t.parent_id = s.node_id
                   AND t.sibling_index = 0 AND t.type IN ('subscript', 'attribute')
                 JOIN cuda_nodes() v ON v.fact_id = s.fact_id AND v.parent_id = s.node_id
                   AND v.sibling_index = 2 AND v.type = 'identifier' AND v.name = k.name
                 WHERE s.fact_id = k.fact_id AND s.type = 'assignment'
                   AND s.scope."function" = k.scope."function")
               OR EXISTS (
                 SELECT 1 FROM cuda_calls() c
                 JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
                 JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.node_id = a.value_id
                   AND v.type = 'identifier' AND v.name = k.name
                 WHERE c.fact_id = k.fact_id AND c.scope_id = k.scope."function"
                   AND c.method IN ('setdefault', 'add', 'append', 'put', 'update', 'insert')))
    """


class CudaCompileStateReads(CudaRelation):
    """Reads of what a device function has compiled so far (`kernel.compiled`,
    `kernel.signatures`) where the object is a kernel or device function of the project."""

    stored = True
    query = """
        SELECT a.fact_id, a.path, a.node_id, coalesce(a.scope."function", -1) AS scope_id,
               a.name AS shown, a.start_line, a.start_column, a.end_line, a.end_column,
               k.key AS kernel_key
        FROM cuda_nodes() a
        JOIN cuda_chains() h ON h.fact_id = a.fact_id
          AND h.chain = regexp_replace(a.name, '\\.[^.]+$', '')
        JOIN cuda_definitions() k ON k.key = h.definition AND k.kind = 'function'
        JOIN cuda_device_code() d ON d.fact_id = k.fact_id AND d.function_id = k.node_id
          AND d.role IN ('kernel', 'device')
        WHERE a.type = 'attribute'
          AND regexp_matches(a.name, '\\.(compiled|signatures|nopython_signatures|is_compiled)$')
    """
