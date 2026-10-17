"""PY-CUDA rules about when device code compiles."""

from typing import ClassVar

from ...catalog import Example, Rule
from .cuda_paths import MEASURING

_HASHABLE = r"^(int|str|bool|float|bytes|tuple\[[^\]]*\]|frozenset\[[^\]]*\]|Literal\[[^\]]*\])$"


class CompileOnTheHotPath(Rule, id="PY-CUDA0006", applicability="safe"):
    """Compilation that happens more than once or depends on history: a device function defined
    inside an unmemoized function (a new function object compiles on every call), a
    `torch.compile` in a loop, a route chosen by whether a kernel is already compiled. A factory
    over hashable arguments that returns the function, or a value built from it, is memoized by
    `@cache`: one compile per argument is what a closure over compile-time constants is for."""

    exempt: tuple[str, ...] = MEASURING
    change: ClassVar[str] = (
        "decorated `{factory}` with `@cache`, so `{name}` compiles once per argument"
    )

    query = r"""
        WITH nested AS (
          SELECT k.path, k.start_line, k.start_column, k.end_line, k.end_column, k.name,
                 'nested' AS problem, f.qualname AS factory, f.node_id AS factory_id, k.fact_id,
                 d.framework, f.start_line AS factory_line, f.start_column AS factory_column,
                 f.end_line AS factory_end
          FROM cuda_device_code() d
          JOIN cuda_definitions() k ON k.fact_id = d.fact_id AND k.node_id = d.function_id
          JOIN cuda_definitions() f ON f.fact_id = k.fact_id AND f.node_id = k.outer_id
            AND f.kind = 'function'
          WHERE d.role IN ('kernel', 'device') AND k.outer_id IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM cuda_scopes() s
                            JOIN cuda_memoized() m ON m.fact_id = s.fact_id
                              AND m.function_id = s.visible_id
                            WHERE s.fact_id = k.fact_id AND s.scope_id = f.node_id)
            AND NOT EXISTS (SELECT 1 FROM cuda_device_code() o
                            WHERE o.fact_id = f.fact_id AND o.function_id = f.node_id)
        ), compiled_in_loop AS (
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column,
                 regexp_extract(c.shown, '^[^(]+') AS name, 'compile' AS problem,
                 CAST(NULL AS VARCHAR) AS factory, CAST(NULL AS BIGINT) AS factory_id, c.fact_id,
                 'torch' AS framework, CAST(NULL AS INTEGER) AS factory_line,
                 CAST(NULL AS INTEGER) AS factory_column, CAST(NULL AS INTEGER) AS factory_end
          FROM cuda_loop_calls() c WHERE c.qualified_name = 'torch.compile'
        ), routed AS (
          SELECT r.path, r.start_line, r.start_column, r.end_line, r.end_column, r.shown AS name,
                 'state' AS problem, CAST(NULL AS VARCHAR), CAST(NULL AS BIGINT), r.fact_id,
                 d.framework, CAST(NULL AS INTEGER), CAST(NULL AS INTEGER), CAST(NULL AS INTEGER)
          FROM cuda_compile_state_reads() r
          JOIN cuda_definitions() k ON k.key = r.kernel_key
          JOIN cuda_device_code() d ON d.fact_id = k.fact_id AND d.function_id = k.node_id
          WHERE NOT EXISTS (SELECT 1 FROM cuda_device_code() o
                            WHERE o.fact_id = r.fact_id AND o.function_id = r.scope_id)
        ), found AS (
          SELECT * FROM nested UNION ALL SELECT * FROM compiled_in_loop
          UNION ALL SELECT * FROM routed
        ), factories AS (
          SELECT f.*,
                 f.problem = 'nested'
                 AND EXISTS (SELECT 1 FROM cuda_nodes() r
                             JOIN cuda_nodes() v ON v.fact_id = r.fact_id
                               AND v.node_id BETWEEN r.node_id + 1 AND r.node_id + r.descendant_count
                             WHERE r.fact_id = f.fact_id AND r.type = 'return_statement'
                               AND r.scope."function" = f.factory_id
                               AND v.type = 'identifier' AND v.name = f.name)
                 AND NOT EXISTS (SELECT 1 FROM cuda_parameters() p
                                 WHERE p.fact_id = f.fact_id AND p.function_id = f.factory_id
                                   AND (p.name IN ('self', 'cls')
                                        OR NOT regexp_matches(replace(p.annotation, ' ', ''),
                                                              '$hashable')))
                 AND NOT EXISTS (SELECT 1 FROM cuda_decorators() x
                                 WHERE x.fact_id = f.fact_id AND x.definition_id = f.factory_id) AS cacheable
          FROM found f
        )
        SELECT f.path, f.start_line, f.start_column, f.end_line, f.end_column, f.name, f.problem,
               f.factory, f.framework, f.cacheable,
               coalesce((SELECT w.start_line FROM cuda_nodes() w
                         WHERE w.fact_id = f.fact_id AND w.type = 'decorated_definition'
                           AND w.node_id = (SELECT x.parent_id FROM cuda_nodes() x
                                            WHERE x.fact_id = f.fact_id AND x.node_id = f.factory_id)),
                        f.factory_line) AS insert_line,
               f.factory_column AS insert_column,
               CASE f.problem
                 WHEN 'nested' THEN 'defines `' || f.name || '` inside `' || f.factory || '`: each call '
                      || 'makes a new function object, and ' || f.framework || ' compiles each one '
                      || 'again, so compilation lands on the hot path'
                 WHEN 'compile' THEN 'compiles inside a loop: each iteration re-traces and re-guards '
                      || 'the function'
                 ELSE 'branches on compile history: a cold process and a warm one take different '
                      || 'routes for the same input, and the route changes the first time '
                      || 'anything compiles' END AS what,
               CASE f.problem
                 WHEN 'nested' THEN CASE WHEN f.cacheable THEN 'decorate `' || f.factory
                      || '` with `@cache` so one compile serves every call'
                      ELSE 'define it at module level, or memoize `' || f.factory
                      || '` on hashable arguments' END
                 WHEN 'compile' THEN 'compile once before the loop'
                 ELSE 'choose the route from the input alone (size, flags) and warm kernels up '
                      || 'explicitly where compile time matters' END AS remedy
        FROM factories f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = "`{name}` {what}. Fix: {remedy}"
    fix = """
        SELECT finding_key, path, insert_line AS start_line, 0 AS start_column,
               insert_line AS end_line, 0 AS end_column,
               repeat(' ', insert_column) || '@cache' || chr(10) AS text,
               'from functools import cache' AS import
        FROM finding WHERE cacheable
        QUALIFY row_number() OVER (PARTITION BY path, insert_line ORDER BY finding_key) = 1
    """
    examples = (
        Example(
            bad=(
                "from numba import cuda\n\n\ndef make(width: int):\n    @cuda.jit\n"
                "    def fill(x):\n        x[0] = width\n\n    return fill\n"
            ),
            good=(
                "from numba import cuda\nfrom functools import cache\n\n\n@cache\n"
                "def make(width: int):\n    @cuda.jit\n"
                "    def fill(x):\n        x[0] = width\n\n    return fill\n"
            ),
        ),
        Example(
            bad=(
                "from numba import cuda\n\n\ndef make(width: int):\n    @cuda.jit\n"
                "    def fill(x):\n        x[0] = width\n\n    return (fill, width)\n"
            ),
            good=(
                "from numba import cuda\nfrom functools import cache\n\n\n@cache\n"
                "def make(width: int):\n    @cuda.jit\n"
                "    def fill(x):\n        x[0] = width\n\n    return (fill, width)\n"
            ),
        ),
        Example(
            bad="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef route(x):\n    if kernel.compiled:\n        return 1\n    return 0\n",
            good="",
        ),
    )

    def sql(self) -> str:
        """The finding query with the hashable annotation pattern spelled in."""
        return self.query.replace("$hashable", _HASHABLE)
