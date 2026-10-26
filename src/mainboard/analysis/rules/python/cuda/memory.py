"""PY-CUDA rules about device allocation."""

from ....catalog import Example, Rule
from ..constants import cuda_paths


class HotAllocation(Rule, id="PY-CUDA0005", applicability="review"):
    """Device memory allocated on every iteration of a loop or call of a function a loop reaches,
    or a buffer regrown to exactly the size asked for, so a ramp of sizes reallocates each step.
    A buffer built only when a capacity check says it must grow, when it is still `None`, when
    its key is not yet in the dict that memoizes it, or when a counter the branch then updates
    says so, is not reported, and neither is a result buffer the function returns or yields: each
    result must be a buffer of its own, which no reuse can replace."""

    exempt: tuple[str, ...] = cuda_paths.MEASURING

    query = r"""
        WITH hot AS (
          SELECT * FROM cuda_hot()
          WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(origin_path, p))), false)
          QUALIFY row_number() OVER (PARTITION BY fact_id, function_id
                                     ORDER BY hops, origin_path, origin_line) = 1
        ), placed AS (
          SELECT a.*, CASE WHEN EXISTS (SELECT 1 FROM cuda_loop_calls() l
                                        WHERE l.fact_id = a.fact_id AND l.node_id = a.node_id)
                           THEN 'loop' ELSE 'called' END AS problem,
                 h.origin_path, h.origin_line
          FROM cuda_allocations() a
          LEFT JOIN hot h ON h.fact_id = a.fact_id AND h.function_id = a.scope_id
          WHERE (EXISTS (SELECT 1 FROM cuda_loop_calls() l
                         WHERE l.fact_id = a.fact_id AND l.node_id = a.node_id)
                 OR h.function_id IS NOT NULL)
            AND a.scope_id <> -1 AND NOT a.grows AND NOT a.output
            AND NOT EXISTS (SELECT 1 FROM cuda_device_code() d
                            WHERE d.fact_id = a.fact_id AND d.function_id = a.scope_id)
        ), repeated AS (
          SELECT arg_min(path, rank) AS path, arg_min(start_line, rank) AS start_line,
                 arg_min(start_column, rank) AS start_column,
                 arg_min(end_line, rank) AS end_line, arg_min(end_column, rank) AS end_column,
                 arg_min(shown, rank) AS shown,
                 CASE WHEN count(*) > 1 THEN ' (and ' || (count(*) - 1) || ' more in this function)'
                      ELSE '' END AS extra,
                 CASE WHEN bool_or(problem = 'loop') THEN 'loop' ELSE 'called' END AS problem,
                 arg_min(origin_path, rank) AS origin_path, arg_min(origin_line, rank) AS origin_line,
                 CAST(NULL AS VARCHAR) AS buffer
          FROM (SELECT *, (problem <> 'loop')::INTEGER * 100000000
                          + start_line * 10000 + start_column AS rank FROM placed)
          GROUP BY fact_id, scope_id
        ), growth AS (
          SELECT k.path, k.start_line, k.start_column, k.end_line, k.end_column, k.shown,
                 '' AS extra, 'exact' AS problem, CAST(NULL AS VARCHAR) AS origin_path,
                 CAST(NULL AS BIGINT) AS origin_line,
                 syntax_slice(s.content, lhs.start_line, lhs.start_column, lhs.end_line,
                              lhs.end_column) AS buffer
          FROM cuda_nodes() i
          JOIN cuda_nodes() cond ON cond.fact_id = i.fact_id AND cond.parent_id = i.node_id
            AND cond.sibling_index = 1
          JOIN cuda_nodes() cmp ON cmp.fact_id = i.fact_id AND cmp.type = 'comparison_operator'
            AND cmp.node_id BETWEEN cond.node_id AND cond.node_id + cond.descendant_count
          JOIN cuda_nodes() op ON op.fact_id = i.fact_id AND op.parent_id = cmp.node_id
            AND op.type IN ('<', '<=')
          JOIN cuda_nodes() lhs ON lhs.fact_id = i.fact_id AND lhs.parent_id = cmp.node_id
            AND lhs.sibling_index = op.sibling_index - 1
          JOIN cuda_nodes() rhs ON rhs.fact_id = i.fact_id AND rhs.parent_id = cmp.node_id
            AND rhs.sibling_index = op.sibling_index + 1
          JOIN cuda_nodes() blk ON blk.fact_id = i.fact_id AND blk.parent_id = i.node_id
            AND blk.type = 'block' AND blk.sibling_index = (
              SELECT min(b.sibling_index) FROM cuda_nodes() b
              WHERE b.fact_id = i.fact_id AND b.parent_id = i.node_id AND b.type = 'block')
          JOIN cuda_calls() k ON k.fact_id = i.fact_id
            AND k.node_id BETWEEN blk.node_id AND blk.node_id + blk.descendant_count
          JOIN cuda_arguments() a ON a.fact_id = k.fact_id AND a.call_id = k.node_id
            AND a.position = 0
          JOIN cuda_nodes() arg ON arg.fact_id = a.fact_id AND arg.node_id = a.value_id
          JOIN source_files s ON s.path = i.path
          WHERE i.type = 'if_statement' AND EXISTS (SELECT 1 FROM cuda_device_files())
            AND regexp_matches(coalesce(k.method, regexp_extract(k.callee_text, '[^.]+$')),
                               '^(empty|zeros|ones|full|device_array|alloc|malloc)$')
            AND regexp_replace(syntax_slice(s.content, arg.start_line, arg.start_column,
                                            arg.end_line, arg.end_column), '\s+', '', 'g')
                IN (regexp_replace(syntax_slice(s.content, rhs.start_line, rhs.start_column,
                                                rhs.end_line, rhs.end_column), '\s+', '', 'g'),
                    'max(' || regexp_replace(syntax_slice(s.content, rhs.start_line,
                         rhs.start_column, rhs.end_line, rhs.end_column), '\s+', '', 'g') || ',1)',
                    'int(' || regexp_replace(syntax_slice(s.content, rhs.start_line,
                         rhs.start_column, rhs.end_line, rhs.end_column), '\s+', '', 'g') || ')')
        )
        SELECT f.*,
               CASE f.problem
                 WHEN 'loop' THEN ' on every iteration of its loop'
                 WHEN 'called' THEN ' on every call (the function runs once per iteration of the '
                                    || 'loop at ' || f.origin_path || ':' || f.origin_line || ')'
                 ELSE '' END AS where_text,
               CASE f.problem
                 WHEN 'exact' THEN 'grows `' || f.buffer || '` to exactly the size asked for, so '
                      || 'every request a little larger than the last allocates again (and drops '
                      || 'any graph captured on the old buffer)'
                 ELSE 'allocates device memory' END AS what,
               CASE WHEN f.problem = 'exact'
                    THEN 'grow by a factor (`max(size, 2 * held)`) so a ramp of sizes '
                         || 'reallocates a logarithmic number of times'
                    ELSE 'allocate once outside the loop, with headroom, and reuse the buffer '
                         || '(`out=`, `fill`)' END AS remedy
        FROM (SELECT * FROM repeated UNION ALL SELECT * FROM growth) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = (
        "`{shown}`{extra} {what}{where_text}: each allocation goes through the pool, may reach "
        "`cudaMalloc`, and fragments it. Fix: {remedy}"
    )
    examples = (
        Example(
            bad=(
                """import cupy as cp


def fill(batches, n):
    for batch in batches:
        scratch = cp.empty(n)
        scratch[:] = batch
"""
            ),
            good=(
                """import cupy as cp


def fill(batches, n):
    scratch = cp.empty(n)
    for batch in batches:
        scratch[:] = batch
"""
            ),
        ),
        Example(
            bad="import cupy as cp\n\n\nclass Buffer:\n    def __init__(self):\n        self.held = cp.empty(0)\n\n    def take(self, size):\n        if self.held.shape[0] < size:\n            self.held = cp.empty(size)\n        return self.held[:size]\n",
            good="import cupy as cp\n\n\nclass Buffer:\n    def __init__(self):\n        self.held = cp.empty(0)\n\n    def take(self, size):\n        if self.held.shape[0] < size:\n            self.held = cp.empty(max(size, 2 * self.held.shape[0]))\n        return self.held[:size]\n",
        ),
    )


class ParameterShapedKernelArray(Rule, id="PY-CUDA0020"):
    """A `cuda.local.array` or `cuda.shared.array` whose shape reads a parameter of the kernel or
    device function around it (mcmr's PY-NUMB0004): these arrays need a compile-time constant
    shape, which a launch-time argument cannot be. A record method's `self` is left alone, since
    its `Constant[...]` fields are compile-time literals the facts cannot tell apart."""

    query = r"""
        WITH allocations AS (
          SELECT c.fact_id, c.path, c.start_line, c.start_column, c.end_line, c.end_column,
                 c.scope_id, c.arguments_id, d.qualname
          FROM cuda_calls() c
          JOIN cuda_device_code() d ON d.fact_id = c.fact_id AND d.function_id = c.scope_id
          WHERE regexp_matches(c.qualified_name, '\.(local|shared)\.array$')
        ), shapes AS (
          SELECT a.*, s.node_id AS shape_id, s.descendant_count AS shape_size
          FROM allocations a
          JOIN cuda_nodes() s ON s.fact_id = a.fact_id AND s.parent_id = a.arguments_id
            AND s.type <> 'keyword_argument'
            AND s.semantic_type NOT IN ('PARSER_DELIMITER', 'METADATA_COMMENT')
          QUALIFY row_number() OVER (PARTITION BY a.fact_id, a.arguments_id
                                     ORDER BY s.sibling_index) = 1
        )
        SELECT DISTINCT s.path, s.start_line, s.start_column, s.end_line, s.end_column,
               s.qualname, p.name AS parameter
        FROM shapes s
        JOIN cuda_nodes() i ON i.fact_id = s.fact_id AND i.type = 'identifier'
          AND i.node_id BETWEEN s.shape_id AND s.shape_id + s.shape_size
        JOIN core_file f ON f.fact_id = s.fact_id
        JOIN core_param p ON p.file_id = f.file_id AND p.def_id = s.scope_id AND p.name = i.name
        WHERE p.name <> 'self'
    """
    message = (
        "`{qualname}` shapes a kernel array by its parameter `{parameter}`, which is not a "
        "compile-time constant"
    )
    examples = (
        Example(
            bad=(
                """from numba import cuda


@cuda.jit
def reduce(values, width):
    scratch = cuda.local.array(width, values.dtype)
    scratch[0] = values[0]
"""
            ),
            good=(
                """from numba import cuda

WIDTH = 32


@cuda.jit
def reduce(values):
    scratch = cuda.local.array(WIDTH, values.dtype)
    scratch[0] = values[0]
"""
            ),
        ),
    )
