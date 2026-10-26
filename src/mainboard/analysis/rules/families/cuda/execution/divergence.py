"""The barriers and collectives of device code, and the conditions that govern them: calls that
need every thread of a block or warp to arrive, the functions that reach them, and the branches
whose condition decides whether code runs (`cuda_levels` says how much threads agree on each).
"""

from ...constants.cuda_base import NUMBA_CUDA
from ..base import CudaRelation

# patos's warp operations whose result is one value for the whole warp, and the ones that give each
# lane its own (a prefix sum, the slot a lane claims). `ballot_prefix` returns a record whose
# fields `CudaLibraryResults` levels one by one.
_WARP_SHARED = "ballot|broadcast|sum|min|max|and_|or_|xor"
_WARP_PER_LANE = "inclusive_sum|reserve"
_WARP_API = f"{_WARP_SHARED}|{_WARP_PER_LANE}|ballot_prefix"
_FULL_MASK = r"(?i)^(0x0*f{8}|.*full.*|.*all_lanes.*)$"


class CudaLibraryResults(CudaRelation):
    """What a library collective returns in a record, field by field: `function` returns a
    `record` whose `field` (the `position`th) holds a value of `level` (0 the whole block agrees,
    1 the lanes of a warp, 2 each lane has its own). A project that defines the record or the
    function derives the same from their bodies; a project that only calls them (a package built
    on patos) cannot see those bodies, so patos's own contract is written here, from
    `warp.ballot_prefix`: `before` is the lane's rank among the lanes that voted (its own),
    `total` their count (the warp's)."""

    stored = True
    query = """
        SELECT * FROM (VALUES
          ('patos.cuda.primitives.warp.ballot_prefix', 'patos.cuda.primitives.warp.Prefix',
           'before', 0, 2),
          ('patos.cuda.primitives.warp.ballot_prefix', 'patos.cuda.primitives.warp.Prefix',
           'total', 1, 1)
        ) AS t(function, record, field, position, level)
    """


class CudaCollectiveCalls(CudaRelation):
    """The calls that need every thread of a block (`requires` `block`) or of a warp (`warp`) to
    arrive: `cuda.syncthreads()`, `cuda.syncwarp()`, the shuffles and votes, patos's `warp.*` and
    `block.over(...)` operations, `cuda.cooperative` block and warp primitives. A warp call with a
    mask that is not the full warp is left out, since it names the lanes that arrive.
    `result_level` is the level of the value the call returns when every thread gets the same
    (`syncthreads_count`, a vote, a reduction), 2 for a collective that gives each lane its own
    value (`warp.inclusive_sum`, `warp.reserve`), null when the call is not one that fixes its
    result's level; a call that returns a record (`cuda_library_results`) has the highest level
    among its fields."""

    stored = True
    query = rf"""
        WITH named AS (
          SELECT c.fact_id, c.path, c.node_id, c.scope_id, c.start_line, c.start_column,
                 c.end_line, c.end_column, c.shown, c.arguments_id, c.qualified_name AS q,
                 c.method, a.value_id AS mask_id
          FROM cuda_calls() c
          LEFT JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
            AND a.position = 0
          WHERE EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = c.fact_id AND d.function_id = c.scope_id)
        ), classified AS (
          SELECT n.*,
                 CASE WHEN regexp_matches(q, '^{NUMBA_CUDA}\.syncthreads(_count|_and|_or)?$') THEN 'block'
                      WHEN regexp_matches(q, '^{NUMBA_CUDA}\.(syncwarp|shfl_sync|shfl_up_sync|shfl_down_sync|shfl_xor_sync|all_sync|any_sync|eq_sync|ballot_sync|match_any_sync|match_all_sync)$')
                      THEN 'warp'
                      WHEN regexp_matches(q, '^patos\.cuda\.primitives\.warp\.({_WARP_API})$')
                      THEN 'warp'
                      WHEN regexp_matches(q, '^cuda\.(coop|cooperative)(\.\w+)*\.block\.') THEN 'block'
                      WHEN regexp_matches(q, '^cuda\.(coop|cooperative)(\.\w+)*\.warp\.') THEN 'warp'
                 END AS requires,
                 coalesce(
                   (SELECT max(r.level) FROM cuda_library_results() r WHERE r.function = n.q),
                   CASE WHEN regexp_matches(q, '\.syncthreads_(count|and|or)$') THEN 0
                        WHEN regexp_matches(q, '\.(all_sync|any_sync|eq_sync|ballot_sync)$') THEN 1
                        WHEN regexp_matches(q, '^patos\.cuda\.primitives\.warp\.({_WARP_SHARED})$')
                        THEN 1
                        WHEN regexp_matches(q, '^patos\.cuda\.primitives\.warp\.({_WARP_PER_LANE})$')
                        THEN 2 END) AS result_level,
                 regexp_matches(q, '^{NUMBA_CUDA}\.(syncwarp|\w+_sync)$') AS masked
          FROM named n
        )
        SELECT c.fact_id, c.path, c.node_id, c.scope_id, c.start_line, c.start_column, c.end_line,
               c.end_column, c.shown, c.arguments_id, c.requires, c.result_level
        FROM classified c
        LEFT JOIN cuda_nodes() m ON m.fact_id = c.fact_id AND m.node_id = c.mask_id
        LEFT JOIN source_files s ON s.path = c.path
        WHERE c.requires IS NOT NULL
          AND (NOT c.masked
               OR (c.mask_id IS NULL AND c.method = 'syncwarp')
               OR regexp_matches(syntax_slice(s.content, m.start_line, m.start_column, m.end_line,
                                              m.end_column), '{_FULL_MASK}'))
        UNION ALL
        SELECT c.fact_id, c.path, c.node_id, c.scope_id, c.start_line, c.start_column, c.end_line,
               c.end_column, c.shown, c.arguments_id, 'block', 0
        FROM cuda_calls() c
        JOIN cuda_name_uses() u ON u.fact_id = c.fact_id AND u.node_id = c.receiver_id
        JOIN cuda_variable_sources() v ON v.variable = u.variable
        JOIN cuda_calls() o ON 'e:' || o.fact_id || ':' || o.node_id = v.source
          AND o.qualified_name = 'patos.cuda.primitives.block.over'
        WHERE c.method IN ('sum', 'min', 'max', 'and_', 'or_', 'xor', 'exclusive_sum')
          AND EXISTS (SELECT 1 FROM cuda_device_code() d
                      WHERE d.fact_id = c.fact_id AND d.function_id = c.scope_id)
    """


class CudaCollectiveSites(CudaRelation):
    """Each call that reaches a collective: the collective itself, and a call of a function that
    contains one (up to three calls down), with the strictest `requires` of what it reaches and
    `via` the function it goes through."""

    stored = True
    query = """
        WITH RECURSIVE reach(fact_id, function_id, requires, hops)
          USING KEY (fact_id, function_id, requires) AS (
          SELECT fact_id, scope_id, requires, 0 FROM cuda_collective_calls() WHERE scope_id <> -1
          UNION
          (SELECT c.fact_id, c.scope_id, r.requires, r.hops + 1
           FROM reach r
           JOIN cuda_resolved_calls() t ON t.definition_fact = r.fact_id
             AND t.definition_id = r.function_id AND t.kind = 'function' AND t.via <> 'override'
           JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
           WHERE r.hops < 3 AND c.scope_id <> -1
             AND NOT EXISTS (SELECT 1 FROM recurring.reach p
                             WHERE p.fact_id = c.fact_id AND p.function_id = c.scope_id
                               AND p.requires = r.requires))
        ), through AS (
          SELECT c.fact_id, c.path, c.node_id, c.scope_id, c.start_line, c.start_column,
                 c.end_line, c.end_column, c.shown, r.requires, f.qualname AS via
          FROM reach r
          JOIN cuda_resolved_calls() t ON t.definition_fact = r.fact_id
            AND t.definition_id = r.function_id AND t.kind = 'function' AND t.via <> 'override'
          JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
          JOIN cuda_definitions() f ON f.fact_id = r.fact_id AND f.node_id = r.function_id
          WHERE c.scope_id <> -1
            AND EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = c.fact_id AND d.function_id = c.scope_id)
            AND NOT EXISTS (SELECT 1 FROM cuda_collective_calls() k
                            WHERE k.fact_id = c.fact_id AND k.node_id = c.node_id)
        ), every AS (
          SELECT fact_id, path, node_id, scope_id, start_line, start_column, end_line, end_column,
                 shown, requires, CAST(NULL AS VARCHAR) AS via
          FROM cuda_collective_calls()
          UNION ALL
          SELECT * FROM through
        )
        SELECT fact_id, path, node_id, scope_id, start_line, start_column, end_line, end_column,
               shown, CASE WHEN bool_or(requires = 'block') THEN 'block' ELSE 'warp' END AS requires,
               min(via) AS via
        FROM every GROUP BY fact_id, path, node_id, scope_id, start_line, start_column, end_line,
                            end_column, shown
    """


class CudaKernelItems(CudaRelation):
    """What one item of a patos kernel runs on, from its `per=` argument: `BLOCK`, `WARP` or
    `THREAD` (the default)."""

    stored = True
    query = """
        SELECT f.fact_id, f.node_id AS function_id,
               coalesce(max(upper(regexp_extract(syntax_slice(s.content, kv.start_line,
                                                              kv.start_column, kv.end_line,
                                                              kv.end_column),
                                                 '(BLOCK|WARP|THREAD)', 1))), 'THREAD') AS per
        FROM cuda_device_code() d
        JOIN cuda_nodes() f ON f.fact_id = d.fact_id AND f.node_id = d.function_id
        JOIN source_files s ON s.path = d.path
        LEFT JOIN cuda_nodes() w ON w.fact_id = f.fact_id AND w.node_id = f.parent_id
          AND w.type = 'decorated_definition'
        LEFT JOIN cuda_nodes() kw ON kw.fact_id = w.fact_id AND kw.type = 'keyword_argument'
          AND kw.node_id BETWEEN w.node_id + 1 AND f.node_id - 1
        LEFT JOIN cuda_nodes() kn ON kn.fact_id = kw.fact_id AND kn.parent_id = kw.node_id
          AND kn.sibling_index = 0 AND kn.name = 'per'
        LEFT JOIN cuda_nodes() kv ON kv.fact_id = kw.fact_id AND kv.parent_id = kw.node_id
          AND kv.sibling_index = 2 AND kn.node_id IS NOT NULL
        WHERE d.framework = 'patos' AND d.role = 'kernel'
        GROUP BY f.fact_id, f.node_id
    """


class CudaBranches(CudaRelation):
    """Each place where whether code runs depends on a condition: an `if` or `elif` (its whole
    statement), a `while`, a `for` (its iterable), a conditional expression and the right side of
    `and`/`or`. `first`..`last` are the nodes the condition governs, `cond_first`..`cond_last`
    the condition itself."""

    stored = True
    query = """
        SELECT n.fact_id, n.path, n.node_id AS branch_id, n.type AS kind,
               coalesce(n.scope."function", -1) AS scope_id, c.node_id AS cond_id,
               c.node_id AS cond_first, c.node_id + c.descendant_count AS cond_last,
               n.node_id AS first, n.node_id + n.descendant_count AS last
        FROM cuda_nodes() n
        JOIN cuda_nodes() c ON c.fact_id = n.fact_id AND c.parent_id = n.node_id
          AND c.sibling_index = CASE n.type WHEN 'for_statement' THEN 3
                                            WHEN 'conditional_expression' THEN 2 ELSE 1 END
        WHERE n.type IN ('if_statement', 'elif_clause', 'while_statement', 'for_statement',
                         'conditional_expression')
          AND EXISTS (SELECT 1 FROM cuda_device_code() d
                      WHERE d.fact_id = n.fact_id AND d.function_id = n.scope."function")
        UNION ALL
        SELECT n.fact_id, n.path, n.node_id, n.type, coalesce(n.scope."function", -1), c.node_id,
               c.node_id, c.node_id + c.descendant_count, r.node_id,
               r.node_id + r.descendant_count
        FROM cuda_nodes() n
        JOIN cuda_nodes() c ON c.fact_id = n.fact_id AND c.parent_id = n.node_id
          AND c.sibling_index = 0
        JOIN cuda_nodes() r ON r.fact_id = n.fact_id AND r.parent_id = n.node_id
          AND r.sibling_index = 2
        WHERE n.type = 'boolean_operator'
          AND EXISTS (SELECT 1 FROM cuda_device_code() d
                      WHERE d.fact_id = n.fact_id AND d.function_id = n.scope."function")
    """
