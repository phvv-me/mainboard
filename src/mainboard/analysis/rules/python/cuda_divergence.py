"""PY-CUDA rules about threads that do not reach the same code."""

from ...catalog import Example, Rule
from .cuda_paths import TESTS


class DivergentCollective(Rule, id="PY-CUDA0019", applicability="review"):
    """A barrier, warp collective or block-wide primitive (`cuda.syncthreads()`, `cuda.syncwarp()`,
    a shuffle or vote, patos's `warp.*` and `block.over(...)` operations) reached under a branch
    whose condition differs between the threads it synchronizes. Block-level calls need a
    condition every thread of the block agrees on (scalar parameters, constants, block indices, a
    value a block reduction returned); warp-level calls one every lane of the warp agrees on (those
    and warp indices, the result of a vote). A `return` under such a condition, or a `break` or
    `continue` in the same loop, before the collective leaves it to the threads that stayed.
    Arithmetic that makes a warp-wide value (`tid // 32`, `tid >> 5`, `tid - tid % 32`, a
    division by `cuda.warpsize` or by a constant bound to a multiple of 32) is uniform across a
    warp. Triton has no such hazard: its branches are taken per program, not per thread."""

    exempt: tuple[str, ...] = TESTS

    query = r"""
        WITH conditions AS (
          SELECT b.fact_id, b.branch_id, b.kind, b.scope_id, b.cond_id, b.cond_first, b.cond_last,
                 b.first, b.last, v.level, v.why
          FROM cuda_branches() b
          JOIN cuda_levels() v ON v.key = 'b:' || b.fact_id || ':' || b.branch_id
        ), exits AS (
          SELECT n.fact_id, n.node_id, n.scope."function" AS scope_id,
                 replace(n.type, '_statement', '') AS kind,
                 (SELECT max(l.node_id) FROM cuda_nodes() l
                  WHERE l.fact_id = n.fact_id AND l.type IN ('for_statement', 'while_statement')
                    AND n.node_id BETWEEN l.node_id AND l.node_id + l.descendant_count) AS loop_id
          FROM cuda_nodes() n
          WHERE n.type IN ('return_statement', 'break_statement', 'continue_statement')
            AND n.scope."function" IS NOT NULL
        ), reasons AS (
          SELECT c.fact_id, c.node_id, b.branch_id, b.cond_id, b.level, b.why, c.requires,
                 'sits under' AS placement, 0 AS rank
          FROM cuda_collective_sites() c
          JOIN conditions b ON b.fact_id = c.fact_id AND b.scope_id = c.scope_id
            AND c.node_id BETWEEN b.first AND b.last
            AND NOT c.node_id BETWEEN b.cond_first AND b.cond_last
          UNION ALL
          SELECT c.fact_id, c.node_id, b.branch_id, b.cond_id, b.level, b.why, c.requires,
                 'comes after a `' || e.kind || '` under', 1
          FROM cuda_collective_sites() c
          JOIN exits e ON e.fact_id = c.fact_id AND e.scope_id = c.scope_id AND e.node_id < c.node_id
          JOIN conditions b ON b.fact_id = e.fact_id AND b.scope_id = e.scope_id
            AND b.kind IN ('if_statement', 'elif_clause')
            AND e.node_id BETWEEN b.first AND b.last
            AND NOT e.node_id BETWEEN b.cond_first AND b.cond_last
            AND c.node_id > b.last
          LEFT JOIN cuda_nodes() lp ON lp.fact_id = e.fact_id AND lp.node_id = e.loop_id
          WHERE e.kind = 'return'
             OR (b.first > lp.node_id AND c.node_id <= lp.node_id + lp.descendant_count)
        ), divergent AS (
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown, c.via,
                 c.requires, r.level, r.why, r.placement,
                 regexp_replace(syntax_slice(s.content, t.start_line, t.start_column, t.end_line,
                                             t.end_column), '\s+', ' ', 'g') AS condition
          FROM reasons r
          JOIN cuda_collective_sites() c ON c.fact_id = r.fact_id AND c.node_id = r.node_id
          JOIN cuda_nodes() t ON t.fact_id = r.fact_id AND t.node_id = r.cond_id
          JOIN source_files s ON s.path = c.path
          WHERE r.level > CASE c.requires WHEN 'block' THEN 0 ELSE 1 END
          QUALIFY row_number() OVER (PARTITION BY c.fact_id, c.node_id
                                     ORDER BY r.rank, r.branch_id DESC) = 1
        )
        SELECT d.path, d.start_line, d.start_column, d.end_line, d.end_column, d.shown,
               d.condition, d.why, d.placement,
               CASE WHEN d.placement = 'sits under' THEN 'outside the branch'
                    ELSE 'before the early exit' END AS outside,
               CASE d.requires WHEN 'block' THEN 'every thread of the block'
                               ELSE 'every lane of the warp' END AS group_name,
               CASE d.requires WHEN 'block' THEN 'the threads of the block'
                               ELSE 'the lanes of the warp' END AS members,
               CASE WHEN d.via IS NULL THEN 'waits for' ELSE 'calls `' || d.via || '`, which waits for' END
                 AS action,
               CASE d.requires WHEN 'block' THEN 'a value every thread of the block shares (a parameter, '
                                    || 'a block index, a block reduction)'
                               ELSE 'a value every lane of the warp shares (a warp index, a vote)'
               END AS uniform
        FROM divergent d
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(d.path, p))), false)
    """
    message = (
        "`{shown}` {action} {group_name} but {placement} `{condition}`, which differs between "
        "{members} (it depends on `{why}`): the threads that skip it never arrive, so the others "
        "hang or read what the skipped ones never wrote. Fix: keep the collective {outside} and "
        "guard only the work around it, or branch on {uniform}"
    )
    examples = (
        Example(
            bad=(
                "from numba import cuda\n\n\n@cuda.jit\ndef kernel(x, n):\n"
                "    i = cuda.grid(1)\n    if i < n:\n        x[i] = 1\n"
                "        cuda.syncthreads()\n"
            ),
            good=(
                "from numba import cuda\n\n\n@cuda.jit\ndef kernel(x, n):\n"
                "    i = cuda.grid(1)\n    if i < n:\n        x[i] = 1\n"
                "    cuda.syncthreads()\n"
            ),
        ),
        Example(
            bad=(
                "from patos.cuda.typed import Per, cuda, kernel, thread_in_block\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef fill(x: int) -> None:\n"
                "    if thread_in_block() < 64:\n        cuda.syncthreads()\n"
            ),
            good=(
                "from patos.cuda.typed import Per, cuda, kernel, thread_in_block\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef fill(x: int) -> None:\n"
                "    if x > 64:\n        cuda.syncthreads()\n"
            ),
        ),
        Example(
            bad=(
                "from patos.cuda.primitives import warp\n"
                "from patos.cuda.typed import Per, cuda, kernel, thread_in_block\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef pack(x: int) -> None:\n"
                "    emitted = warp.ballot_prefix(thread_in_block() < x)\n"
                "    if emitted.before == 0:\n        cuda.syncwarp()\n"
            ),
            good=(
                "from patos.cuda.primitives import warp\n"
                "from patos.cuda.typed import Per, cuda, kernel, thread_in_block\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef pack(x: int) -> None:\n"
                "    emitted = warp.ballot_prefix(thread_in_block() < x)\n"
                "    if emitted.total == 0:\n        cuda.syncwarp()\n"
            ),
        ),
        Example(
            bad=(
                "from patos.cuda.primitives import warp\n"
                "from patos.cuda.typed import Per, cuda, kernel\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef scan(x: int) -> None:\n"
                "    upto = warp.inclusive_sum(x)\n"
                "    if upto > 64:\n        cuda.syncwarp()\n"
            ),
            good=(
                "from patos.cuda.primitives import warp\n"
                "from patos.cuda.typed import Per, cuda, kernel\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef scan(x: int) -> None:\n"
                "    total = warp.sum(x)\n"
                "    if total > 64:\n        cuda.syncwarp()\n"
            ),
        ),
        Example(
            bad=(
                "from patos.cuda.typed import Per, Struct, cuda, device, i32, kernel, thread_in_block\n\n\n"
                "class Pair(Struct):\n    each: i32\n    every: i32\n\n\n"
                "@device\ndef pair(x: i32) -> Pair:\n    return Pair(thread_in_block(), x)\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef run(x: int) -> None:\n"
                "    both = pair(x)\n    if both.each < 64:\n        cuda.syncthreads()\n"
            ),
            good=(
                "from patos.cuda.typed import Per, Struct, cuda, device, i32, kernel, thread_in_block\n\n\n"
                "class Pair(Struct):\n    each: i32\n    every: i32\n\n\n"
                "@device\ndef pair(x: i32) -> Pair:\n    return Pair(thread_in_block(), x)\n\n\n"
                "@kernel(per=Per.BLOCK, threads=128)\ndef run(x: int) -> None:\n"
                "    both = pair(x)\n    if both.every < 64:\n        cuda.syncthreads()\n"
            ),
        ),
    )
