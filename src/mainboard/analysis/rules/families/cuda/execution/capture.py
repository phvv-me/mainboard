"""Code that runs while a CUDA graph is being captured, where a wait, a read of a device value or
a pageable host-to-device copy is not allowed."""

from ..base import CudaRelation


class CudaCaptureRegions(CudaRelation):
    """The spans of code that run under capture: the body passed to a `capture(...)` call, a
    `with torch.cuda.graph(...)` block, the code between `begin_building()`/`begin_capture()`
    and `end_building()`/`end_capture()`. `first`..`last` are node ids in `scope_id`'s function."""

    stored = True
    query = """
        SELECT c.fact_id, c.path, c.scope_id, l.node_id AS first, l.node_id + l.descendant_count AS last
        FROM cuda_calls() c
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
        JOIN cuda_nodes() l ON l.fact_id = a.fact_id AND l.node_id = a.value_id AND l.type = 'lambda'
        WHERE (c.method = 'capture' AND EXISTS (SELECT 1 FROM cuda_arguments() k
                                                WHERE k.fact_id = c.fact_id AND k.call_id = c.node_id
                                                  AND k.keyword = 'stream'))
           OR c.qualified_name IN ('torch.cuda.make_graphed_callables')
        UNION ALL
        SELECT w.fact_id, w.path, coalesce(w.scope."function", -1), w.node_id,
               w.node_id + w.descendant_count
        FROM cuda_nodes() w
        JOIN cuda_nodes() i ON i.fact_id = w.fact_id AND i.node_id BETWEEN w.node_id + 1
          AND w.node_id + w.descendant_count AND i.type = 'with_item'
        JOIN cuda_calls() c ON c.fact_id = i.fact_id AND c.node_id BETWEEN i.node_id
          AND i.node_id + i.descendant_count
        WHERE w.type = 'with_statement' AND c.qualified_name IN ('torch.cuda.graph')
        UNION ALL
        SELECT b.fact_id, b.path, b.scope_id, b.node_id, coalesce(min(e.node_id), b.node_id)
        FROM cuda_calls() b
        LEFT JOIN cuda_calls() e ON e.fact_id = b.fact_id AND e.scope_id = b.scope_id
          AND e.node_id > b.node_id AND e.method IN ('end_building', 'end_capture')
        WHERE b.method IN ('begin_building', 'begin_capture')
        GROUP BY b.fact_id, b.path, b.scope_id, b.node_id
    """


class CudaCaptured(CudaRelation):
    """The functions that run under capture: those a call inside a capture region reaches,
    directly or through up to three calls."""

    stored = True
    query = """
        WITH RECURSIVE reach(fact_id, function_id, hops, region_path, region_line)
          USING KEY (fact_id, function_id) AS (
          SELECT t.definition_fact, t.definition_id, 1, c.path, c.start_line
          FROM cuda_capture_regions() r
          JOIN cuda_calls() c ON c.fact_id = r.fact_id AND c.node_id BETWEEN r.first AND r.last
          JOIN cuda_resolved_calls() t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
            AND t.kind = 'function' AND t.via <> 'override'
          UNION
          SELECT t.definition_fact, t.definition_id, r.hops + 1, r.region_path, r.region_line
          FROM reach r
          JOIN cuda_calls() c ON c.fact_id = r.fact_id AND c.scope_id = r.function_id
          JOIN cuda_resolved_calls() t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
            AND t.kind = 'function' AND t.via <> 'override'
          WHERE r.hops < 3 AND NOT EXISTS (
            SELECT 1 FROM recurring.reach p
            WHERE p.fact_id = t.definition_fact AND p.function_id = t.definition_id)
        )
        SELECT * FROM reach
    """


class CudaUncaptured(CudaRelation):
    """The spans of code a branch has already taken out of capture: the body of `if recording is
    None:` (or `if not recording:`), and every `elif` and `else` after `if recording is not
    None:` (or `if recording:`), where `recording` is a name that says it holds the capture
    (`recording`, `capture`, `graph`, `builder` as a part of it, in any case). Nothing there runs
    while a graph is recorded, whatever function holds it. `first`..`last` are node ids."""

    stored = True
    query = r"""
        WITH tested AS (
          SELECT n.fact_id, n.node_id, n.sibling_index, n.type,
                 CASE WHEN n.type = 'if_statement' THEN n.node_id ELSE n.parent_id END AS chain,
                 trim(syntax_slice(s.content, c.start_line, c.start_column, c.end_line,
                                   c.end_column)) AS test
          FROM cuda_nodes() n
          JOIN cuda_nodes() c ON c.fact_id = n.fact_id AND c.parent_id = n.node_id
            AND c.sibling_index = 1
          JOIN source_files s ON s.path = n.path
          WHERE n.type IN ('if_statement', 'elif_clause')
        ), judged AS (
          SELECT *, CASE
                 WHEN regexp_matches(test, '^(not\s+[\w.]+|[\w.]+\s+(is|==)\s+None)$') THEN 'absent'
                 WHEN regexp_matches(test, '^([\w.]+\s+(is\s+not|!=)\s+None|[\w.]+)$')
                 THEN 'present' END AS verdict,
                 regexp_extract(test, '([A-Za-z_]\w*)(\s+(is|==|is\s+not|!=)\s+None)?$', 1) AS name
          FROM tested
        )
        SELECT j.fact_id, b.node_id AS first, b.node_id + b.descendant_count AS last
        FROM judged j
        JOIN cuda_nodes() b ON b.fact_id = j.fact_id AND b.parent_id = j.node_id
          AND b.type = 'block' AND b.sibling_index = (
            SELECT min(k.sibling_index) FROM cuda_nodes() k
            WHERE k.fact_id = j.fact_id AND k.parent_id = j.node_id AND k.type = 'block')
        WHERE j.verdict = 'absent' AND regexp_matches(j.name, '(?i)record|captur|graph|build')
        UNION ALL
        SELECT j.fact_id, min(a.node_id) AS first, x.node_id + x.descendant_count AS last
        FROM judged j
        JOIN cuda_nodes() x ON x.fact_id = j.fact_id AND x.node_id = j.chain
        JOIN cuda_nodes() a ON a.fact_id = j.fact_id AND a.parent_id = j.chain
          AND a.type IN ('elif_clause', 'else_clause')
          AND (j.type = 'if_statement' OR a.sibling_index > j.sibling_index)
        WHERE j.verdict = 'present' AND regexp_matches(j.name, '(?i)record|captur|graph|build')
        GROUP BY j.fact_id, j.node_id, x.node_id, x.descendant_count
    """
