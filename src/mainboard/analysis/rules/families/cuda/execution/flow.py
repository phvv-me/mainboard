"""Where code runs more than once: the loops of a function, the calls made inside them, and the
functions those calls reach, which run once per iteration of someone else's loop.

Syntax tree ids run in document order and a node's subtree is the ids after it for its
`descendant_count`, so "inside" is a range test.
"""

from ..base import CudaRelation


class CudaLoops(CudaRelation):
    """The regions a loop repeats: a `for` body, all of a `while` (its test reruns), a
    comprehension without its first iterable (evaluated once). A loop over a literal of up to ten
    elements (`for column in (keys, values)`) or `range(8)` or less repeats too few times to be one,
    and neither is a retry loop, a `while` whose test is a call and whose body repeats a call made
    before it (`x = run(); while x.settle(): x = run()`): its body is the exceptional path.
    `scope_id` is the function the loop is in; nodes between `first` and `last` run once per
    iteration."""

    stored = True
    query = """
        SELECT n.fact_id, n.node_id AS loop_id, 'for' AS kind, coalesce(n.scope."function", -1) AS scope_id,
               b.node_id AS first, b.node_id + b.descendant_count AS last
        FROM cuda_nodes() n JOIN cuda_nodes() b ON b.fact_id = n.fact_id AND b.parent_id = n.node_id
          AND b.type = 'block'
        JOIN cuda_nodes() it ON it.fact_id = n.fact_id AND it.parent_id = n.node_id
          AND it.sibling_index = 3
        WHERE n.type = 'for_statement' AND NOT (it.type IN ('tuple', 'list', 'set') AND it.children_count <= 12)
          AND NOT (it.type = 'call' AND it.name = 'range' AND EXISTS (
                     SELECT 1 FROM cuda_nodes() al
                     JOIN cuda_nodes() k ON k.fact_id = al.fact_id AND k.parent_id = al.node_id
                       AND k.type = 'integer'
                     WHERE al.fact_id = it.fact_id AND al.parent_id = it.node_id
                       AND al.type = 'argument_list' AND al.children_count = 3
                       AND try_cast(k.name AS INTEGER) <= 8))
        UNION ALL
        SELECT n.fact_id, n.node_id, 'while', coalesce(n.scope."function", -1), n.node_id + 1,
               n.node_id + n.descendant_count
        FROM cuda_nodes() n
        JOIN cuda_nodes() t ON t.fact_id = n.fact_id AND t.parent_id = n.node_id
          AND t.sibling_index = 1
        WHERE n.type = 'while_statement'
          AND NOT (t.type = 'call' AND EXISTS (
                     SELECT 1 FROM cuda_calls() again
                     JOIN cuda_calls() first ON first.fact_id = again.fact_id
                       AND first.scope_id = again.scope_id AND first.callee_text = again.callee_text
                       AND first.node_id < n.node_id
                     WHERE again.fact_id = n.fact_id
                       AND again.node_id BETWEEN t.node_id + t.descendant_count + 1
                           AND n.node_id + n.descendant_count))
        UNION ALL
        SELECT n.fact_id, n.node_id, 'comprehension', coalesce(n.scope."function", -1),
               n.node_id + 1, i.node_id - 1
        FROM cuda_nodes() n
        JOIN cuda_nodes() f ON f.fact_id = n.fact_id AND f.parent_id = n.node_id
          AND f.type = 'for_in_clause'
        JOIN cuda_nodes() i ON i.fact_id = f.fact_id AND i.parent_id = f.node_id AND i.sibling_index = 3
        WHERE n.type IN ('list_comprehension', 'set_comprehension', 'dictionary_comprehension',
                         'generator_expression')
          AND NOT (i.type IN ('tuple', 'list', 'set') AND i.children_count <= 12)
          AND NOT (i.type = 'call' AND i.name = 'range' AND EXISTS (
                     SELECT 1 FROM cuda_nodes() al
                     JOIN cuda_nodes() k ON k.fact_id = al.fact_id AND k.parent_id = al.node_id
                       AND k.type = 'integer'
                     WHERE al.fact_id = i.fact_id AND al.parent_id = i.node_id
                       AND al.type = 'argument_list' AND al.children_count = 3
                       AND try_cast(k.name AS INTEGER) <= 8))
          AND f.sibling_index = (SELECT min(g.sibling_index) FROM cuda_nodes() g
                                 WHERE g.fact_id = n.fact_id AND g.parent_id = n.node_id
                                   AND g.type = 'for_in_clause')
        UNION ALL
        SELECT n.fact_id, n.node_id, 'comprehension', coalesce(n.scope."function", -1),
               i.node_id + i.descendant_count + 1, n.node_id + n.descendant_count
        FROM cuda_nodes() n
        JOIN cuda_nodes() f ON f.fact_id = n.fact_id AND f.parent_id = n.node_id
          AND f.type = 'for_in_clause'
        JOIN cuda_nodes() i ON i.fact_id = f.fact_id AND i.parent_id = f.node_id AND i.sibling_index = 3
        WHERE n.type IN ('list_comprehension', 'set_comprehension', 'dictionary_comprehension',
                         'generator_expression')
          AND NOT (i.type IN ('tuple', 'list', 'set') AND i.children_count <= 12)
          AND NOT (i.type = 'call' AND i.name = 'range' AND EXISTS (
                     SELECT 1 FROM cuda_nodes() al
                     JOIN cuda_nodes() k ON k.fact_id = al.fact_id AND k.parent_id = al.node_id
                       AND k.type = 'integer'
                     WHERE al.fact_id = i.fact_id AND al.parent_id = i.node_id
                       AND al.type = 'argument_list' AND al.children_count = 3
                       AND try_cast(k.name AS INTEGER) <= 8))
          AND f.sibling_index = (SELECT min(g.sibling_index) FROM cuda_nodes() g
                                 WHERE g.fact_id = n.fact_id AND g.parent_id = n.node_id
                                   AND g.type = 'for_in_clause')
    """


class CudaLoopCalls(CudaRelation):
    """Each call made once per iteration of a loop of its own function, with the innermost loop."""

    stored = True
    query = """
        SELECT c.*, l.loop_id, l.kind AS loop_kind
        FROM cuda_calls() c
        JOIN cuda_loops() l ON l.fact_id = c.fact_id AND l.scope_id = c.scope_id
          AND c.node_id BETWEEN l.first AND l.last
        QUALIFY row_number() OVER (PARTITION BY c.fact_id, c.node_id ORDER BY l.loop_id DESC) = 1
    """


class CudaHot(CudaRelation):
    """The functions that run once per iteration of a loop somewhere: those a loop's calls reach
    directly (`hops` 1) and those they call in turn (up to seven), one row per loop call that
    reaches them (`origin_*`), since whether that loop counts depends on where it is."""

    stored = True
    query = """
        WITH RECURSIVE reach(fact_id, function_id, origin_fact, origin_node, hops, origin_path,
                             origin_line, origin_call)
          USING KEY (fact_id, function_id, origin_fact, origin_node) AS (
          SELECT d.definition_fact, d.definition_id, c.fact_id, c.node_id, 1, c.path, c.start_line,
                 c.shown
          FROM cuda_loop_calls() c
          JOIN cuda_resolved_calls() d ON d.fact_id = c.fact_id AND d.node_id = c.node_id
            AND d.kind = 'function'
          UNION
          SELECT d.definition_fact, d.definition_id, r.origin_fact, r.origin_node, r.hops + 1,
                 r.origin_path, r.origin_line, r.origin_call
          FROM reach r
          JOIN cuda_calls() c ON c.fact_id = r.fact_id AND c.scope_id = r.function_id
          JOIN cuda_resolved_calls() d ON d.fact_id = c.fact_id AND d.node_id = c.node_id
            AND d.kind = 'function'
          WHERE r.hops < 7 AND NOT EXISTS (
            SELECT 1 FROM recurring.reach p
            WHERE p.fact_id = d.definition_fact AND p.function_id = d.definition_id
              AND p.origin_fact = r.origin_fact AND p.origin_node = r.origin_node)
        )
        SELECT * FROM reach
    """
