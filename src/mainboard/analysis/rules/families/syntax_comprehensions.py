"""mcmr's ComprehensionFact, derived from sitting_duck's Python syntax trees.

One fact per Python file with two kinds of evidence: how many `for` clauses each comprehension or
generator expression has, and every loop that fills a fresh set by hand (`seen = set()` followed
by `for ...: seen.add(x)`, the add optionally guarded by one `if`) which a set comprehension
replaces without changing the program. mcmr reports a loop only when it proved the rewrite safe,
and so does this derivation: the pair sits in a function body outside every `except` handler and
class body, the loop is synchronous with no `else`, nothing in the loop's iterable, element or
condition can not move into a comprehension (`:=`, `await`, `yield`), the element does not read
the set, the loop's targets are read nowhere else in the function, `set` is not rebound, and the
function calls no `locals`, `vars`, `eval` or `exec`.

The set loops are found in the whole tree (a few dozen of 2 million nodes on a large project) and
every later stage reads `local`, the nodes of the files that hold one; a CTE the whole tree feeds
and several stages read is materialized by DuckDB, which spilled past the 2 GB cap.
"""

from ...catalog import Family, Function


class SyntaxIsLoad(Function):
    """Whether an identifier reads a variable, from its parent's type, its place among the
    parent's children, the parent's own place and the grandparent's type: the names a `def`,
    `class`, parameter, import, `global`, keyword or attribute states are no reads, and neither
    are the targets of an assignment, a `for` or a `with`."""

    parent_type: str
    place: int
    parent_place: int
    grand_type: str

    expression = r"""
        NOT CASE
          WHEN parent_type = 'attribute' THEN place > 0
          WHEN parent_type IN ('keyword_argument', 'default_parameter', 'typed_default_parameter')
            THEN place = 0
          WHEN parent_type IN ('function_definition', 'class_definition', 'parameters',
                               'lambda_parameters', 'typed_parameter', 'dotted_name',
                               'aliased_import', 'global_statement', 'nonlocal_statement',
                               'type_parameter', 'import_from_statement', 'as_pattern_target')
            THEN true
          WHEN parent_type IN ('assignment', 'augmented_assignment', 'named_expression')
            THEN place = 0
          WHEN parent_type = 'for_statement' THEN place = 1
          WHEN parent_type = 'for_in_clause' THEN place IN (1, 2)
          WHEN parent_type IN ('pattern_list', 'tuple_pattern', 'list_pattern',
                               'list_splat_pattern')
            THEN grand_type IN ('assignment', 'augmented_assignment', 'for_in_clause',
                                'pattern_list', 'tuple_pattern', 'list_pattern',
                                'list_splat_pattern', 'as_pattern_target')
                 OR (grand_type = 'for_statement' AND parent_place = 1)
          ELSE false END
    """


class Comprehension(Family):
    family = "ComprehensionFact"
    verified = True
    relations = {
        "facts": r"""
            SELECT row_number() OVER (ORDER BY f.path) - 1 AS fact_order,
                   'comprehensionfact:' || f.path AS fact_id, f.path,
                   1 AS start_line, 0 AS start_column, m.end_line, m.end_column,
                   'python' AS language
            FROM ast_files f JOIN ast_nodes m ON m.fact_id = f.fact_id AND m.type = 'module'
            WHERE f.language = 'python'
        """,
        "values": r"""
            SELECT f.fact_order, f.fact_id, 'loop_counts' AS relation,
                   row_number() OVER (PARTITION BY f.fact_id ORDER BY c.node_id) - 1 AS ordinal,
                   count(*)::BIGINT AS integer_value, f.fact_id AS parent_id
            FROM ast_nodes c JOIN comprehension_facts f ON f.path = c.path
            JOIN ast_nodes k ON k.fact_id = c.fact_id AND k.parent_id = c.node_id
                            AND k.type = 'for_in_clause'
            WHERE c.type IN ('list_comprehension', 'set_comprehension', 'dictionary_comprehension',
                             'generator_expression')
            GROUP BY f.fact_order, f.fact_id, c.fact_id, c.node_id
        """,
        "set_loop_candidates": r"""
            WITH python AS NOT MATERIALIZED (
              SELECT n.fact_id AS file, n.path, n.node_id AS id, n.parent_id AS pid,
                     n.sibling_index AS sib, n.type, n.name, n.children_count AS cc,
                     n.descendant_count AS dc, n.start_line AS sl, n.start_column AS sc,
                     n.end_line AS el, n.end_column AS ec
              FROM ast_nodes n SEMI JOIN ast_files f ON f.fact_id = n.fact_id
                                                    AND f.language = 'python'
            ), inits AS MATERIALIZED (
              SELECT st.file, st.path, st.id AS init, st.pid AS block, st.sib AS init_sib,
                     st.sl AS isl, st.sc AS isc, st.el AS iel, st.ec AS iec, l.name,
                     asg.cc > 3 AS annotated
              FROM python call
              JOIN python fn ON fn.file = call.file AND fn.pid = call.id AND fn.sib = 0
                            AND fn.type = 'identifier' AND fn.name = 'set'
              JOIN python args ON args.file = call.file AND args.pid = call.id
                              AND args.type = 'argument_list' AND args.cc = 2
              JOIN python asg ON asg.file = call.file AND asg.id = call.pid
                             AND asg.type = 'assignment' AND call.sib = asg.cc - 1
              JOIN python l ON l.file = asg.file AND l.pid = asg.id AND l.sib = 0
                           AND l.type = 'identifier'
              JOIN python st ON st.file = asg.file AND st.id = asg.pid
                            AND st.type = 'expression_statement'
              JOIN python blk ON blk.file = st.file AND blk.id = st.pid AND blk.type = 'block'
              WHERE call.type = 'call'
            ), loops AS MATERIALIZED (
              SELECT i.*, lp.id AS loop, lp.dc AS ldc, lp.sl AS lsl, lp.sc AS lsc, lp.el AS lel,
                     lp.ec AS lec
              FROM (SELECT i.file, i.init, arg_min(s.id, s.sib) AS next
                    FROM inits i JOIN python s ON s.file = i.file AND s.pid = i.block
                                              AND s.sib > i.init_sib AND s.type <> 'comment'
                    GROUP BY i.file, i.init) n
              JOIN inits i ON i.file = n.file AND i.init = n.init
              JOIN python lp ON lp.file = n.file AND lp.id = n.next AND lp.type = 'for_statement'
              JOIN python kw ON kw.file = lp.file AND kw.pid = lp.id AND kw.sib = 0
                            AND kw.type = 'for'
              WHERE NOT EXISTS (SELECT 1 FROM python e WHERE e.file = lp.file AND e.pid = lp.id
                                                         AND e.type = 'else_clause')
            ), local AS MATERIALIZED (
              SELECT * FROM python WHERE file IN (SELECT file FROM loops)
            ), shapes AS MATERIALIZED (
              SELECT l.*, t.id AS target, t.type AS target_type, t.sl AS tsl, t.sc AS tsc,
                     t.el AS tel, t.ec AS tec, it.id AS iterable, it.sl AS itsl, it.sc AS itsc,
                     it.el AS itel, it.ec AS itec, b.id AS body
              FROM loops l
              JOIN local t ON t.file = l.file AND t.pid = l.loop AND t.sib = 1
                           AND t.type IN ('identifier', 'pattern_list', 'tuple_pattern',
                                          'list_pattern', 'list_splat_pattern')
              JOIN local it ON it.file = l.file AND it.pid = l.loop AND it.sib = 3
              JOIN local b ON b.file = l.file AND b.pid = l.loop AND b.sib = 4
                           AND b.type = 'block'
            ), statements AS MATERIALIZED (
              SELECT s.*, a.id AS arm, a.type AS arm_type, c.id AS test, c.sl AS csl, c.sc AS csc,
                     c.el AS cel, c.ec AS cec, (a.type = 'if_statement')::INTEGER AS conditions
              FROM shapes s
              JOIN (SELECT file, pid, count(*) AS n, any_value(id) AS id, any_value(type) AS type
                    FROM local WHERE type <> 'comment' GROUP BY file, pid) a
                ON a.file = s.file AND a.pid = s.body AND a.n = 1
              LEFT JOIN local c ON c.file = s.file AND c.pid = a.id AND c.sib = 1
                                AND a.type = 'if_statement'
              WHERE (a.type = 'expression_statement')
                 OR (a.type = 'if_statement'
                     AND NOT EXISTS (SELECT 1 FROM local e
                                     WHERE e.file = s.file AND e.pid = a.id
                                       AND e.type IN ('elif_clause', 'else_clause')))
            ), bodies AS (
              SELECT file, pid, count(*) AS n, any_value(id) AS id, any_value(type) AS type
              FROM local WHERE type <> 'comment' GROUP BY file, pid
            ), adds AS MATERIALIZED (
              SELECT s.*, x.id AS stmt
              FROM statements s
              JOIN local cb ON cb.file = s.file AND cb.pid = s.arm AND cb.type = 'block'
              JOIN bodies x ON x.file = cb.file AND x.pid = cb.id AND x.n = 1
                           AND x.type = 'expression_statement'
              WHERE s.arm_type = 'if_statement'
              UNION ALL
              SELECT s.*, s.arm FROM statements s WHERE s.arm_type = 'expression_statement'
            ), arguments AS (
              SELECT file, pid, count(*) AS n, any_value(id) AS id, any_value(type) AS type
              FROM local WHERE type NOT IN ('(', ')', ',', 'comment') GROUP BY file, pid
            ), calls AS MATERIALIZED (
              SELECT a.*, g.id AS element, g.sl AS esl, g.sc AS esc, g.el AS eel, g.ec AS eec,
                     m.el AS mel, m.ec AS mec
              FROM adds a
              JOIN local m ON m.file = a.file AND m.id = a.stmt
              JOIN local call ON call.file = a.file AND call.pid = a.stmt AND call.type = 'call'
              JOIN local attr ON attr.file = a.file AND attr.pid = call.id AND attr.sib = 0
                              AND attr.type = 'attribute' AND attr.cc = 2
              JOIN local recv ON recv.file = a.file AND recv.pid = attr.id AND recv.sib = 0
                              AND recv.type = 'identifier' AND recv.name = a.name
              JOIN local meth ON meth.file = a.file AND meth.pid = attr.id AND meth.sib = 1
                              AND meth.type = 'identifier' AND meth.name = 'add'
              JOIN local al ON al.file = a.file AND al.pid = call.id AND al.sib = 1
                            AND al.type = 'argument_list'
              JOIN arguments r ON r.file = a.file AND r.pid = al.id AND r.n = 1
                              AND r.type NOT IN ('keyword_argument', 'dictionary_splat')
              JOIN local g ON g.file = a.file AND g.id = r.id
            ), owners AS (
              SELECT c.file, c.init, arg_max(anc.id, anc.id) AS owner,
                     arg_max(anc.type, anc.id) AS owner_type, arg_max(anc.dc, anc.id) AS owner_dc
              FROM calls c JOIN local anc ON anc.file = c.file AND anc.id < c.init
                                          AND c.init <= anc.id + anc.dc
                                          AND anc.type IN ('function_definition',
                                                           'class_definition', 'except_clause',
                                                           'except_group_clause')
              GROUP BY c.file, c.init
            ), regions AS (
              SELECT c.file, c.init, 'iterable' AS part, c.iterable AS lo, c.iterable + i.dc AS hi
              FROM calls c JOIN local i ON i.file = c.file AND i.id = c.iterable
              UNION ALL
              SELECT c.file, c.init, 'element', c.element, c.element + e.dc
              FROM calls c JOIN local e ON e.file = c.file AND e.id = c.element
              UNION ALL
              SELECT c.file, c.init, 'test', c.test, c.test + t.dc
              FROM calls c JOIN local t ON t.file = c.file AND t.id = c.test
              UNION ALL
              SELECT c.file, c.init, 'target', c.target, c.target + t.dc
              FROM calls c JOIN local t ON t.file = c.file AND t.id = c.target
              UNION ALL
              SELECT o.file, o.init, 'function', o.owner, o.owner + o.owner_dc FROM owners o
            ), hazards AS (
              SELECT DISTINCT r.file, r.init
              FROM regions r JOIN local h ON h.file = r.file AND h.id BETWEEN r.lo AND r.hi
              WHERE (r.part IN ('iterable', 'element', 'test')
                     AND h.type IN ('named_expression', 'await', 'yield'))
                 OR (r.part = 'target'
                     AND h.type NOT IN ('identifier', 'pattern_list', 'tuple_pattern',
                                        'list_pattern', 'list_splat_pattern', ',', '(', ')', '[',
                                        ']', '*', 'comment'))
            ), reads AS (
              SELECT r.file, r.init, r.part, x.name, x.id
              FROM regions r
              JOIN local x ON x.file = r.file AND x.id BETWEEN r.lo AND r.hi
                           AND x.type = 'identifier'
              JOIN local p ON p.file = x.file AND p.id = x.pid
              LEFT JOIN local g ON g.file = p.file AND g.id = p.pid
              WHERE r.part IN ('element', 'function', 'target')
                AND (r.part = 'target' OR syntax_is_load(p.type, x.sib, p.sib, g.type))
            ), unsafe AS (
              SELECT c.file, c.init
              FROM calls c
              JOIN reads e ON e.file = c.file AND e.init = c.init AND e.part = 'element'
                          AND e.name = c.name
              UNION
              SELECT c.file, c.init
              FROM calls c
              JOIN (SELECT file, init, name FROM reads WHERE part = 'target') t
                ON t.file = c.file AND t.init = c.init
              JOIN reads f ON f.file = c.file AND f.init = c.init AND f.part = 'function'
                          AND f.name = t.name AND NOT (f.id BETWEEN c.loop AND c.loop + c.ldc)
              UNION
              SELECT c.file, c.init
              FROM calls c JOIN (SELECT file, init, name FROM reads WHERE part = 'target') t
                ON t.file = c.file AND t.init = c.init AND t.name = c.name
              UNION
              SELECT c.file, c.init
              FROM calls c JOIN owners o ON o.file = c.file AND o.init = c.init
              JOIN local x ON x.file = c.file AND x.id BETWEEN o.owner AND o.owner + o.owner_dc
              JOIN local p ON p.file = x.file AND p.id = x.pid
              WHERE (x.type = 'identifier'
                     AND x.name IN ('locals', 'globals', 'vars', 'eval', 'exec')
                     AND p.type = 'call' AND x.sib = 0)
                 OR (x.type = 'identifier' AND x.name = c.name
                     AND p.type IN ('global_statement', 'nonlocal_statement'))
            ), shadows AS (
              SELECT x.file, coalesce(arg_max(f.id, f.id), -1) AS scope,
                     coalesce(arg_max(f.id + f.dc, f.id), -1) AS scope_end
              FROM local x
              JOIN local p ON p.file = x.file AND p.id = x.pid
              LEFT JOIN local g ON g.file = p.file AND g.id = p.pid
              LEFT JOIN local f ON f.file = x.file AND f.type = 'function_definition'
                                AND f.id < x.id AND x.id <= f.id + f.dc
              WHERE x.type = 'identifier' AND x.name = 'set'
                AND p.type NOT IN ('attribute', 'keyword_argument', 'call', 'argument_list')
                AND syntax_is_load(p.type, x.sib, p.sib, g.type) IS FALSE
              GROUP BY x.file, x.id
            ), candidates AS (
              SELECT c.*, o.owner
              FROM calls c JOIN owners o ON o.file = c.file AND o.init = c.init
              WHERE o.owner_type = 'function_definition'
                AND NOT EXISTS (SELECT 1 FROM hazards h WHERE h.file = c.file AND h.init = c.init)
                AND NOT EXISTS (SELECT 1 FROM unsafe u WHERE u.file = c.file AND u.init = c.init)
                AND NOT EXISTS (SELECT 1 FROM shadows s
                                WHERE s.file = c.file AND (s.scope = -1 OR (s.scope <= o.owner
                                      AND o.owner <= s.scope_end)))
            ), written AS (
              SELECT c.*, f.fact_order, f.fact_id, s.content,
                     row_number() OVER (PARTITION BY c.file ORDER BY c.init) - 1 AS ordinal
              FROM candidates c JOIN comprehension_facts f ON f.path = c.path
              JOIN source_files s ON s.path = c.path
            )
            SELECT fact_order, fact_id, fact_id AS parent_id,
                   fact_id || '/set_loop_candidates:' || ordinal AS record_id, ordinal, name,
                   true AS has_unshadowed_set_initialization, true AS loop_is_synchronous,
                   true AS only_effect_is_add, conditions AS conditional_count,
                   false AS has_else,
                   CASE WHEN annotated
                          OR contains(syntax_slice(content, isl, isc, mel + 1, 0), '#') THEN NULL
                        ELSE syntax_slice(content, esl, esc, eel, eec) END AS element_text,
                   syntax_slice(content, tsl, tsc, tel, tec) AS target_text,
                   syntax_slice(content, itsl, itsc, itel, itec) AS iterable_text,
                   CASE WHEN test IS NOT NULL THEN syntax_slice(content, csl, csc, cel, cec) END
                     AS condition_text,
                   isl AS initialization_span_start_line, isc AS initialization_span_start_column,
                   iel AS initialization_span_end_line, iec AS initialization_span_end_column,
                   lsl AS loop_span_start_line, lsc AS loop_span_start_column,
                   mel AS loop_span_end_line, mec AS loop_span_end_column
            FROM written
        """,
        "set_loop_candidates_conditions": """
            SELECT fact_order, fact_id, record_id AS parent_id, 0 AS ordinal,
                   condition_text AS text
            FROM comprehension_set_loop_candidates WHERE condition_text IS NOT NULL
        """,
    }
