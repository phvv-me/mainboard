"""mcmr's CollectionFact, derived from sitting_duck's Python syntax trees.

One fact per Python file with two kinds of evidence about literals a function binds with a plain
`name = [...]` / `name = (...)` statement (module and class bodies are not candidates, since their
readers are other files):

- `local_collections`: the literal is bound once in its function and every read of the name is
  an iteration (`for x in name`, a comprehension over it) or a membership test (`x in name`), which
  is what a representation change needs to prove (`PY-COLL0003` swaps list, tuple and set);
- `pair_sequences`: a literal of 2-tuples, with whether the name is bound once, its keys are unique
  literals of one kind, and every read is a loop returning the value of the key it matched.

Reads are the identifiers of the function that read the name, nested functions and classes
excluded (they are scopes of their own); stores are assignment, augmented assignment, `for` and
`del` targets. mcmr's walker never visits `elif` tests, `assert` messages, `match` subjects,
`yield from` or slices, so reads there do not count for it and do count here.
"""

from ....catalog import Family, Function


class SyntaxIsStore(Function):
    """Whether an identifier is a target of an assignment, an augmented assignment, a `for` or a
    `del`, from its parent's type, its place among the parent's children, the parent's own place
    and the grandparent's type."""

    parent_type: str
    place: int
    parent_place: int
    grand_type: str

    expression = r"""
        CASE
          WHEN parent_type IN ('assignment', 'augmented_assignment') THEN place = 0
          WHEN parent_type = 'for_statement' THEN place = 1
          WHEN parent_type = 'delete_statement' THEN true
          WHEN parent_type IN ('pattern_list', 'tuple_pattern', 'list_pattern',
                               'list_splat_pattern')
            THEN (grand_type IN ('assignment', 'augmented_assignment') AND parent_place = 0)
                 OR (grand_type = 'for_statement' AND parent_place = 1)
                 OR grand_type IN ('pattern_list', 'tuple_pattern', 'list_pattern',
                                   'list_splat_pattern')
          ELSE false END
    """


class Collection(Family):
    family = "CollectionFact"
    relations = {
        "facts": r"""
            SELECT row_number() OVER (ORDER BY f.path) - 1 AS fact_order,
                   'collectionfact:' || f.path AS fact_id, f.path,
                   1 AS start_line, 0 AS start_column, m.end_line, m.end_column,
                   'python' AS language
            FROM ast_files f JOIN ast_nodes m ON m.fact_id = f.fact_id AND m.type = 'module'
            WHERE f.language = 'python'
        """,
        "literals": r"""
            WITH python AS (
              SELECT n.fact_id AS file, n.path, n.node_id AS id, n.parent_id AS pid,
                     n.sibling_index AS sib, n.type, n.name, n.children_count AS cc,
                     n.descendant_count AS dc
              FROM ast_nodes n SEMI JOIN ast_files f ON f.fact_id = n.fact_id
                                                    AND f.language = 'python'
              WHERE n.type IN ('assignment', 'identifier', 'list', 'tuple', 'expression_list',
                               'expression_statement', 'function_definition', 'class_definition')
            ), assigned AS (
              SELECT st.file, st.path, st.id AS stmt, nm.name, v.id AS value, v.dc AS value_dc,
                     CASE v.type WHEN 'list' THEN 'list' ELSE 'tuple' END AS kind
              FROM python asg
              JOIN python nm ON nm.file = asg.file AND nm.pid = asg.id AND nm.sib = 0
                            AND nm.type = 'identifier'
              JOIN python v ON v.file = asg.file AND v.pid = asg.id AND v.sib = 2
                           AND v.type IN ('list', 'tuple', 'expression_list')
              JOIN python st ON st.file = asg.file AND st.id = asg.pid
                            AND st.type = 'expression_statement'
              WHERE asg.type = 'assignment' AND asg.cc = 3
            ), scoped AS (
              SELECT a.*, arg_max(s.id, s.id) AS owner, arg_max(s.dc, s.id) AS owner_dc,
                     arg_max(s.type, s.id) AS owner_type
              FROM assigned a JOIN python s ON s.file = a.file AND s.id < a.stmt
                                           AND a.stmt <= s.id + s.dc
                                           AND s.type IN ('function_definition',
                                                          'class_definition')
              GROUP BY ALL
            )
            SELECT f.fact_order, f.fact_id, s.file, s.path, s.stmt, s.name, s.kind, s.value,
                   s.value_dc, s.owner, s.owner_dc,
                   row_number() OVER (PARTITION BY s.file ORDER BY s.owner, s.stmt) - 1 AS ordinal
            FROM scoped s JOIN collection_facts f ON f.path = s.path
            WHERE s.owner_type = 'function_definition'
        """,
        "elements": r"""
            WITH python AS (
              SELECT n.fact_id AS file, n.node_id AS id, n.parent_id AS pid,
                     n.sibling_index AS sib, n.type, n.name, n.start_line AS sl,
                     n.start_column AS sc, n.end_line AS el, n.end_column AS ec
              FROM ast_nodes n SEMI JOIN collection_literals l ON l.file = n.fact_id
              WHERE n.type NOT IN ('[', ']', '(', ')', ',', 'comment')
            ), members AS (
              SELECT l.fact_id, l.file, l.stmt, e.id, e.type, e.name, e.sl, e.sc, e.el, e.ec,
                     row_number() OVER (PARTITION BY l.stmt, l.file ORDER BY e.id) - 1 AS place,
                     CASE WHEN e.type IN ('string', 'concatenated_string')
                            AND NOT regexp_matches(e.name, '^[bBfFtT]|^[rRuU][bBfFtT]')
                         THEN 'string'
                          WHEN e.type IN ('integer', 'float') THEN 'number'
                          WHEN e.type IN ('true', 'false') THEN 'boolean'
                          WHEN e.type = 'none' THEN 'none'
                          WHEN e.type IN ('list', 'tuple', 'set') THEN 'sequence'
                          WHEN e.type = 'dictionary' THEN 'mapping' END AS literal_kind
              FROM collection_literals l JOIN python e ON e.file = l.file AND e.pid = l.value
            ), keyed AS (
              SELECT m.*, k.id AS key, k.type AS key_type, k.name AS key_name, k.sl AS ksl,
                     k.sc AS ksc, k.el AS kel, k.ec AS kec,
                     CASE WHEN k.type IN ('string', 'concatenated_string')
                            AND NOT regexp_matches(k.name, '^[bBfFtT]|^[rRuU][bBfFtT]')
                         THEN 'string'
                          WHEN k.type IN ('integer', 'float') THEN 'number'
                          WHEN k.type IN ('true', 'false') THEN 'boolean'
                          WHEN k.type = 'none' THEN 'none'
                          WHEN k.type IN ('list', 'tuple', 'set') THEN 'sequence'
                          WHEN k.type = 'dictionary' THEN 'mapping' END AS key_kind,
                     (SELECT count(*) FROM python c
                      WHERE c.file = m.file AND c.pid = m.id) AS width
              FROM members m
              LEFT JOIN python k ON k.file = m.file AND k.pid = m.id AND m.type = 'tuple'
                                AND k.id = (SELECT min(c.id) FROM python c
                                            WHERE c.file = m.file AND c.pid = m.id)
            )
            SELECT k.fact_id, k.file, k.stmt, k.place, k.literal_kind, k.width,
                   k.type = 'tuple' AND k.width = 2 AS is_pair, k.key_kind,
                   CASE WHEN k.type IN ('string', 'concatenated_string', 'integer', 'float',
                                        'true', 'false', 'none', 'identifier') THEN k.name
                        ELSE syntax_slice(s.content, k.sl, k.sc, k.el, k.ec) END AS text,
                   CASE WHEN k.key IS NULL THEN NULL
                        WHEN k.key_type IN ('string', 'concatenated_string', 'integer', 'float',
                                            'true', 'false', 'none', 'identifier') THEN k.key_name
                        ELSE syntax_slice(s.content, k.ksl, k.ksc, k.kel, k.kec) END AS key_text
            FROM keyed k JOIN collection_literals l ON l.file = k.file AND l.stmt = k.stmt
            JOIN source_files s ON s.path = l.path
        """,
        "reads": r"""
            WITH python AS (
              SELECT n.fact_id AS file, n.node_id AS id, n.parent_id AS pid,
                     n.sibling_index AS sib, n.type, n.name, n.children_count AS cc,
                     n.descendant_count AS dc
              FROM ast_nodes n SEMI JOIN collection_literals l ON l.file = n.fact_id
            ), nested AS (
              SELECT c.file, c.owner, s.id, s.id + s.dc AS last
              FROM (SELECT DISTINCT file, owner, owner_dc FROM collection_literals) c
              JOIN python s ON s.file = c.file AND s.id > c.owner AND s.id <= c.owner + c.owner_dc
                           AND s.type IN ('function_definition', 'class_definition')
            ), uses AS (
              SELECT c.file, c.stmt, x.id, p.id AS parent, p.type AS ptype,
                     syntax_is_load(p.type, x.sib, p.sib, g.type) AS is_load,
                     syntax_is_store(p.type, x.sib, p.sib, g.type) AS is_store,
                     (p.type IN ('for_statement', 'for_in_clause')
                      AND i.id IS NOT NULL) AS iterates,
                     (p.type = 'comparison_operator' AND p.cc = 3 AND x.sib = 2
                      AND o.type IN ('in', 'not in')) AS member
              FROM collection_literals c
              JOIN python x ON x.file = c.file AND x.type = 'identifier' AND x.name = c.name
                           AND x.id BETWEEN c.owner + 1 AND c.owner + c.owner_dc
              JOIN python p ON p.file = x.file AND p.id = x.pid
              LEFT JOIN python g ON g.file = p.file AND g.id = p.pid
              LEFT JOIN python i ON i.file = x.file AND i.pid = x.pid AND i.sib = x.sib - 1
                                AND i.type = 'in'
              LEFT JOIN python o ON o.file = x.file AND o.pid = x.pid AND o.sib = 1
              WHERE NOT EXISTS (SELECT 1 FROM nested n
                                WHERE n.file = c.file AND n.owner = c.owner
                                  AND x.id BETWEEN n.id AND n.last)
            ), alone AS (
              SELECT file, pid, count(*) AS n, any_value(id) AS id, any_value(type) AS type
              FROM python WHERE type <> 'comment' GROUP BY file, pid
            ), targets AS (
              SELECT file, pid, count(*) AS n, bool_and(type = 'identifier') AS names_only,
                     array_agg(name ORDER BY id) AS names
              FROM python WHERE type NOT IN (',', '(', ')') GROUP BY file, pid
            ), lookups AS (
              SELECT l.file, l.parent
              FROM (SELECT DISTINCT file, parent FROM uses
                    WHERE iterates AND ptype = 'for_statement') l
              JOIN python tg ON tg.file = l.file AND tg.pid = l.parent AND tg.sib = 1
                            AND tg.type IN ('pattern_list', 'tuple_pattern')
              JOIN targets t ON t.file = l.file AND t.pid = tg.id AND t.n = 2 AND t.names_only
              JOIN python b ON b.file = l.file AND b.pid = l.parent AND b.type = 'block'
              JOIN alone ib ON ib.file = l.file AND ib.pid = b.id AND ib.n = 1
                           AND ib.type = 'if_statement'
              JOIN python cmp ON cmp.file = l.file AND cmp.pid = ib.id AND cmp.sib = 1
                             AND cmp.type = 'comparison_operator' AND cmp.cc = 3
              JOIN python op ON op.file = l.file AND op.pid = cmp.id AND op.sib = 1
                            AND op.type = '=='
              JOIN python lhs ON lhs.file = l.file AND lhs.pid = cmp.id AND lhs.sib = 0
                             AND lhs.type = 'identifier' AND lhs.name = t.names[1]
              JOIN python cb ON cb.file = l.file AND cb.pid = ib.id AND cb.type = 'block'
              JOIN alone rs ON rs.file = l.file AND rs.pid = cb.id AND rs.n = 1
                           AND rs.type = 'return_statement'
              JOIN python rv ON rv.file = l.file AND rv.pid = rs.id AND rv.sib = 1
                            AND rv.type = 'identifier' AND rv.name = t.names[2]
              WHERE NOT EXISTS (SELECT 1 FROM python e
                                WHERE e.file = l.file AND e.pid IN (l.parent, ib.id)
                                  AND e.type IN ('else_clause', 'elif_clause'))
            )
            SELECT u.file, u.stmt,
                   count(*) FILTER (WHERE is_store) AS stores,
                   count(*) FILTER (WHERE is_load) AS loads,
                   count(*) FILTER (WHERE iterates) AS iteration,
                   count(*) FILTER (WHERE member) AS membership,
                   count(*) FILTER (WHERE iterates AND k.parent IS NOT NULL) AS lookup
            FROM uses u LEFT JOIN lookups k ON k.file = u.file AND k.parent = u.parent
            GROUP BY u.file, u.stmt
        """,
        "local_collections": r"""
            SELECT l.fact_order, l.fact_id, l.fact_id AS parent_id,
                   l.fact_id || '/local_collections:' || l.ordinal AS record_id, l.ordinal,
                   l.name, l.kind, coalesce(e.value_count, 0) AS value_count,
                   coalesce(e.homogeneous, false) AS has_homogeneous_literals,
                   r.loads > 0 AND r.loads = r.iteration AS all_reads_are_iteration,
                   r.loads > 0 AND r.loads = r.membership AS all_reads_are_membership,
                   coalesce(e.distinct_texts, true) AS values_are_unique
            FROM (SELECT x.*, row_number() OVER (PARTITION BY x.file ORDER BY x.owner, x.stmt) - 1
                               AS ordinal
                  FROM collection_literals x
                  JOIN collection_reads y ON y.file = x.file AND y.stmt = x.stmt
                                        AND y.stores = 1) l
            JOIN collection_reads r ON r.file = l.file AND r.stmt = l.stmt
            LEFT JOIN (SELECT file, stmt, count(*) AS value_count,
                              count(DISTINCT literal_kind) = 1
                                AND bool_and(literal_kind IS NOT NULL) AS homogeneous,
                              count(DISTINCT text) = count(*) AS distinct_texts
                       FROM collection_elements GROUP BY file, stmt) e
              ON e.file = l.file AND e.stmt = l.stmt
        """,
        "pair_sequences": r"""
            SELECT l.fact_order, l.fact_id, l.fact_id AS parent_id,
                   l.fact_id || '/pair_sequences:' || l.ordinal AS record_id, l.ordinal,
                   e.pairs AS pair_count,
                   e.key_kinds = 1 AND e.typed_keys = e.pairs AND e.distinct_keys = e.pairs
                     AS keys_are_unique_literals,
                   r.stores = 1 AS has_single_assignment,
                   r.loads > 0 AND r.loads = r.lookup AS all_reads_are_lookup_loops
            FROM (SELECT x.*, row_number() OVER (PARTITION BY x.file ORDER BY x.owner, x.stmt) - 1
                               AS ordinal
                  FROM collection_literals x
                  JOIN (SELECT file, stmt FROM collection_elements GROUP BY file, stmt
                        HAVING count(*) = count(*) FILTER (WHERE is_pair)) y
                    ON y.file = x.file AND y.stmt = x.stmt) l
            JOIN (SELECT file, stmt, count(*) AS pairs, count(DISTINCT key_kind) AS key_kinds,
                         count(key_kind) AS typed_keys, count(DISTINCT key_text) AS distinct_keys
                  FROM collection_elements GROUP BY file, stmt) e
              ON e.file = l.file AND e.stmt = l.stmt
            JOIN collection_reads r ON r.file = l.file AND r.stmt = l.stmt
        """,
    }
