# How a call's callee resolves: the file's imports, the project's definitions and the declared
# types of receivers, as the CallFact family's SQL relations.

_SYMBOLS = r"""
    SELECT qualified_name, arg_min(kind, priority) AS kind
    FROM (
      SELECT qualified_name, kind,
             CASE kind WHEN 'class' THEN 1 WHEN 'function' THEN 2 WHEN 'method' THEN 3
                       WHEN 'module' THEN 5 WHEN 'variable' THEN 6 ELSE 7 END AS priority
      FROM (
        SELECT qualified_name, kind FROM call_defs
        UNION ALL
        SELECT module, 'module' FROM call_files
        UNION ALL
        SELECT CASE WHEN a.scope = 0 THEN f.module ELSE c.qualified_name END || '.' || l.name,
               CASE WHEN a.scope = 0 THEN 'variable' ELSE 'attribute' END
        FROM call_nodes a
        JOIN call_nodes l ON l.file = a.file AND l.parent = a.node AND l.sibling_index = 0
          AND l.type = 'identifier'
        JOIN call_files f ON f.file = a.file
        LEFT JOIN call_defs c ON c.file = a.file AND c.node = a.scope AND c.kind = 'class'
        WHERE a.type = 'assignment' AND (a.scope = 0 OR c.node IS NOT NULL)
        UNION ALL
        SELECT c.qualified_name || '.' || regexp_extract(l.name, '^(self|cls)\.(\w+)$', 2),
               'attribute'
        FROM call_nodes a
        JOIN call_nodes l ON l.file = a.file AND l.parent = a.node AND l.sibling_index = 0
          AND l.type = 'attribute' AND regexp_matches(l.name, '^(self|cls)\.\w+$')
        JOIN call_defs c ON c.file = a.file AND c.node = a.klass
        WHERE a.type = 'assignment'))
    GROUP BY qualified_name
"""

_BINDINGS = r"""
    WITH RECURSIVE slots(file, node, type) AS (
      SELECT c.file, c.node, c.type
      FROM call_nodes p JOIN call_nodes c ON c.file = p.file AND c.parent = p.node
      WHERE (p.type IN ('assignment', 'augmented_assignment') AND c.sibling_index = 0)
         OR (p.type = 'for_statement' AND c.sibling_index = 1)
      UNION ALL
      SELECT c.file, c.node, c.type
      FROM call_nodes t JOIN call_nodes w ON w.file = t.file AND w.node = t.parent
        AND w.type = 'as_pattern'
      JOIN call_nodes i ON i.file = w.file AND i.node = w.parent AND i.type = 'with_item'
      JOIN call_nodes c ON c.file = t.file AND c.parent = t.node
      WHERE t.type = 'as_pattern_target'
      UNION ALL
      SELECT c.file, c.node, c.type
      FROM slots s JOIN call_nodes c ON c.file = s.file AND c.parent = s.node
      WHERE s.type IN ('pattern_list', 'tuple_pattern', 'list_pattern', 'list_splat_pattern',
                       'tuple', 'list', 'parenthesized_expression')
    )
    SELECT n.file, n.scope, n.name FROM slots s JOIN call_nodes n USING (file, node)
    WHERE n.type = 'identifier'
    UNION
    SELECT file, scope, name FROM call_defs
    UNION
    SELECT n.file, n.scope, n.name
    FROM call_nodes n JOIN call_nodes p ON p.file = n.file AND p.node = n.parent
    LEFT JOIN call_nodes g ON g.file = p.file AND g.node = p.parent
    WHERE n.type = 'identifier'
      AND ((p.type = 'parameters' AND p.parent IS NOT NULL)
           OR (p.type IN ('typed_parameter', 'default_parameter', 'typed_default_parameter')
               AND g.type = 'parameters' AND n.sibling_index = 0)
           OR (p.type IN ('list_splat_pattern', 'dictionary_splat_pattern')
               AND g.type IN ('parameters', 'typed_parameter')))
"""

_TYPES = r"""
    SELECT p.file, coalesce(d.func, 0) AS scope, c.name, calls_annotation_name(t.name) AS kind,
           c.node AS at
    FROM call_nodes p
    JOIN call_nodes d ON d.file = p.file AND d.node = p.parent
    JOIN call_nodes c ON c.file = p.file AND c.parent = p.node
      AND c.type IN ('typed_parameter', 'typed_default_parameter')
    JOIN call_nodes t ON t.file = c.file AND t.parent = c.node AND t.type = 'type'
    WHERE p.type = 'parameters'
    UNION ALL
    SELECT a.file, coalesce(a.func, 0), l.name, calls_annotation_name(t.name), a.node
    FROM call_nodes a
    JOIN call_nodes l ON l.file = a.file AND l.parent = a.node AND l.sibling_index = 0
      AND l.type = 'identifier'
    JOIN call_nodes t ON t.file = a.file AND t.parent = a.node AND t.type = 'type'
    WHERE a.type = 'assignment'
"""

_LINKS = r"""
    SELECT d.file, d.node AS scope,
           CASE WHEN d.type = 'function_definition' AND p.type = 'class_definition'
                THEN p.scope ELSE d.scope END AS next_scope
    FROM call_defs d LEFT JOIN call_defs p ON p.file = d.file AND p.node = d.scope
    UNION ALL
    SELECT file, node, scope FROM call_nodes WHERE type = 'lambda' AND node <> 0
      AND sibling_index >= 0 AND scope <> node
"""

_REACH = r"""
    WITH RECURSIVE reach(file, scope, ancestor) AS (
      SELECT file, scope, scope FROM (SELECT DISTINCT file, scope FROM call_nodes)
      UNION ALL
      SELECT r.file, r.scope, l.next_scope
      FROM reach r JOIN call_links l ON l.file = r.file AND l.scope = r.ancestor
    )
    SELECT * FROM reach
"""

_RESOLVED = r"""
    WITH named AS (
      SELECT s.file, s.call_node, s.scope, s.klass, f.module, t.text AS callee_text,
             p.root_name, p.parts, s.in_default,
             CASE WHEN p.plain AND p.root_name IS NOT NULL AND NOT s.in_default
                  THEN array_to_string(list_prepend(p.root_name, p.parts), '.') END AS dotted,
             coalesce(p.root_name, '') AS collapsed_head,
             coalesce(CASE WHEN len(p.parts) > 0 THEN p.parts[-1] ELSE p.root_name END, '')
               AS last_name,
             trim(regexp_replace(t.text, '\s+', ' ', 'g')) AS flat
      FROM call_sites s
      JOIN call_files f ON f.file = s.file
      JOIN call_spine p ON p.file = s.file AND p.call_node = s.call_node
      JOIN call_texts t ON t.file = s.file AND t.start_line = s.callee_start_line
        AND t.start_column = s.callee_start_column AND t.end_line = s.callee_end_line
        AND t.end_column = s.callee_end_column
    ), typed AS (
      SELECT p.file, p.call_node, arg_max(t.kind, [r.ancestor, t.at]) AS kind
      FROM named p
      JOIN call_reach r ON r.file = p.file AND r.scope = p.scope
      JOIN call_types t ON t.file = r.file AND t.scope = r.ancestor
        AND t.name = split_part(p.dotted, '.', 1) AND t.at < p.call_node
      WHERE contains(p.dotted, '.') AND t.kind <> ''
      GROUP BY p.file, p.call_node
    ), expressions AS (
      SELECT p.*, coalesce(p.dotted,
                           CASE WHEN contains(p.flat, chr(39)) THEN p.flat
                                ELSE replace(p.flat, chr(34), chr(39)) END) AS expr,
             split_part(p.dotted, '.', 1) AS head,
             CASE WHEN contains(p.dotted, '.') THEN substr(p.dotted, strpos(p.dotted, '.') + 1)
                  ELSE '' END AS rest,
             a.target AS alias_target, c.qualified_name AS owner,
             coalesce(k.target || substr(y.kind, length(split_part(y.kind, '.', 1)) + 1), y.kind)
               AS declared
      FROM named p
      LEFT JOIN (SELECT file, binding, arg_max(target, item) AS target
                 FROM call_imports GROUP BY file, binding) a
        ON a.file = p.file AND a.binding = split_part(p.dotted, '.', 1)
      LEFT JOIN typed y ON y.file = p.file AND y.call_node = p.call_node
      LEFT JOIN (SELECT file, binding, arg_max(target, item) AS target
                 FROM call_imports GROUP BY file, binding) k
        ON k.file = p.file AND k.binding = split_part(y.kind, '.', 1)
      LEFT JOIN call_defs c ON c.file = p.file AND c.node = p.klass
    ), expanded AS (
      SELECT *, CASE WHEN dotted IS NULL THEN NULL WHEN alias_target IS NULL THEN dotted
                     WHEN rest = '' THEN alias_target ELSE alias_target || '.' || rest END
                  AS expanded
      FROM expressions
    ), candidates AS (
      SELECT file, call_node, 1 AS rank, declared || '.' || rest AS cand FROM expanded
      WHERE declared IS NOT NULL
      UNION ALL SELECT file, call_node, 2, module || '.' || declared || '.' || rest FROM expanded
      WHERE declared IS NOT NULL
      UNION ALL SELECT file, call_node, 3, owner || '.' || rest FROM expanded
      WHERE head IN ('self', 'cls') AND owner IS NOT NULL AND rest <> ''
      UNION ALL SELECT file, call_node, 4, module || '.' || expanded FROM expanded
      WHERE expanded IS NOT NULL
      UNION ALL SELECT file, call_node, 5, expanded FROM expanded WHERE expanded IS NOT NULL
      UNION ALL SELECT file, call_node, 6, dotted FROM expanded WHERE dotted IS NOT NULL
    ), hits AS (
      SELECT c.file, c.call_node, arg_min(c.cand, c.rank) AS qualified_name,
             arg_min(s.kind, c.rank) AS kind
      FROM candidates c JOIN call_symbols s ON s.qualified_name = c.cand
      GROUP BY c.file, c.call_node
    ), roots AS (
      SELECT DISTINCT file, split_part(target, '.', 1) AS root FROM call_imports
    ), placed AS (
      SELECT e.file, e.call_node, e.scope, e.expr, e.head, e.last_name, e.collapsed_head,
             e.dotted, e.parts, e.in_default,
             coalesce(h.qualified_name,
                      CASE WHEN NOT e.in_default AND NOT contains(e.expr, '.')
                                AND list_contains(@builtins@, e.expr)
                           THEN 'builtins.' || e.expr
                           WHEN e.expanded IS NOT NULL AND r.root IS NOT NULL
                                AND regexp_matches(e.expanded, '^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$')
                           THEN e.expanded
                           ELSE e.module || '::' || e.expr END) AS qualified_name,
             CASE WHEN h.kind IS NOT NULL THEN 'python:' || h.kind
                  WHEN NOT e.in_default AND NOT contains(e.expr, '.')
                       AND list_contains(@builtins@, e.expr)
                  THEN 'python:external-symbol'
                  WHEN e.expanded IS NOT NULL AND r.root IS NOT NULL
                       AND regexp_matches(e.expanded, '^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$')
                  THEN 'python:external-symbol'
                  ELSE 'python:unresolved-symbol' END AS target_kind,
             h.kind IS NOT NULL AS exact, h.kind AS hit_kind
      FROM expanded e
      LEFT JOIN hits h ON h.file = e.file AND h.call_node = e.call_node
      LEFT JOIN roots r ON r.file = e.file AND r.root = split_part(e.expanded, '.', 1)
    ), prefixed AS (
      SELECT DISTINCT p.file, p.call_node
      FROM (SELECT file, call_node,
                   unnest(list_transform(range(1, len(string_split(qualified_name, '.')) + 1),
                     lambda i: array_to_string(
                       list_slice(string_split(qualified_name, '.'), 1, i), '.'))) AS prefix
            FROM placed WHERE NOT in_default) p
      SEMI JOIN call_files m ON m.module = p.prefix
    ), shadowed AS (
      SELECT DISTINCT p.file, p.call_node
      FROM placed p
      JOIN call_reach r ON r.file = p.file AND r.scope = p.scope
      JOIN call_bindings b ON b.file = r.file AND b.scope = r.ancestor
        AND b.name = p.collapsed_head
      WHERE p.collapsed_head <> ''
    ), ambiguous AS (
      SELECT file, split_part(binding, '.', 1) AS name
      FROM call_imports GROUP BY ALL HAVING count(*) > 1
    )
    SELECT p.file, p.call_node, p.qualified_name,
           p.target_kind || ':' || p.qualified_name AS target_id,
           p.expr, p.dotted, p.exact, p.hit_kind,
           p.exact OR f.call_node IS NOT NULL AS is_first_party,
           (p.target_kind = 'python:external-symbol') AND NOT (p.exact OR f.call_node IS NOT NULL)
             AS is_external,
           (p.target_kind = 'python:external-symbol') AND NOT (p.exact OR f.call_node IS NOT NULL)
             AND (split_part(p.qualified_name, '.', 1) = 'builtins'
                  OR list_contains(@stdlib@, split_part(p.qualified_name, '.', 1)))
             AS is_standard_library,
           coalesce(p.hit_kind = 'class', false)
             OR left(p.last_name, 1) <> lower(left(p.last_name, 1)) AS is_constructor,
           s.call_node IS NOT NULL
             OR (len(p.parts) = 0 AND list_contains(@builtins@, p.collapsed_head)
                 AND (p.exact OR f.call_node IS NOT NULL))
             AS is_shadowed,
           a.name IS NOT NULL AS has_ambiguous_alias
    FROM placed p
    LEFT JOIN prefixed f ON f.file = p.file AND f.call_node = p.call_node
    LEFT JOIN shadowed s ON s.file = p.file AND s.call_node = p.call_node
    LEFT JOIN ambiguous a ON a.file = p.file AND a.name = p.collapsed_head
"""


def resolution() -> dict[str, str]:
    """The relations resolving callees, by relation name."""
    return {
        "symbols": _SYMBOLS,
        "bindings": _BINDINGS,
        "types": _TYPES,
        "links": _LINKS,
        "reach": _REACH,
        "resolved": _RESOLVED,
    }
