"""mcmr's CallFact, derived from sitting_duck's syntax trees (Python and the C family's calls).

A call is a `call` node; its callee is the first child and its arguments the second (an
`argument_list`, or the generator expression that is the sole argument). mcmr resolves the
callee's dotted name through the file's imports, the project's own definitions, the builtins and
the declared type of its receiver (an annotated parameter or variable seen earlier in the file,
by name): `resolved` does the same in SQL, except that a call in a parameter default stays
unresolved, as in mcmr (`x: frozenset = frozenset()` is a deliberate immutable default, which a
resolved `builtins.frozenset` would make PY-COLL0004 flag). The expression trees `call_expressions`
holds (positional arguments, the receiver, and below a nested call its arguments or below a
dictionary its values) come from walking those children. tree-sitter reads `[*a.b()]` as an
attribute whose receiver is the `list_splat`: `call_splats` moves such spans past the star.
"""

from ...catalog import Family
from ...facts import Query
from .calls_sql import BUILTINS, STANDARD_LIBRARY, PythonBase, listed

_BASE = PythonBase("CallFact")


def _relations() -> dict[str, str]:
    """The family's relations in dependency order; each reads the earlier ones by their view
    names (`call_nodes` ...), the base tables `PythonBase` builds among them."""
    own = {
        "symbols": _SYMBOLS,
        "bindings": _BINDINGS,
        "types": _TYPES,
        "links": _LINKS,
        "reach": _REACH,
        "splats": _SPLATS,
        "defaults": _DEFAULTS,
        "sites": _SITES,
        "spine": _SPINE,
        "roots": _ROOTS,
        "edges": _EDGES,
        "tree": _TREE,
        "keys": _KEYS,
        "bools": _BOOLS,
        "texts": _BASE.texts(
            "SELECT file, start_line, start_column, end_line, end_column FROM call_sites "
            "UNION ALL SELECT file, callee_start_line, callee_start_column, callee_end_line, "
            "callee_end_column FROM call_sites "
            "UNION ALL SELECT file, start_line, start_column, end_line, end_column FROM call_tree "
            "UNION ALL SELECT file, key_start_line, key_start_column, key_end_line, "
            "key_end_column FROM call_keys"
        ),
        "resolved": _RESOLVED,
        "native_files": _NATIVE_FILES,
        "native_lines": _NATIVE_LINES,
        "native_sites": _NATIVE_SITES,
        "native_texts": _BASE.texts(
            "SELECT file, start_line, start_column, end_line, end_column FROM call_native_sites "
            "UNION ALL SELECT file, callee_start_line, callee_start_column, callee_end_line, "
            "callee_end_column FROM call_native_sites",
            "call_native_lines",
        ),
        "assigned": _ASSIGNED,
        "calls": _CALLS,
        "keywords": _KEYWORDS,
        "module_bindings": _MODULE_BINDINGS,
        "expressions": _EXPRESSIONS,
        "expression_ancestry": _ANCESTRY,
        "mapping_entries": _MAPPING_ENTRIES,
    }
    return {
        "facts": _FACTS,
        **_BASE.relations(),
        **{
            name: sql.replace("@stdlib@", listed(STANDARD_LIBRARY)).replace(
                "@builtins@", listed(BUILTINS)
            )
            for name, sql in own.items()
        },
    }


_FACTS = r"""
    SELECT (row_number() OVER (ORDER BY f.path) - 1)::UBIGINT AS fact_order,
           'calls:' || f.path AS fact_id, f.path, 1 AS start_line, 0 AS start_column,
           coalesce(m.end_line, 1) AS end_line, coalesce(m.end_column, 0) AS end_column,
           f.language,
           (regexp_extract(f.path, '[^/]*$') LIKE 'test\_%' ESCAPE '\'
            OR regexp_extract(f.path, '[^/]*$') IN ('test.py', 'tests.py', 'conftest.py')
            OR contains(regexp_extract(f.path, '[^/]*$'), '_test.')
            OR contains(regexp_extract(f.path, '[^/]*$'), '.test.')
            OR contains(regexp_extract(f.path, '[^/]*$'), '.spec.')
            OR list_has_any(list_slice(string_split(f.path, '/'), 1, -2),
                            ['test', 'tests', '__tests__'])) AS is_test
    FROM ast_files f
    LEFT JOIN (SELECT fact_id, end_line, end_column FROM ast_nodes WHERE parent_id IS NULL) m
      USING (fact_id)
    WHERE f.language IN ('python', 'c', 'cpp', 'cuda')
"""

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

_SPLATS = r"""
    SELECT l.file, l.start_line, l.start_column, i.start_line AS new_line,
           i.start_column AS new_column
    FROM call_nodes l
    JOIN call_nodes i ON i.file = l.file AND i.parent = l.node AND i.sibling_index = 1
    WHERE l.type = 'list_splat'
"""

_DEFAULTS = r"""
    SELECT d.file, v.node AS lo, v.node + v.descendant_count AS hi
    FROM call_nodes d
    JOIN call_nodes v ON v.file = d.file AND v.parent = d.node
      AND v.sibling_index = d.children_count - 1
    WHERE d.type IN ('default_parameter', 'typed_default_parameter')
"""

_SITES = r"""
    WITH c AS (SELECT * FROM call_nodes WHERE type = 'call'),
    callees AS (
      SELECT c.file, c.node AS call_node, coalesce(i2.node, i1.node, f.node) AS node,
             coalesce(i2.type, i1.type, f.type) AS type,
             coalesce(i2.name, i1.name, f.name) AS name,
             coalesce(i2.start_line, i1.start_line, f.start_line) AS start_line,
             coalesce(i2.start_column, i1.start_column, f.start_column) AS start_column,
             coalesce(i2.end_line, i1.end_line, f.end_line) AS end_line,
             coalesce(i2.end_column, i1.end_column, f.end_column) AS end_column
      FROM c
      JOIN call_nodes f ON f.file = c.file AND f.parent = c.node AND f.sibling_index = 0
      LEFT JOIN call_nodes i1 ON i1.file = f.file AND i1.parent = f.node
        AND f.type = 'parenthesized_expression' AND i1.sibling_index = 1
      LEFT JOIN call_nodes i2 ON i2.file = i1.file AND i2.parent = i1.node
        AND i1.type = 'parenthesized_expression' AND i2.sibling_index = 1
    )
    SELECT c.file, c.node AS call_node,
           row_number() OVER (PARTITION BY c.file ORDER BY c.node)::UBIGINT - 1 AS ordinal,
           coalesce(sc.new_line, c.start_line) AS start_line,
           coalesce(sc.new_column, c.start_column) AS start_column, c.end_line, c.end_column,
           c.scope, c.klass,
           f.node AS callee_node, f.type AS callee_type, f.name AS callee_name,
           coalesce(sf.new_line, f.start_line) AS callee_start_line,
           coalesce(sf.new_column, f.start_column) AS callee_start_column,
           f.end_line AS callee_end_line, f.end_column AS callee_end_column,
           a.node AS arglist_node, a.type AS arglist_type,
           a.start_line AS arglist_start_line, a.start_column AS arglist_start_column,
           a.end_line AS arglist_end_line, a.end_column AS arglist_end_column,
           CASE WHEN p1.type <> 'parenthesized_expression' THEN p1.node
                WHEN p2.type <> 'parenthesized_expression' THEN p2.node
                ELSE p3.node END AS top_node,
           CASE WHEN p1.type <> 'parenthesized_expression' THEN p1.type
                WHEN p2.type <> 'parenthesized_expression' THEN p2.type
                ELSE p3.type END AS top_type,
           CASE WHEN p1.type <> 'parenthesized_expression' THEN p1.children_count
                WHEN p2.type <> 'parenthesized_expression' THEN p2.children_count
                ELSE p3.children_count END AS top_children,
           CASE WHEN p1.type <> 'parenthesized_expression' THEN c.node
                WHEN p2.type <> 'parenthesized_expression' THEN p1.node
                ELSE p2.node END AS wrapper_node,
           EXISTS (SELECT 1 FROM call_defaults d
                   WHERE d.file = c.file AND c.node BETWEEN d.lo AND d.hi) AS in_default
    FROM c
    JOIN callees f ON f.file = c.file AND f.call_node = c.node
    LEFT JOIN call_splats sc ON sc.file = c.file AND sc.start_line = c.start_line
      AND sc.start_column = c.start_column
    LEFT JOIN call_splats sf ON sf.file = c.file AND sf.start_line = f.start_line
      AND sf.start_column = f.start_column
    LEFT JOIN call_nodes a ON a.file = c.file AND a.parent = c.node AND a.sibling_index = 1
    LEFT JOIN call_nodes p1 ON p1.file = c.file AND p1.node = c.parent
    LEFT JOIN call_nodes p2 ON p2.file = c.file AND p2.node = p1.parent
    LEFT JOIN call_nodes p3 ON p3.file = c.file AND p3.node = p2.parent
"""

_SPINE = r"""
    WITH RECURSIVE walk(file, call_node, cur, parts, plain) AS (
      SELECT file, call_node, callee_node, []::VARCHAR[], true FROM call_sites
      UNION ALL
      SELECT w.file, w.call_node, c.node,
             CASE WHEN n.type = 'attribute' THEN list_prepend(a.name, w.parts) ELSE w.parts END,
             w.plain AND n.type IN ('attribute', 'list_splat')
      FROM walk w
      JOIN call_nodes n ON n.file = w.file AND n.node = w.cur
        AND n.type IN ('attribute', 'call', 'list_splat')
      JOIN call_nodes c ON c.file = n.file AND c.parent = n.node
        AND c.sibling_index = CASE n.type WHEN 'list_splat' THEN 1 ELSE 0 END
      LEFT JOIN call_nodes a ON a.file = n.file AND a.parent = n.node
        AND a.sibling_index = n.children_count - 1 AND a.type = 'identifier'
    )
    SELECT w.file, w.call_node, CASE WHEN r.type = 'identifier' THEN r.name END AS root_name,
           w.parts, w.plain
    FROM walk w JOIN call_nodes r ON r.file = w.file AND r.node = w.cur
    WHERE r.type NOT IN ('attribute', 'call', 'list_splat')
"""

_ROOTS = r"""
    WITH args AS (
      SELECT s.file, s.call_node, a.node, a.type, a.sibling_index, a.start_line, a.start_column,
             a.end_line, a.end_column
      FROM call_sites s
      JOIN call_nodes a ON a.file = s.file AND a.parent = s.arglist_node
      WHERE s.arglist_type = 'argument_list'
        AND a.type NOT IN ('keyword_argument', 'dictionary_splat', 'comment', '(', ')', ',')
      UNION ALL
      SELECT file, call_node, arglist_node, 'generator_expression', 0, arglist_start_line,
             arglist_start_column + 1, arglist_end_line, arglist_end_column - 1
      FROM call_sites WHERE arglist_type = 'generator_expression'
      UNION ALL
      SELECT s.file, s.call_node, coalesce(q.node, o.node), coalesce(q.type, o.type), -1,
             coalesce(q.start_line, o.start_line), coalesce(q.start_column, o.start_column),
             o.end_line, o.end_column
      FROM call_sites s
      JOIN call_nodes o ON o.file = s.file AND o.parent = s.callee_node AND o.sibling_index = 0
      LEFT JOIN call_nodes q ON q.file = o.file AND q.parent = o.node AND o.type = 'list_splat'
        AND q.sibling_index = 1
      WHERE s.callee_type = 'attribute'
    ), unwrapped AS (
      SELECT p.file, p.call_node, p.sibling_index,
             coalesce(i2.node, i1.node, p.node) AS node,
             coalesce(i2.type, i1.type, p.type) AS type,
             coalesce(i2.start_line, i1.start_line, p.start_line) AS start_line,
             coalesce(i2.start_column, i1.start_column, p.start_column) AS start_column,
             coalesce(i2.end_line, i1.end_line, p.end_line) AS end_line,
             coalesce(i2.end_column, i1.end_column, p.end_column) AS end_column
      FROM args p
      LEFT JOIN call_nodes i1 ON i1.file = p.file AND i1.parent = p.node
        AND p.type = 'parenthesized_expression' AND i1.sibling_index = 1
      LEFT JOIN call_nodes i2 ON i2.file = i1.file AND i2.parent = i1.node
        AND i1.type = 'parenthesized_expression' AND i2.sibling_index = 1
    )
    SELECT file, call_node,
           CASE WHEN sibling_index = -1 THEN 'receiver' ELSE 'argument' END AS relation,
           CASE WHEN sibling_index = -1 THEN 0
                ELSE row_number() OVER (PARTITION BY file, call_node, sibling_index = -1
                                        ORDER BY sibling_index) - 1 END::UBIGINT AS ordinal,
           node AS node, type, start_line, start_column, end_line, end_column
    FROM unwrapped
"""

_EDGES = r"""
    SELECT file, call_node AS parent, node AS child, relation, ordinal, type, start_line,
           start_column, end_line, end_column, NULL::INT AS entry
    FROM call_roots WHERE relation = 'argument'
    UNION ALL
    SELECT w.file, w.parent, coalesce(i2.node, i1.node, w.node) AS child,
           'mapping_value',
           (row_number() OVER (PARTITION BY w.file, w.parent ORDER BY w.entry) - 1)::UBIGINT,
           coalesce(i2.type, i1.type, w.type),
           coalesce(i2.start_line, i1.start_line, w.start_line),
           coalesce(i2.start_column, i1.start_column, w.start_column),
           coalesce(i2.end_line, i1.end_line, w.end_line),
           coalesce(i2.end_column, i1.end_column, w.end_column), w.entry_node
    FROM (
      SELECT d.file, d.node AS parent, e.sibling_index AS entry, e.node AS entry_node,
             v.node, v.type, v.start_line, v.start_column, v.end_line, v.end_column
      FROM call_nodes d
      JOIN call_nodes e ON e.file = d.file AND e.parent = d.node
        AND e.type IN ('pair', 'dictionary_splat')
      JOIN call_nodes v ON v.file = e.file AND v.parent = e.node
        AND v.sibling_index = (e.children_count - 1)
      WHERE d.type = 'dictionary') w
    LEFT JOIN call_nodes i1 ON i1.file = w.file AND i1.parent = w.node
      AND w.type = 'parenthesized_expression' AND i1.sibling_index = 1
    LEFT JOIN call_nodes i2 ON i2.file = i1.file AND i2.parent = i1.node
      AND i1.type = 'parenthesized_expression' AND i2.sibling_index = 1
"""

_TREE = r"""
    WITH RECURSIVE tree(file, call_node, node, parent, relation, ordinal, root_relation,
                        root_ordinal, depth, key, start_line, start_column, end_line,
                        end_column, type, entry) AS (
      SELECT file, call_node, node, NULL::INT, relation, ordinal, relation, ordinal, 0,
             [CASE relation WHEN 'argument' THEN 0 ELSE 1 END, ordinal::INT], start_line,
             start_column, end_line, end_column, type, NULL::INT
      FROM call_roots
      UNION ALL
      SELECT t.file, t.call_node, e.child, t.node, e.relation, e.ordinal, t.root_relation,
             t.root_ordinal, t.depth + 1, list_append(t.key, e.ordinal::INT), e.start_line,
             e.start_column, e.end_line, e.end_column, e.type, e.entry
      FROM tree t JOIN call_edges e ON e.file = t.file AND e.parent = t.node
    )
    SELECT file, call_node, node, parent, relation, ordinal, root_relation, root_ordinal, depth,
           (row_number() OVER (PARTITION BY file, call_node ORDER BY key) - 1)::UBIGINT AS idx,
           key, start_line, start_column, end_line, end_column, type, entry
    FROM tree
"""

_BOOLS = r"""
    WITH RECURSIVE base AS (
      SELECT file, node FROM call_nodes
      WHERE type IN ('true', 'false', 'comparison_operator', 'not_operator')
    ), operands AS (
      SELECT n.file, n.node, c.node AS operand
      FROM call_nodes n JOIN call_nodes c ON c.file = n.file AND c.parent = n.node
      WHERE (n.type = 'boolean_operator' AND c.sibling_index IN (0, 2))
         OR (n.type = 'conditional_expression' AND c.sibling_index IN (0, 4))
         OR (n.type = 'parenthesized_expression' AND c.sibling_index = 1)
    ), leaves(file, node, leaf) AS (
      SELECT DISTINCT file, node, node FROM operands
      UNION ALL
      SELECT l.file, l.node, o.operand
      FROM leaves l JOIN operands o ON o.file = l.file AND o.node = l.leaf
    )
    SELECT file, node FROM base
    UNION
    SELECT l.file, l.node
    FROM leaves l
    ANTI JOIN operands o ON o.file = l.file AND o.node = l.leaf
    LEFT JOIN base b ON b.file = l.file AND b.node = l.leaf
    GROUP BY l.file, l.node HAVING count(b.node) = count(*)
"""

_KEYS = r"""
    SELECT t.file, t.call_node, t.idx, k.start_line AS key_start_line,
           k.start_column AS key_start_column, k.end_line AS key_end_line,
           k.end_column AS key_end_column, e.type = 'dictionary_splat' AS is_spread
    FROM call_tree t
    JOIN call_nodes e ON e.file = t.file AND e.node = t.entry
    LEFT JOIN call_nodes k ON k.file = e.file AND k.parent = e.node AND k.sibling_index = 0
      AND e.type = 'pair'
    WHERE t.relation = 'mapping_value'
"""

_NATIVE_FILES = r"""
    SELECT row_number() OVER (ORDER BY path)::INT AS file, fact_id AS ast_fact, path, language
    FROM ast_files WHERE language IN ('c', 'cpp', 'cuda')
"""

_NATIVE_LINES = r"""
    SELECT f.file, l.n::INT AS n, l.text, strlen(l.text) = length(l.text) AS plain
    FROM call_native_files f
    JOIN (SELECT path, generate_subscripts(string_split(content, chr(10)), 1) AS n,
                 unnest(string_split(content, chr(10))) AS text
          FROM source_files) l USING (path)
"""

_NATIVE_SITES = r"""
    SELECT f.file, n.node_id::INT AS call_node,
           (row_number() OVER (PARTITION BY f.file ORDER BY n.node_id) - 1)::UBIGINT AS ordinal,
           n.start_line::INT AS start_line, n.start_column::INT AS start_column,
           n.end_line::INT AS end_line, n.end_column::INT AS end_column,
           c.start_line::INT AS callee_start_line, c.start_column::INT AS callee_start_column,
           c.end_line::INT AS callee_end_line, c.end_column::INT AS callee_end_column,
           p.type = 'expression_statement' AS discarded
    FROM ast_nodes n
    JOIN call_native_files f ON f.ast_fact = n.fact_id
    JOIN ast_nodes c ON c.fact_id = n.fact_id AND c.parent_id = n.node_id AND c.sibling_index = 0
    LEFT JOIN ast_nodes p ON p.fact_id = n.fact_id AND p.node_id = n.parent_id
    WHERE n.type = 'call_expression'
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

_ASSIGNED = r"""
    WITH RECURSIVE climb(file, call_node, assignment) AS (
      SELECT s.file, s.call_node, t.node
      FROM call_sites s
      JOIN call_nodes t ON t.file = s.file AND t.node = s.top_node AND t.type = 'assignment'
      JOIN call_nodes w ON w.file = s.file AND w.node = s.wrapper_node AND w.parent = t.node
        AND w.sibling_index = t.children_count - 1
      UNION ALL
      SELECT c.file, c.call_node, o.node
      FROM climb c
      JOIN call_nodes t ON t.file = c.file AND t.node = c.assignment
      JOIN call_nodes o ON o.file = t.file AND o.node = t.parent AND o.type = 'assignment'
        AND t.sibling_index = o.children_count - 1
    )
    SELECT c.file, c.call_node,
           CASE l.type WHEN 'identifier' THEN l.name
                       WHEN 'attribute' THEN
                         CASE WHEN regexp_matches(l.name, '^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$')
                              THEN l.name END END AS target
    FROM (SELECT file, call_node, min(assignment) AS assignment FROM climb GROUP BY ALL) c
    JOIN call_nodes l ON l.file = c.file AND l.parent = c.assignment AND l.sibling_index = 0
"""

_CALLS = r"""
    WITH python_calls AS (
    SELECT 'calls:' || f.path || ':call:' || s.ordinal AS call_id, 'calls:' || f.path AS fact_id,
           s.ordinal, r.qualified_name, r.target_id, f.path,
           coalesce(a.target, '') AS assigned_target,
           s.top_type = 'expression_statement' AND s.top_children = 1 AS result_is_discarded,
           r.is_external, r.is_standard_library, r.is_first_party, r.is_constructor,
           r.is_shadowed, r.has_ambiguous_alias, s.top_type = 'decorator' AS is_decorator_factory,
           EXISTS (SELECT 1 FROM call_roots q WHERE q.file = s.file AND q.call_node = s.call_node
                   AND q.type = 'list_splat') AS has_starred_arguments,
           f.path AS node_path, s.start_line AS node_start_line,
           s.start_column AS node_start_column, s.end_line AS node_end_line,
           s.end_column AS node_end_column, nt.text AS node_text,
           f.path AS callee_path, s.callee_start_line, s.callee_start_column, s.callee_end_line,
           s.callee_end_column, ct.text AS callee_text
    FROM call_sites s
    JOIN call_files f ON f.file = s.file
    JOIN call_resolved r ON r.file = s.file AND r.call_node = s.call_node
    JOIN call_texts nt ON nt.file = s.file AND nt.start_line = s.start_line
      AND nt.start_column = s.start_column AND nt.end_line = s.end_line
      AND nt.end_column = s.end_column
    JOIN call_texts ct ON ct.file = s.file AND ct.start_line = s.callee_start_line
      AND ct.start_column = s.callee_start_column AND ct.end_line = s.callee_end_line
      AND ct.end_column = s.callee_end_column
    LEFT JOIN call_assigned a ON a.file = s.file AND a.call_node = s.call_node
    ), native_calls AS (
    SELECT 'calls:' || f.path || ':call:' || s.ordinal AS call_id, 'calls:' || f.path AS fact_id,
           s.ordinal, ct.text AS qualified_name,
           'cpp:unresolved-symbol:' || replace(regexp_replace(f.path, '\.[^./]+$', ''), '/', '::')
             || '::' || ct.text AS target_id,
           f.path, '' AS assigned_target, s.discarded AS result_is_discarded,
           false AS is_external, false AS is_standard_library, false AS is_first_party,
           false AS is_constructor, false AS is_shadowed, false AS has_ambiguous_alias,
           false AS is_decorator_factory, false AS has_starred_arguments,
           f.path AS node_path, s.start_line AS node_start_line,
           s.start_column AS node_start_column, s.end_line AS node_end_line,
           s.end_column AS node_end_column, nt.text AS node_text
    FROM call_native_sites s
    JOIN call_native_files f ON f.file = s.file
    JOIN call_native_texts nt ON nt.file = s.file AND nt.start_line = s.start_line
      AND nt.start_column = s.start_column AND nt.end_line = s.end_line
      AND nt.end_column = s.end_column
    JOIN call_native_texts ct ON ct.file = s.file AND ct.start_line = s.callee_start_line
      AND ct.start_column = s.callee_start_column AND ct.end_line = s.callee_end_line
      AND ct.end_column = s.callee_end_column
    )
    SELECT * FROM python_calls
    UNION ALL BY NAME
    SELECT * FROM native_calls
    ORDER BY path, ordinal
"""

_KEYWORDS = r"""
    SELECT 'calls:' || f.path || ':call:' || s.ordinal AS call_id,
           (row_number() OVER (PARTITION BY s.file, s.call_node ORDER BY k.sibling_index)
            - 1)::UBIGINT AS ordinal, k.name
    FROM call_sites s
    JOIN call_files f ON f.file = s.file
    JOIN call_nodes k ON k.file = s.file AND k.parent = s.arglist_node
      AND k.type = 'keyword_argument'
    ORDER BY f.path, s.ordinal, k.sibling_index
"""

_MODULE_BINDINGS = r"""
    SELECT 'calls:' || f.path AS fact_id,
           (row_number() OVER (PARTITION BY d.file ORDER BY d.node) - 1)::UBIGINT AS ordinal,
           d.name
    FROM call_defs d
    JOIN call_files f ON f.file = d.file
    LEFT JOIN call_nodes n ON n.file = d.file AND n.node = d.node
    LEFT JOIN call_nodes p ON p.file = d.file AND p.node = n.parent
    WHERE d.scope = 0 AND (n.parent = 0 OR (p.type = 'decorated_definition' AND p.parent = 0))
    ORDER BY f.path, d.node
"""

_EXPRESSIONS = r"""
    SELECT 'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || t.idx AS expression_id,
           'calls:' || f.path || ':call:' || c.ordinal AS call_id,
           (row_number() OVER (ORDER BY f.path, c.ordinal, t.idx) - 1)::UBIGINT AS preorder,
           CASE WHEN t.parent IS NOT NULL THEN
             'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || p.idx END
             AS parent_expression_id,
           t.relation, t.ordinal, t.root_relation, t.root_ordinal, t.depth::UBIGINT AS depth,
           x.text, coalesce(r.qualified_name, '') AS qualified_name,
           CASE WHEN t.type = 'string' THEN
                  CASE WHEN regexp_matches(x.text, '^[A-Za-z]*[fFbB][A-Za-z]*[''"]')
                       THEN 'none' ELSE 'string' END
                WHEN t.type = 'concatenated_string' THEN
                  CASE WHEN regexp_matches(x.text, '(^|[''"\s])[A-Za-z]*[fFbB][A-Za-z]*[''"]')
                       THEN 'none' ELSE 'string' END
                WHEN t.type IN ('integer', 'float') THEN 'number'
                WHEN t.type IN ('true', 'false') THEN 'boolean'
                WHEN t.type = 'dictionary' THEN 'mapping'
                WHEN t.type IN ('list', 'tuple', 'set') THEN 'sequence'
                ELSE 'none' END AS literal_kind,
           CASE WHEN bl.node IS NOT NULL THEN 'bool' ELSE '' END AS resolved_type,
           f.path AS node_path, t.start_line AS node_start_line,
           t.start_column AS node_start_column,
           t.end_line AS node_end_line, t.end_column AS node_end_column, 'expression' AS node_kind,
           x.text AS node_text
    FROM call_tree t
    JOIN call_sites c ON c.file = t.file AND c.call_node = t.call_node
    JOIN call_files f ON f.file = t.file
    LEFT JOIN call_tree p ON p.file = t.file AND p.call_node = t.call_node AND p.node = t.parent
    JOIN call_texts x ON x.file = t.file AND x.start_line = t.start_line
      AND x.start_column = t.start_column AND x.end_line = t.end_line
      AND x.end_column = t.end_column
    LEFT JOIN call_resolved r ON r.file = t.file AND r.call_node = t.node AND t.type = 'call'
    LEFT JOIN call_bools bl ON bl.file = t.file AND bl.node = t.node
    ORDER BY f.path, c.ordinal, t.idx
"""

_ANCESTRY = r"""
    SELECT 'calls:' || f.path || ':call:' || c.ordinal AS call_id,
           'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || e.idx
             AS descendant_expression_id,
           a.depth::UBIGINT AS step,
           CASE WHEN a.depth = 0 THEN 'calls:' || f.path || ':call:' || c.ordinal
                ELSE 'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || p.idx END
             AS parent_id,
           CASE WHEN a.depth = 0 THEN 'call' ELSE 'expression' END AS parent_kind,
           'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || a.idx
             AS child_expression_id,
           a.relation, a.ordinal
    FROM call_tree e
    JOIN call_tree a ON a.file = e.file AND a.call_node = e.call_node
      AND list_slice(e.key, 1, len(a.key)) = a.key
    LEFT JOIN call_tree p ON p.file = a.file AND p.call_node = a.call_node AND p.node = a.parent
    JOIN call_sites c ON c.file = e.file AND c.call_node = e.call_node
    JOIN call_files f ON f.file = e.file
"""

_MAPPING_ENTRIES = r"""
    SELECT 'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || d.idx AS expression_id,
           t.ordinal, CASE WHEN k.is_spread THEN '' ELSE trim(x.text, chr(34) || chr(39)) END
             AS key,
           k.is_spread,
           'calls:' || f.path || ':call:' || c.ordinal || ':expression:' || t.idx
             AS value_expression_id
    FROM call_tree t
    JOIN call_keys k ON k.file = t.file AND k.call_node = t.call_node AND k.idx = t.idx
    JOIN call_tree d ON d.file = t.file AND d.call_node = t.call_node AND d.node = t.parent
    JOIN call_sites c ON c.file = t.file AND c.call_node = t.call_node
    JOIN call_files f ON f.file = t.file
    LEFT JOIN call_texts x ON x.file = k.file AND x.start_line = k.key_start_line
      AND x.start_column = k.key_start_column AND x.end_line = k.key_end_line
      AND x.end_column = k.key_end_column
    WHERE t.relation = 'mapping_value'
    ORDER BY f.path, c.ordinal, d.idx, t.ordinal
"""


class CallCalls(Family):
    family = "CallFact"
    verified = True
    relations = _relations()

    @classmethod
    def queries(cls) -> list[Query]:
        """What the family reads from outside: the syntax trees and the sources (its own private
        tables are created as it is built)."""
        return [Query(sql="SELECT 1 FROM ast_files, ast_nodes, source_files")]
