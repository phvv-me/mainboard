"""mcmr's OverrideFact, derived from ClassFact's inheritance edges and sitting_duck's trees.

One fact per link between a class and a class above it, nearest first in the order Python looks a
name up (left to right, depth first, each class once, at the depth it is first met). A link states
what the subclass declares and what it inherits from that ancestor: the members no nearer ancestor
already supplied. A member is a method or property written in the class body (at any depth of
`if`/`try`/`for`/`match` blocks, not inside another function), or an attribute assigned there by
`name = ...` or by `self.name = ...` in any of its methods; a callable wins over an attribute of
its name. A name defined more than once stays as every definition Python may still bind: one a
later definition runs after, in the same block or in a block around it, is shadowed and left out,
while the arms of an `if`/`else` or `try`/`except` each stay. A `@name.setter`, `.getter` or
`.deleter` definition is no member of its own: the property it follows is. A stub
`x.pyi` beside `x.py` is no class of its own: its classes are the `.py` classes of the same name,
and its members count only where the `.py` class has none of that name.

Differences from the kernel, kept on purpose:

- a base written `Base[int]` or `Generic[T]` is read as `Base` or `Generic`; the kernel read it as
  an empty name, so such a subclass had no link to `Base` and counted `Generic[T]` as a nameless
  ancestor;
- `base_decorators` holds the written decorators only, without the kernel's `registered-component`
  and `external-binding` marks, and `derived_decorators`, `overridden_member_count` and the
  list lengths no rule reads are left out;
- a base written through a module (`base.Base`, `a.base.Base`, `b.Base` after `import a.base as b`)
  links to the class `from`-imports and `import`s reach, the way the kernel does, and to nothing
  when no repository module is reached: ClassFact's bare-name edge to a same-named class of the
  subclass's own module is dropped for it, though a dotted base whose head is no import
  (`Outer.Inner`) keeps that edge;
- a bare base name that some import in the file binds (the last one by line, in any scope, as
  mcmr's alias table keeps it, which leaves out `if TYPE_CHECKING:` imports) is the class that
  import reaches, or nothing when it reaches none:
  a same-named class of the module (`class Widget(Widget)`, an `except ImportError` fallback
  class, a later redefinition) never takes the link;
- a class defined more than once (`if`/`else` branches) links once per definition, and so does a
  class above it defined more than once (`class Mixin(Mixin)`): the first link of a pair of
  qualnames (the nearest ancestor definition of the first subclass definition) keeps the bare
  `fact_id`, the later ones add `@path:line` of the subclass definition and `>path:line` of the
  ancestor definition, so every key is unique and the same on each run; the kernel kept only the
  first and merged the members of all. A subclass of a class defined more than once directly in
  a module or class body inherits the latest definition above it in its own file, or the last one
  when it is in another file, as Python binds the name: the kernel inherited the merge of all, so
  a member only a shadowed definition holds is no longer overridden;
- a method defined more than once is one declared record, as the kernel has it: the line, source
  and decorators of its first definition Python may still bind, and the parameters of all of them
  stacked by position (`self, self, a, a`) the way the kernel states them. Only the arms Python
  may still run are stacked (see above): where the kernel also stacked an `@overload` series, the
  shadowed overloads, or a property's setter, those add nothing here;
- an `__init__` that calls its base in any of its definitions counts as calling it, and so does
  one that calls it inside a lambda, which is not a scope of its own for mcmr's calls;
- only Python classes are linked.
"""

from ...catalog import Family

BINDINGS = r"""
    WITH guarded AS (
      SELECT f.path, s.start_line, s.end_line
      FROM ast_nodes s
      JOIN ast_nodes b ON b.fact_id = s.fact_id AND b.node_id = s.parent_id AND b.type = 'block'
      JOIN ast_nodes i ON i.fact_id = b.fact_id AND i.node_id = b.parent_id
        AND i.type = 'if_statement'
      JOIN ast_nodes t ON t.fact_id = i.fact_id AND t.parent_id = i.node_id AND t.sibling_index = 1
        AND regexp_matches(t.name, '(^|\.)TYPE_CHECKING$')
      JOIN ast_files f ON f.fact_id = s.fact_id
      WHERE s.type IN ('import_statement', 'import_from_statement')
    )
    SELECT DISTINCT ON (path, name) path, name, start_line, imported_name AS original,
           starts_with(ltrim(declaration_text), 'from') AS is_from
    FROM import_binding_facts i
    WHERE language = 'python' AND NOT EXISTS (
      SELECT 1 FROM guarded g WHERE g.path = i.path AND i.start_line BETWEEN g.start_line
                                                                          AND g.end_line)
    ORDER BY path, name, start_line DESC
"""

EDGES = r"""
    WITH dotted AS (
      SELECT b.class_id, b.ordinal, b.path, w.text, split_part(w.text, '.', 1) AS head,
             substr(w.text, strpos(w.text, '.') + 1) AS rest
      FROM class_bases b
      CROSS JOIN LATERAL (SELECT trim(regexp_extract(b.text, '^[^\[\(]*')) AS text) w
      WHERE contains(w.text, '.')
    ), through_import AS (
      SELECT d.* FROM dotted d
      WHERE EXISTS (SELECT 1 FROM import_binding_facts i
                    WHERE i.language = 'python' AND i.path = d.path AND i.name = d.head)
    ), written AS (
      SELECT * FROM through_import
      UNION ALL
      SELECT d.* FROM dotted d ANTI JOIN through_import USING (class_id, ordinal)
      WHERE NOT EXISTS (SELECT 1 FROM class_edges e
                        WHERE e.child = d.class_id AND e.ordinal = d.ordinal)
    ), bindings AS (
      SELECT path, name, module, regexp_matches(declaration_text, '\sas\s') AS aliased
      FROM import_binding_facts
      WHERE language = 'python' AND NOT starts_with(ltrim(declaration_text), 'from')
    ), claimed AS (
      SELECT b.class_id, b.ordinal, l.path, l.name, l.is_from, l.start_line
      FROM class_bases b
      JOIN override_bindings l ON l.path = b.path
                              AND l.name = trim(regexp_extract(b.text, '^[^\[\(]*'))
      WHERE NOT contains(l.name, '.')
    ), spelled AS (
      SELECT c.class_id, c.ordinal, f.target || '.' || f.imported_name AS dotted
      FROM claimed c JOIN class_from_imports f ON f.path = c.path AND f.name = c.name
                                              AND f.start_line = c.start_line
      WHERE c.is_from
      UNION
      SELECT w.class_id, w.ordinal, f.target || '.' || f.imported_name || '.' || w.rest AS dotted
      FROM written w JOIN class_from_imports f ON f.path = w.path AND f.name = w.head
      UNION
      SELECT w.class_id, w.ordinal, CASE WHEN i.aliased THEN i.module || '.' || w.rest ELSE w.text
                                         END
      FROM written w JOIN bindings i ON i.path = w.path AND i.name = w.head
      UNION
      SELECT class_id, ordinal, text FROM written
    ), reached AS (
      SELECT s.class_id, s.ordinal, coalesce(d1.class_id, d2.class_id) AS parent
      FROM spelled s
      LEFT JOIN class_registry d1 ON d1.module = regexp_replace(s.dotted, '\.[^.]*$', '')
                                 AND d1.name = regexp_extract(s.dotted, '[^.]*$')
      LEFT JOIN class_reexports r ON r.module = regexp_replace(s.dotted, '\.[^.]*$', '')
                                 AND r.name = regexp_extract(s.dotted, '[^.]*$')
      LEFT JOIN class_registry d2 ON d2.module = r.origin_module AND d2.name = r.origin_name
    ), linked AS (
      SELECT child, parent, ordinal FROM class_edges
      ANTI JOIN through_import ON child = class_id AND class_edges.ordinal = through_import.ordinal
      ANTI JOIN claimed ON child = claimed.class_id AND class_edges.ordinal = claimed.ordinal
      UNION
      SELECT class_id, parent, ordinal FROM reached WHERE parent IS NOT NULL AND parent <> class_id
    )
    SELECT l.child, coalesce(t.twin, l.parent) AS parent, l.ordinal
    FROM linked l
    LEFT JOIN override_stubs t ON t.stub = l.parent
    ANTI JOIN override_stubs s ON s.stub = l.child
"""

STUBS = """
    SELECT s.class_id AS stub, arg_max(p.class_id, p.class_node) AS twin
    FROM class_keys s
    JOIN class_keys p ON p.module = s.module AND p.qualname = s.qualname
                     AND NOT ends_with(p.path, '.pyi')
    WHERE ends_with(s.path, '.pyi')
    GROUP BY s.class_id
"""

REBOUND = """
    WITH linked AS (SELECT * FROM override_unbound)
    SELECT l.child, l.ordinal, coalesce(arg_max(q.class_id, q.class_node) FILTER (
             WHERE p.path <> c.path OR q.class_node < c.class_node), l.parent) AS parent
    FROM linked l
    JOIN override_plain p ON p.class_id = l.parent
    JOIN override_plain q ON q.qualname = p.qualname AND q.path = p.path
    JOIN class_keys c ON c.class_id = l.child
    GROUP BY l.child, l.ordinal, l.parent
    UNION ALL
    SELECT child, ordinal, parent FROM linked ANTI JOIN override_plain ON class_id = parent
"""

PLAIN = """
    SELECT k.class_id, k.qualname, k.path, k.class_node
    FROM class_keys k
    JOIN class_nodes n ON n.fact_id = k.fact_id AND n.node_id = k.class_node
    LEFT JOIN class_nodes d ON d.fact_id = n.fact_id AND d.node_id = n.parent_id
                           AND d.type = 'decorated_definition'
    JOIN class_nodes o ON o.fact_id = n.fact_id AND o.node_id = coalesce(d.parent_id, n.parent_id)
    LEFT JOIN class_nodes g ON g.fact_id = o.fact_id AND g.node_id = o.parent_id
                           AND o.type = 'block'
    WHERE o.type = 'module' OR g.type = 'class_definition'
"""

LINKS = """
    WITH RECURSIVE walk AS (
      SELECT e.child AS class_id, e.parent AS ancestor, 1 AS depth, [e.ordinal] AS path,
             [e.child, e.parent] AS seen
      FROM override_edges e
      UNION ALL
      SELECT w.class_id, e.parent, w.depth + 1, list_append(w.path, e.ordinal),
             list_append(w.seen, e.parent)
      FROM walk w JOIN override_edges e ON e.child = w.ancestor
      WHERE NOT list_contains(w.seen, e.parent)
    ), ranked AS (
      SELECT class_id, ancestor, arg_min(depth, path) AS depth, min(path) AS path
      FROM walk GROUP BY class_id, ancestor
    )
    SELECT 'override:' || d.qualname || ':' || b.qualname
             || CASE WHEN row_number() OVER (PARTITION BY d.qualname, b.qualname
                                             ORDER BY d.path, c.start_line, r.depth, b.path,
                                                      a.start_line) > 1
                     THEN '@' || d.path || ':' || c.start_line || '>' || b.path || ':'
                          || a.start_line ELSE '' END AS fact_id, d.path,
           c.start_line, 0 AS start_column, c.start_line AS end_line, 0 AS end_column,
           'python' AS language, r.class_id, r.ancestor AS base_id, r.depth,
           row_number() OVER (PARTITION BY r.class_id ORDER BY r.path) AS seq,
           d.qualname AS derived, b.qualname AS base
    FROM ranked r
    JOIN class_keys d ON d.class_id = r.class_id
    JOIN class_keys b ON b.class_id = r.ancestor
    JOIN class_defs c ON c.class_id = r.class_id
    JOIN class_defs a ON a.class_id = r.ancestor
"""

ARMS = r"""
    WITH candidates AS (
      SELECT coalesce(s.twin, c.class_id) AS class_id, s.stub IS NOT NULL AS is_stub, f.fact_id,
             f.name, f.node_id AS node, coalesce(d.node_id, f.node_id) AS statement,
             coalesce(d.parent_id, f.parent_id) AS block, true AS is_callable,
             EXISTS (SELECT 1 FROM class_decorations x
                     WHERE x.fact_id = f.fact_id AND x.decorated_node = d.node_id
                       AND regexp_matches(x.dotted, '^' || f.name || '\.(setter|getter|deleter)$'))
               AS is_accessor,
             f.is_async AS asynchronous,
             coalesce(d.start_line, f.start_line) AS start_line,
             coalesce(d.start_column, f.start_column) AS start_column,
             coalesce(e.end_line, f.end_line) AS end_line,
             coalesce(e.end_column, f.end_column) AS end_column
      FROM class_nodes f
      JOIN ast_nodes z ON z.path = f.path AND z.node_id = f.node_id
      JOIN class_defs c ON c.fact_id = f.fact_id AND c.class_node = z.scope."class"
                       AND coalesce(z.scope."function", -1) < z.scope."class"
      LEFT JOIN override_stubs s ON s.stub = c.class_id
      LEFT JOIN class_nodes d ON d.fact_id = f.fact_id AND d.node_id = f.parent_id
                             AND d.type = 'decorated_definition'
      LEFT JOIN class_block_ends e ON e.fact_id = f.fact_id AND e.owner_node = f.node_id
      WHERE f.type = 'function_definition'
      UNION ALL
      SELECT coalesce(s.twin, c.class_id), s.stub IS NOT NULL, a.fact_id,
             CASE WHEN l.type = 'identifier' THEN l.name ELSE substr(l.name, strpos(l.name,
                   '.') + 1)
                  END, a.node_id, a.node_id, NULL, false, false, false, a.start_line,
             a.start_column, a.end_line, a.end_column
      FROM class_nodes a
      JOIN class_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
                        AND l.sibling_index = 0
      JOIN class_defs c ON c.fact_id = a.fact_id AND c.class_node = a.scope_class
      LEFT JOIN override_stubs s ON s.stub = c.class_id
      WHERE a.type = 'assignment'
        AND ((l.type = 'identifier' AND a.scope_node = a.scope_class)
             OR (l.type = 'attribute' AND regexp_matches(l.name, '^(self|cls)\.\w+$')))
    ), callables AS (
      SELECT * FROM candidates c
      WHERE c.is_callable AND NOT (c.is_accessor AND EXISTS (
        SELECT 1 FROM candidates e
        WHERE e.is_callable AND e.fact_id = c.fact_id AND e.class_id = c.class_id
          AND e.name = c.name AND e.node < c.node))
    ), live AS (
      SELECT * FROM callables a
      WHERE NOT EXISTS (
        SELECT 1 FROM callables b JOIN class_nodes x ON x.fact_id = b.fact_id
                                                    AND x.node_id = b.block
        WHERE b.fact_id = a.fact_id AND b.class_id = a.class_id AND b.name = a.name
          AND b.statement > a.statement
          AND a.statement BETWEEN x.node_id AND x.node_id + x.descendant_count)
      AND NOT (a.is_stub AND EXISTS (
        SELECT 1 FROM callables n
        WHERE NOT n.is_stub AND n.class_id = a.class_id AND n.name = a.name))
    ), data AS (
      SELECT DISTINCT ON (class_id, name) * FROM candidates c
      WHERE NOT c.is_callable AND NOT EXISTS (
        SELECT 1 FROM candidates k
        WHERE k.is_callable AND k.class_id = c.class_id AND k.name = c.name)
      ORDER BY class_id, name, is_stub, node DESC
    )
    SELECT class_id, fact_id, name, node, statement, is_callable, asynchronous, start_line,
           start_column, end_line, end_column, is_stub,
           row_number() OVER (PARTITION BY class_id, name ORDER BY is_stub, fact_id, node) AS rank
    FROM (SELECT * FROM live UNION ALL SELECT * FROM data)
"""

MEMBERS = """
    SELECT w.class_id, w.fact_id, w.name, w.node, w.statement, w.is_callable, w.asynchronous,
           w.start_line AS line,
           classes_text(f.lines, w.start_line, w.start_column, w.end_line, w.end_column) AS source,
           row_number() OVER (PARTITION BY w.class_id ORDER BY w.name) - 1 AS ordinal
    FROM override_arms w JOIN class_files f ON f.fact_id = w.fact_id
    WHERE w.rank = 1
"""

PARAMETERS = """
    WITH written AS (
      SELECT m.class_id, m.name AS member, m.fact_id AS file_id, m.node AS arm, m.rank,
             p.node_id, coalesce(s.type, p.type) AS type, p.sibling_index,
             ltrim(CASE WHEN s.node_id IS NULL THEN p.name ELSE s.name END, '*') AS name
      FROM override_arms m
      JOIN class_nodes l ON l.fact_id = m.fact_id AND l.parent_id = m.node AND
            l.type = 'parameters'
      JOIN class_nodes p ON p.fact_id = m.fact_id AND p.parent_id = l.node_id
                        AND p.type NOT IN ('(', ')', ',', 'comment')
      LEFT JOIN class_nodes s ON s.fact_id = p.fact_id AND s.parent_id = p.node_id
                             AND s.sibling_index = 0 AND p.type = 'typed_parameter'
                             AND s.type IN ('list_splat_pattern', 'dictionary_splat_pattern')
      WHERE m.is_callable
    ), kinds AS (
      SELECT *,
             bool_or(type = '/') OVER later AS before_slash,
             bool_or(type IN ('*', 'list_splat_pattern')) OVER earlier AS after_star
      FROM written
      WINDOW later AS (PARTITION BY file_id, arm ORDER BY sibling_index
                       ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING),
             earlier AS (PARTITION BY file_id, arm ORDER BY sibling_index
                         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ), numbered AS (
      SELECT *, row_number() OVER (PARTITION BY file_id, arm ORDER BY sibling_index) - 1 AS slot
      FROM kinds WHERE type NOT IN ('/', '*')
    )
    SELECT class_id, member, name,
           row_number() OVER (PARTITION BY class_id, member ORDER BY slot, rank) - 1 AS ordinal,
           CASE WHEN type = 'list_splat_pattern' THEN 'var_positional'
                WHEN type = 'dictionary_splat_pattern' THEN 'var_keyword'
                WHEN coalesce(after_star, false) THEN 'keyword_only'
                WHEN before_slash THEN 'positional_only'
                ELSE 'positional_or_keyword' END AS kind,
           type IN ('default_parameter', 'typed_default_parameter') AS has_default
    FROM numbered
"""

LABELS = """
    SELECT b.class_id, b.ordinal, coalesce(i.original, b.base_name) AS label
    FROM class_bases b
    LEFT JOIN override_bindings i ON i.path = b.path AND i.name = b.text AND i.is_from
"""

ANCESTOR_NAMES = """
    SELECT class_id, name, row_number() OVER (PARTITION BY class_id ORDER BY name) - 1 AS ordinal
    FROM (SELECT class_id, label AS name FROM override_labels
          UNION
          SELECT l.class_id, b.label FROM override_links l
          JOIN override_labels b ON b.class_id = l.base_id)
"""

INITIALIZER_CALLS = """
    WITH RECURSIVE lambdas AS (
      SELECT fact_id, node_id, scope_node AS enclosing FROM class_nodes WHERE type = 'lambda'
    ), hosted(fact_id, node_id, host) AS (
      SELECT fact_id, node_id, enclosing FROM lambdas
      UNION ALL
      SELECT h.fact_id, h.node_id, l.enclosing
      FROM hosted h JOIN lambdas l ON l.fact_id = h.fact_id AND l.node_id = h.host
    ), owners AS (
      SELECT c.fact_id, c.node_id, coalesce(h.host, c.scope_node) AS owner
      FROM class_nodes c
      LEFT JOIN hosted h ON h.fact_id = c.fact_id AND h.node_id = c.scope_node
        AND h.host NOT IN (SELECT node_id FROM lambdas l WHERE l.fact_id = h.fact_id)
      WHERE c.type = 'call'
    )
    SELECT class_id, receiver, row_number() OVER (PARTITION BY class_id ORDER BY first_node) - 1
             AS ordinal
    FROM (SELECT x.class_id, coalesce(i.original, x.tail) AS receiver, min(x.node_id) AS first_node
          FROM (SELECT k.class_id, c.path, c.node_id,
                       split_part(regexp_extract(left(a.name, -9), '[^.:]*$'), '(', 1) AS tail
                FROM class_defs k
                JOIN class_nodes m ON m.fact_id = k.fact_id AND m.type = 'function_definition'
                                  AND m.name = '__init__' AND m.scope_node = k.class_node
                JOIN owners o ON o.fact_id = k.fact_id AND o.owner = m.node_id
                JOIN class_nodes c ON c.fact_id = o.fact_id AND c.node_id = o.node_id
                JOIN class_nodes a ON a.fact_id = c.fact_id AND a.parent_id = c.node_id
                                  AND a.sibling_index = 0 AND a.type = 'attribute'
                                  AND ends_with(a.name, '.__init__')) x
          LEFT JOIN override_bindings i ON i.path = x.path AND i.name = x.tail AND i.is_from
          GROUP BY x.class_id, coalesce(i.original, x.tail))
"""

FACTS = """
    SELECT l.*, coalesce(n.length, 0) AS ancestor_names_length
    FROM override_links l
    LEFT JOIN (SELECT class_id, count(*) AS length FROM override_ancestor_names
               GROUP BY class_id) n USING (class_id)
"""

DECLARED = """
    SELECT l.fact_id, l.fact_id AS parent_id, l.fact_id || '/declared:' || m.ordinal AS record_id,
           m.ordinal, m.name, m.is_callable AS parameters_present, m.asynchronous, m.line,
           m.source, l.class_id, m.fact_id AS file_id, m.statement
    FROM override_links l JOIN override_members m ON m.class_id = l.class_id
"""

INHERITED = """
    SELECT fact_id, fact_id AS parent_id,
           fact_id || '/inherited:' || (row_number() OVER (PARTITION BY fact_id ORDER BY name) - 1)
             AS record_id,
           row_number() OVER (PARTITION BY fact_id ORDER BY name) - 1 AS ordinal,
           name, is_callable AS parameters_present, asynchronous, line, source, base_id,
           file_id, statement
    FROM (SELECT l.fact_id, m.name, m.is_callable, m.asynchronous, m.line, m.source, l.base_id,
                 m.fact_id AS file_id, m.statement
          FROM override_links l
          JOIN override_members m ON m.class_id = l.base_id
          WHERE NOT EXISTS (
            SELECT 1 FROM override_links nearer
            JOIN override_members x ON x.class_id = nearer.base_id
            WHERE nearer.class_id = l.class_id AND nearer.seq < l.seq AND x.name = m.name))
"""

DECLARED_PARAMETERS = """
    SELECT d.fact_id, d.record_id AS parent_id, p.ordinal, p.name, p.kind, p.has_default
    FROM override_declared d
    JOIN override_parameters p ON p.class_id = d.class_id AND p.member = d.name
"""

INHERITED_PARAMETERS = """
    SELECT d.fact_id, d.record_id AS parent_id, p.ordinal, p.name, p.kind, p.has_default
    FROM override_inherited d
    JOIN override_parameters p ON p.class_id = d.base_id AND p.member = d.name
"""

VALUES = """
    SELECT l.fact_id, 'ancestor_names' AS relation, l.fact_id AS parent_id, a.ordinal,
           a.name AS string_value
    FROM override_links l JOIN override_ancestor_names a ON a.class_id = l.class_id
    UNION ALL
    SELECT l.fact_id, 'base_names', l.fact_id, b.ordinal, b.label
    FROM override_links l JOIN override_labels b ON b.class_id = l.class_id
    UNION ALL
    SELECT l.fact_id, 'base_decorators', l.fact_id, d.ordinal, d.dotted
    FROM override_links l JOIN class_def_decorations d ON d.class_id = l.base_id
    UNION ALL
    SELECT l.fact_id, 'initializer_calls', l.fact_id, i.ordinal, i.receiver
    FROM override_links l JOIN override_initializer_calls i ON i.class_id = l.class_id
    UNION ALL
    SELECT m.fact_id, 'declared.decorators', m.record_id, d.ordinal, d.dotted
    FROM override_declared m
    JOIN class_decorations d ON d.fact_id = m.file_id AND d.decorated_node = m.statement
    WHERE m.parameters_present
    UNION ALL
    SELECT m.fact_id, 'inherited.decorators', m.record_id, d.ordinal, d.dotted
    FROM override_inherited m
    JOIN class_decorations d ON d.fact_id = m.file_id AND d.decorated_node = m.statement
    WHERE m.parameters_present
"""


class Overrides(Family):
    family = "OverrideFact"
    verified = True
    relations = {
        "bindings": BINDINGS,
        "stubs": STUBS,
        "unbound": EDGES,
        "plain": PLAIN,
        "edges": REBOUND,
        "links": LINKS,
        "arms": ARMS,
        "members": MEMBERS,
        "parameters": PARAMETERS,
        "labels": LABELS,
        "ancestor_names": ANCESTOR_NAMES,
        "initializer_calls": INITIALIZER_CALLS,
        "facts": FACTS,
        "declared": DECLARED,
        "inherited": INHERITED,
        "declared_parameters": DECLARED_PARAMETERS,
        "inherited_parameters": INHERITED_PARAMETERS,
        "values": VALUES,
    }
