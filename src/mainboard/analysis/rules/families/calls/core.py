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

from ....catalog import Family
from ....engine.facts import Query
from ..constants.calls_sql import BUILTINS, STANDARD_LIBRARY
from .expressions import expressions
from .resolution import resolution
from .sql import PythonBase, listed

_BASE = PythonBase("CallFact")


def _relations() -> dict[str, str]:
    """The family's relations in dependency order, `PythonBase`'s base tables among them.

    Each reads the earlier ones by their view names (`call_nodes` ...).
    """
    resolving, walking = resolution(), expressions()
    own = {
        "symbols": resolving["symbols"],
        "bindings": resolving["bindings"],
        "types": resolving["types"],
        "links": resolving["links"],
        "reach": resolving["reach"],
        "splats": _SPLATS,
        "defaults": _DEFAULTS,
        "sites": _SITES,
        "spine": walking["spine"],
        "roots": walking["roots"],
        "edges": walking["edges"],
        "tree": walking["tree"],
        "keys": walking["keys"],
        "bools": walking["bools"],
        "texts": _BASE.texts(
            "SELECT file, start_line, start_column, end_line, end_column FROM call_sites "
            "UNION ALL SELECT file, callee_start_line, callee_start_column, callee_end_line, "
            "callee_end_column FROM call_sites "
            "UNION ALL SELECT file, start_line, start_column, end_line, end_column FROM call_tree "
            "UNION ALL SELECT file, key_start_line, key_start_column, key_end_line, "
            "key_end_column FROM call_keys "
            "UNION ALL SELECT v.file, v.start_line, v.start_column, v.end_line, v.end_column "
            "FROM call_nodes k JOIN call_nodes v ON v.file = k.file AND v.parent = k.node "
            "AND v.sibling_index = 2 WHERE k.type = 'keyword_argument'"
        ),
        "resolved": resolving["resolved"],
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
        "expressions": walking["expressions"],
        "expression_ancestry": walking["expression_ancestry"],
        "mapping_entries": walking["mapping_entries"],
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
            - 1)::UBIGINT AS ordinal, k.name, coalesce(x.text, '') AS value_text
    FROM call_sites s
    JOIN call_files f ON f.file = s.file
    JOIN call_nodes k ON k.file = s.file AND k.parent = s.arglist_node
      AND k.type = 'keyword_argument'
    LEFT JOIN call_nodes v ON v.file = k.file AND v.parent = k.node AND v.sibling_index = 2
    LEFT JOIN call_texts x ON x.file = v.file AND x.start_line = v.start_line
      AND x.start_column = v.start_column AND x.end_line = v.end_line
      AND x.end_column = v.end_column
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


class CallCalls(Family):
    family = "CallFact"
    relations = _relations()

    @classmethod
    def queries(cls) -> list[Query]:
        """What the family reads from outside: the syntax trees and the sources.

        Its own private tables are created as it is built.
        """
        return [Query(sql="SELECT 1 FROM ast_files, ast_nodes, source_files")]
