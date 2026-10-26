"""mcmr's SymbolReachFact for Python, derived from sitting_duck's syntax trees: every class,
function, method, property, module variable and class attribute a file declares, with how far the
references to it spread.

mcmr resolves every name a file reads against the declarations of the repository by plain text:
the first of a few candidate qualified names (the owner's member for `self.m`, the name inside the
module, the name through the file's import aliases, the text itself) that some declaration
carries, the receiver's annotated type first for `x.member`. The relations here do the same, one
step each: `nodes` (declarations), `aliases` (what each import binds), `types` (the annotated
names), `references` (the names a file reads), `landed` (each reference on the declaration its
candidates name, or on nothing), then `declarations` counts what reaches each declaration.
Differences from mcmr: decorators, default values, subscript indexes, `elif` tests and `except`
types count as reads, and declarations under `if TYPE_CHECKING:` exist. The target of `del name`
is a read, as it is for mcmr: it names what it removes.

Like mcmr, every positional argument of a class is an inheritance edge whatever its shape: a
class written `Generic[T]`, `list[int]`, `*bases` or `make_base()` has inheritance even when the
edge names nothing; a base written as a call (`namedtuple("P", "a")`) is read as its callee;
keyword arguments (`metaclass=M`) are no inheritance, though they still count as reads here
where mcmr never looks at them.
"""

from ....catalog import Family, Function
from ..constants.imports_common import MODULE_NAMES
from ..constants.symbols_common import BUILTIN_LIST, STORES
from ..imports.common import is_test_path, python_files
from . import common as symbols_common
from .common import import_relations

_PARSED = python_files("parsed")

_FACTS = f"""
    WITH {_PARSED}
    SELECT (row_number() OVER (ORDER BY path) - 1)::UBIGINT AS fact_order,
           'reach:' || path AS fact_id, path, 1::UBIGINT AS start_line, 0::UBIGINT AS start_column,
           1::UBIGINT AS end_line, 0::UBIGINT AS end_column, 'python' AS language,
           {is_test_path("path")} AS is_test_module
    FROM parsed
"""

_MODULES = f"""
    WITH {_PARSED}, {MODULE_NAMES}
    SELECT p.fact_id, p.path, m.module,
           regexp_matches(p.path, '(^|/)__init__\\.pyi?$') AS is_package
    FROM parsed p JOIN module_names m USING (fact_id)
"""

_SCOPES = """
    WITH defined AS (
      SELECT n.fact_id, n.node_id, n.parent_id, m.path, m.module, n.type, n.name, n.start_line,
             n.scope, m.module || coalesce('.' || array_to_string(
               list_transform(n.qualified_name, lambda q: q.name), '.'), '') AS qualname
      FROM ast_nodes n JOIN symbol_reach_modules m USING (fact_id)
      WHERE n.type IN ('class_definition', 'function_definition', 'module')
    )
    SELECT * FROM defined
    UNION ALL
    SELECT l.fact_id, l.node_id, l.parent_id, h.path, h.module, l.type, l.name, l.start_line,
           l.scope, h.qualname
    FROM ast_nodes l
    JOIN defined h ON h.fact_id = l.fact_id AND h.node_id = list_filter(
      l.scope.stack, lambda x: x.kind IN (240, 248, 252))[-1].id
    WHERE l.type = 'lambda'
"""

_NODES = """
    WITH RECURSIVE decorators AS (
      SELECT d.fact_id, d.parent_id AS wrapper,
             bool_or(regexp_extract(c.name, '[^.]*$') IN ('property', 'cached_property')
                     AND c.type IN ('identifier', 'attribute')) AS is_property
      FROM ast_nodes d JOIN ast_nodes c ON c.fact_id = d.fact_id AND c.parent_id = d.node_id
        AND c.sibling_index = 1
      WHERE d.type = 'decorator' GROUP BY ALL
    ), definitions AS (
      SELECT d.fact_id, d.path, d.module, d.node_id, d.name, coalesce(w.start_line, d.start_line)
               AS line, d.qualname, o.type AS owner_type,
             CASE WHEN d.type = 'class_definition' THEN 'class'
                  WHEN o.type <> 'class_definition' THEN 'function'
                  WHEN coalesce(x.is_property, false) THEN 'property' ELSE 'method' END AS kind,
             w.node_id IS NOT NULL
               OR (o.type = 'module' AND ends_with(d.path, '.pyi')) AS decorated
      FROM symbol_reach_scopes d
      JOIN symbol_reach_scopes o ON o.fact_id = d.fact_id
        AND o.node_id = greatest(tests_home(d.scope), 0)
      LEFT JOIN ast_nodes w ON w.fact_id = d.fact_id AND w.node_id = d.parent_id
        AND w.type = 'decorated_definition'
      LEFT JOIN decorators x ON x.fact_id = d.fact_id AND x.wrapper = w.node_id
      WHERE d.type IN ('class_definition', 'function_definition')
    ), targets(fact_id, assignment, node_id, type, name, children_count, parenthesized) AS (
      SELECT a.fact_id, a.node_id, l.node_id, l.type, l.name, l.children_count, false
      FROM ast_nodes a
      JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
        AND l.sibling_index = 0 AND l.type IN ('identifier', 'tuple_pattern')
      WHERE a.type = 'assignment'
      UNION ALL
      SELECT t.fact_id, t.assignment, i.node_id, i.type, i.name, i.children_count, true
      FROM targets t
      JOIN ast_nodes i ON i.fact_id = t.fact_id AND i.parent_id = t.node_id
        AND i.sibling_index = 1 AND i.type IN ('identifier', 'tuple_pattern')
      WHERE t.type = 'tuple_pattern' AND t.children_count = 3
    ), assigned AS (
      SELECT a.fact_id, o.path, o.module, a.node_id, l.name, a.start_line AS line,
             o.qualname || '.' || l.name AS qualname, o.type AS owner_type,
             CASE o.type WHEN 'class_definition' THEN 'attribute' ELSE 'variable' END AS kind,
             false AS decorated
      FROM ast_nodes a
      JOIN targets l ON l.fact_id = a.fact_id AND l.assignment = a.node_id
        AND l.type = 'identifier'
        AND (NOT l.parenthesized OR EXISTS (SELECT 1 FROM ast_nodes v WHERE v.fact_id = a.fact_id
                                            AND v.parent_id = a.node_id AND v.type = '='))
      JOIN symbol_reach_scopes o ON o.fact_id = a.fact_id
        AND o.node_id = greatest(tests_home(a.scope), 0)
        AND o.type IN ('class_definition', 'module')
      WHERE a.type = 'assignment'
    ), received AS (
      SELECT a.fact_id, c.path, c.module, a.node_id, split_part(l.name, '.', 2) AS name,
             a.start_line AS line, c.qualname || '.' || split_part(l.name, '.', 2) AS qualname,
             c.type AS owner_type, 'attribute' AS kind, false AS decorated
      FROM ast_nodes a
      JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
        AND l.sibling_index = 0 AND l.type = 'attribute'
        AND regexp_matches(l.name, '^(self|cls)\\.\\w+$')
      JOIN symbol_reach_scopes c ON c.fact_id = a.fact_id AND c.node_id = a.scope.class
      WHERE a.type = 'assignment'
    )
    SELECT * FROM (SELECT * FROM definitions UNION ALL SELECT * FROM assigned
                   UNION ALL SELECT * FROM received)
    QUALIFY row_number() OVER (PARTITION BY kind, qualname ORDER BY path, node_id) = 1
"""

_UNITS = f"""
    SELECT i.fact_id, i.node_id, i.start_line AS line, i.type,
           regexp_replace(regexp_replace(i.name, '^\\*+', ''), '\\s+', ' ', 'g') AS expression,
           CASE WHEN p.type = 'argument_list' AND g.type = 'class_definition' THEN 'inherit'
                WHEN p.type = 'call' AND i.sibling_index = 0
                THEN CASE WHEN g.type = 'argument_list' AND h.type = 'class_definition'
                          THEN 'inherit' ELSE 'call' END
                ELSE 'access' END AS kind,
           greatest(coalesce(i.scope.function, -1), coalesce(i.scope.class, -1), 0) AS scope_node,
           i.scope.class AS class_node
    FROM ast_nodes i
    {symbols_common.LOAD_JOINS}
    LEFT JOIN ast_nodes g ON g.fact_id = p.fact_id AND g.node_id = p.parent_id
      AND p.type IN ('argument_list', 'call')
    LEFT JOIN ast_nodes h ON h.fact_id = g.fact_id AND h.node_id = g.parent_id
      AND p.type = 'call' AND g.type = 'argument_list'
"""

_DELETED = """(p.type = 'delete_statement'
                   OR (p.type = 'expression_list' AND q.type = 'delete_statement'))"""

_REFERENCES = f"""
    WITH identifiers AS (
      {_UNITS}
      WHERE (({symbols_common.LOAD_FILTER} AND p.type <> 'attribute') OR (i.type = 'identifier' AND {_DELETED}))
        AND i.name NOT IN ('self', 'cls')
    ), attributes AS (
      {_UNITS}
      WHERE i.type = 'attribute' AND NOT (p.type = 'attribute' AND i.sibling_index = 0)
        AND (TRUE {STORES} OR {_DELETED})
    ), quoted AS (
      SELECT s.fact_id, s.node_id, s.start_line AS line, s.type, c.name AS expression,
             'access' AS kind,
             greatest(coalesce(s.scope.function, -1), coalesce(s.scope.class, -1), 0)
               AS scope_node, s.scope.class AS class_node
      FROM ast_nodes s
      JOIN ast_nodes p ON p.fact_id = s.fact_id AND p.node_id = s.parent_id AND p.type = 'type'
      JOIN ast_nodes c ON c.fact_id = s.fact_id AND c.parent_id = s.node_id
        AND c.type = 'string_content'
      WHERE s.type = 'string' AND regexp_matches(c.name, '^[A-Za-z_][A-Za-z_0-9.]*$')
    ), units AS (
      SELECT * FROM identifiers
      UNION ALL
      SELECT * FROM attributes
      WHERE kind = 'call' OR (regexp_matches(expression, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)+$')
                              AND NOT regexp_matches(expression, '^(self|cls)\\.'))
      UNION ALL
      SELECT * FROM quoted
    ), uses AS (
      SELECT u.fact_id, s.path, s.module, u.node_id, u.line, u.kind, u.expression,
             s.qualname AS source, c.qualname AS owner
      FROM units u
      JOIN symbol_reach_scopes s ON s.fact_id = u.fact_id AND s.node_id = u.scope_node
      LEFT JOIN symbol_reach_scopes c ON c.fact_id = u.fact_id AND c.node_id = u.class_node
    ), imported AS (
      SELECT i.fact_id, m.path, i.module, i.statement AS node_id, i.line, 'access' AS kind,
             i.target AS expression, i.module AS source, NULL AS owner
      FROM symbol_reach_imports i JOIN symbol_reach_modules m USING (fact_id)
      WHERE i.is_from
    )
    SELECT row_number() OVER (ORDER BY fact_id, node_id, expression)::BIGINT AS ref, *
    FROM (SELECT * FROM uses UNION ALL SELECT * FROM imported)
"""

_SYMBOLS = """
    SELECT qualname,
           arg_min(kind, CASE kind WHEN 'class' THEN 1 WHEN 'function' THEN 2 WHEN 'method' THEN 3
                                   WHEN 'property' THEN 4 WHEN 'module' THEN 5
                                   WHEN 'variable' THEN 6 ELSE 7 END) AS kind,
           arg_max(kind, kind) AS holder_kind
    FROM (SELECT qualname, kind FROM symbol_reach_nodes
          UNION ALL SELECT module, 'module' FROM symbol_reach_modules)
    GROUP BY qualname
"""

_TYPES = """
    WITH declared AS (
      SELECT p.fact_id, p.node_id, CASE WHEN n.type = 'identifier' THEN n.name ELSE s.name END
               AS name, t.node_id AS type_node, coalesce(d.scope.function, 0) AS scope_id
      FROM ast_nodes p
      JOIN ast_nodes n ON n.fact_id = p.fact_id AND n.parent_id = p.node_id AND n.sibling_index = 0
      LEFT JOIN ast_nodes s ON s.fact_id = n.fact_id AND s.parent_id = n.node_id
        AND s.type = 'identifier' AND n.type IN ('list_splat_pattern', 'dictionary_splat_pattern')
      JOIN ast_nodes t ON t.fact_id = p.fact_id AND t.parent_id = p.node_id AND t.type = 'type'
      JOIN ast_nodes q ON q.fact_id = p.fact_id AND q.node_id = p.parent_id
        AND q.type = 'parameters'
      JOIN ast_nodes d ON d.fact_id = p.fact_id AND d.node_id = q.parent_id
        AND d.type = 'function_definition'
      WHERE p.type IN ('typed_parameter', 'typed_default_parameter')
      UNION ALL
      SELECT a.fact_id, a.node_id, n.name, t.node_id, coalesce(a.scope.function, 0)
      FROM ast_nodes a
      JOIN ast_nodes n ON n.fact_id = a.fact_id AND n.parent_id = a.node_id
        AND n.sibling_index = 0 AND n.type = 'identifier'
      JOIN ast_nodes t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id AND t.type = 'type'
      WHERE a.type = 'assignment'
    ), named AS (
      SELECT d.*, CASE c.type WHEN 'identifier' THEN c.name
                              WHEN 'attribute' THEN regexp_extract(c.name, '[^.]*$')
                              WHEN 'generic_type' THEN regexp_extract(g.name, '[^.]*$')
                              WHEN 'string' THEN g.name WHEN 'none' THEN 'None' END AS type_name
      FROM declared d
      JOIN symbol_reach_modules m USING (fact_id)
      JOIN ast_nodes c ON c.fact_id = d.fact_id AND c.parent_id = d.type_node
        AND c.sibling_index = 0
      LEFT JOIN ast_nodes g ON g.fact_id = c.fact_id AND g.parent_id = c.node_id
        AND g.sibling_index = 0 AND c.type IN ('generic_type', 'string')
    )
    SELECT n.fact_id, n.name, n.type_name, n.node_id, n.scope_id,
           CASE WHEN n.scope_id = 0 THEN 1000000000 ELSE n.scope_id + s.descendant_count END
             AS scope_end
    FROM named n LEFT JOIN ast_nodes s ON s.fact_id = n.fact_id AND s.node_id = n.scope_id
    WHERE n.type_name IS NOT NULL AND n.type_name <> ''
"""

_LANDED = f"""
    WITH typed AS (
      SELECT r.ref, arg_max(t.type_name, (t.scope_id, t.node_id)) AS type_name
      FROM symbol_reach_references r
      JOIN symbol_reach_types t ON t.fact_id = r.fact_id
        AND t.name = split_part(r.expression, '.', 1) AND t.node_id < r.node_id
        AND r.node_id BETWEEN t.scope_id AND t.scope_end
      WHERE contains(r.expression, '.') GROUP BY r.ref
    ), expanded AS (
      SELECT r.*, split_part(r.expression, '.', 1) AS head, a.target AS alias_target,
             CASE WHEN contains(r.expression, '.')
                  THEN substr(r.expression, strpos(r.expression, '.') + 1) ELSE '' END AS rest,
             coalesce(b.target, t.type_name) AS typed_expansion
      FROM symbol_reach_references r
      LEFT JOIN symbol_reach_aliases a
        ON a.fact_id = r.fact_id AND a.alias = split_part(r.expression, '.', 1)
      LEFT JOIN typed t ON t.ref = r.ref
      LEFT JOIN symbol_reach_aliases b ON b.fact_id = r.fact_id AND b.alias = t.type_name
    ), named AS (
      SELECT e.*, CASE WHEN alias_target IS NULL THEN expression WHEN rest = '' THEN alias_target
                       ELSE alias_target || '.' || rest END AS expanded
      FROM expanded e
    ), candidates AS (
      SELECT ref, 1 AS position, typed_expansion || '.' || rest AS candidate FROM named
      WHERE typed_expansion IS NOT NULL
      UNION ALL SELECT ref, 2, module || '.' || typed_expansion || '.' || rest FROM named
      WHERE typed_expansion IS NOT NULL
      UNION ALL SELECT ref, 3, owner || '.' || rest FROM named
      WHERE head IN ('self', 'cls') AND owner IS NOT NULL AND rest <> ''
      UNION ALL SELECT ref, 4, module || '.' || expanded FROM named
      UNION ALL SELECT ref, 5, expanded FROM named
      UNION ALL SELECT ref, 6, expression FROM named
    ), direct AS (
      SELECT c.ref, arg_min(c.candidate, c.position) AS target,
             arg_min(s.kind, c.position) AS target_kind
      FROM candidates c JOIN symbol_reach_symbols s ON s.qualname = c.candidate
      GROUP BY c.ref
    ), receivers AS (
      SELECT ref, module, receiver,
             CASE WHEN alias_target IS NULL THEN receiver
                  WHEN contains(receiver, '.')
                  THEN alias_target || '.' || substr(receiver, strpos(receiver, '.') + 1)
                  ELSE alias_target END AS expanded
      FROM (SELECT n.*, regexp_replace(n.expression, '\\.[^.]*$', '') AS receiver
            FROM named n ANTI JOIN direct d USING (ref)
            WHERE n.kind = 'call' AND contains(n.expression, '.'))
    ), receiver_candidates AS (
      SELECT ref, 1 AS position, module || '.' || expanded AS candidate FROM receivers
      UNION ALL SELECT ref, 2, expanded FROM receivers
      UNION ALL SELECT ref, 3, receiver FROM receivers
    ), through_receiver AS (
      SELECT c.ref, arg_min(c.candidate, c.position) AS target,
             arg_min(s.kind, c.position) AS target_kind
      FROM receiver_candidates c JOIN symbol_reach_symbols s ON s.qualname = c.candidate
      GROUP BY c.ref
    ), roots AS (
      SELECT DISTINCT fact_id, split_part(target, '.', 1) AS root FROM symbol_reach_aliases
    )
    SELECT n.ref, n.fact_id, n.path, n.kind, n.source, n.expanded, n.expression,
           coalesce(d.target, t.target) AS target, coalesce(d.target_kind, t.target_kind)
             AS target_kind,
           CASE WHEN d.ref IS NOT NULL THEN CASE WHEN n.kind = 'call' AND d.target_kind = 'class'
                                                 THEN 'instantiate' ELSE n.kind END
                WHEN t.ref IS NOT NULL THEN 'access' END AS edge_kind,
           CASE WHEN d.ref IS NOT NULL OR t.ref IS NOT NULL THEN 'symbol'
                WHEN NOT contains(n.expression, '.')
                     AND list_contains({BUILTIN_LIST}, n.expression) THEN 'external'
                WHEN r.root IS NOT NULL
                     AND regexp_matches(n.expanded, '^[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)*$')
                  THEN 'external'
                ELSE 'unresolved' END AS outcome,
           regexp_extract(n.expression, '[^.:]+$') AS last_name
    FROM named n
    LEFT JOIN direct d USING (ref)
    LEFT JOIN through_receiver t USING (ref)
    LEFT JOIN roots r ON r.fact_id = n.fact_id AND r.root = split_part(n.expanded, '.', 1)
"""

_CLASSES = """
    WITH RECURSIVE bases AS (
      SELECT source AS qualname, target, expression, expanded, kind FROM symbol_reach_landed
      WHERE kind = 'inherit' AND outcome <> 'unresolved'
    ), marked(qualname, role) AS (
      SELECT qualname, 'enum' FROM bases
      WHERE regexp_extract(coalesce(target, expanded), '[^.]*$')
            IN ('Enum', 'Flag', 'IntEnum', 'IntFlag', 'ReprEnum', 'StrEnum')
      UNION SELECT qualname, 'registered' FROM bases
      WHERE regexp_extract(expression, '[^.]*$') IN ('Component', 'Registry')
      UNION SELECT b.qualname, m.role FROM marked m JOIN bases b ON b.target = m.qualname
    ), subclasses AS (
      SELECT s.qualname
      FROM symbol_reach_scopes s
      JOIN ast_nodes a ON a.fact_id = s.fact_id AND a.parent_id = s.node_id
        AND a.type = 'argument_list'
      JOIN ast_nodes b ON b.fact_id = a.fact_id AND b.parent_id = a.node_id
        AND b.type NOT IN ('(', ')', ',', 'comment', 'keyword_argument', 'dictionary_splat')
      WHERE s.type = 'class_definition'
    ), inheriting AS (
      SELECT DISTINCT source AS qualname FROM symbol_reach_landed WHERE kind = 'inherit'
      UNION SELECT DISTINCT target FROM bases WHERE target IS NOT NULL
      UNION SELECT DISTINCT qualname FROM subclasses
    )
    SELECT qualname, role FROM marked
    UNION ALL SELECT qualname, 'inheriting' FROM inheriting
"""

_DECLARATIONS = """
    WITH edges AS (
      SELECT l.*, split_part(m.module, '.', 1) AS package
      FROM symbol_reach_landed l JOIN symbol_reach_modules m USING (fact_id)
      WHERE l.outcome = 'symbol' AND l.target_kind <> 'module'
    ), arrivals AS (
      SELECT target AS qualname, target_kind AS kind, path, source, edge_kind, package FROM edges
      UNION ALL
      SELECT h.qualname, h.holder_kind, e.path, e.source, e.edge_kind, e.package
      FROM edges e
      JOIN symbol_reach_symbols h ON h.qualname = regexp_replace(e.target, '\\.[^.]*$', '')
      WHERE contains(e.target, '.')
    ), unresolved AS (
      SELECT last_name, count(*) AS unresolved
      FROM symbol_reach_landed WHERE outcome = 'unresolved' GROUP BY last_name
    ), declared AS (
      SELECT n.*, regexp_replace(n.qualname, '\\.[^.]*$', '') AS owner,
             regexp_replace(n.qualname, '^.*\\.', '') AS tail
      FROM symbol_reach_nodes n
      WHERE NOT (n.kind = 'attribute' AND regexp_replace(n.qualname, '\\.[^.]*$', '')
                 IN (SELECT qualname FROM symbol_reach_classes WHERE role = 'enum'))
    ), measured AS (
      SELECT d.*, count(a.path) AS referenced, count(a.path) FILTER (a.path = d.path) AS own,
             count(a.path) FILTER (a.source = d.owner OR starts_with(a.source, d.owner || '.'))
               AS owned,
             count(a.path) FILTER (a.edge_kind = 'call') AS calls,
             count(a.path) FILTER (a.edge_kind = 'instantiate') AS instantiations,
             count(a.path) FILTER (a.edge_kind = 'inherit') AS inheritances,
             count(DISTINCT a.path) AS files,
             count(DISTINCT CASE WHEN contains(a.path, '/')
                                 THEN regexp_replace(a.path, '/[^/]*$', '') ELSE '' END)
               FILTER (a.path IS NOT NULL) AS directories,
             count(DISTINCT a.package) AS packages
      FROM declared d
      LEFT JOIN arrivals a ON a.qualname = d.qualname AND a.kind = d.kind
      GROUP BY ALL
    )
    SELECT f.fact_order::UBIGINT AS fact_order, f.fact_id,
           f.fact_id AS parent_id,
           f.fact_id || '/declarations:' || (row_number() OVER w - 1) AS record_id,
           (row_number() OVER w - 1)::UBIGINT AS ordinal,
           d.calls AS call_count, 0 AS import_count, d.inheritances AS inherit_count,
           d.instantiations AS instantiate_count,
           d.decorated OR EXISTS (SELECT 1 FROM symbol_reach_classes c
                                  WHERE c.qualname = d.qualname AND c.role = 'registered')
             AS is_decorated,
           d.owner_type = 'module' AND d.kind IN ('class', 'function', 'variable')
             AS is_module_scope,
           d.kind, d.referenced - d.owned AS non_owner_references,
           d.referenced - d.own AS other_file_references, d.own AS own_file_references,
           EXISTS (SELECT 1 FROM symbol_reach_classes c
                   WHERE c.qualname = d.owner AND c.role = 'inheriting') AS owner_has_inheritance,
           d.owned AS owner_references,
           CASE WHEN o.holder_kind = 'module' THEN 'public' ELSE symbols_visibility(o.tail) END
             AS owner_visibility,
           d.qualname, d.directories AS referencing_directories, d.files AS referencing_files,
           d.packages AS referencing_packages, d.path AS span_path, d.line AS span_start_line,
           0 AS span_start_column, d.line AS span_end_line, 0 AS span_end_column,
           coalesce(u.unresolved, 0) AS unresolved_name_references,
           symbols_visibility(d.tail) AS visibility
    FROM measured d
    JOIN symbol_reach_facts f ON f.path = d.path
    LEFT JOIN (SELECT qualname, holder_kind, regexp_replace(qualname, '^.*\\.', '') AS tail
               FROM symbol_reach_symbols) o ON o.qualname = d.owner
    LEFT JOIN unresolved u ON u.last_name = d.tail
    WINDOW w AS (PARTITION BY d.path ORDER BY d.kind, d.qualname)
"""

_RELATIONS = {
    "facts": _FACTS,
    **import_relations("symbol_reach") | {"modules": _MODULES},
    "scopes": _SCOPES,
    "nodes": _NODES,
    "references": _REFERENCES,
    "symbols": _SYMBOLS,
    "types": _TYPES,
    "landed": _LANDED,
    "classes": _CLASSES,
    "declarations": _DECLARATIONS,
}


class SymbolsVisibility(Function):
    """Python's visibility from a name alone: a dunder is public, a leading double underscore
    private, a single one internal."""

    name: str

    expression = """
        CASE WHEN starts_with(name, '__') AND ends_with(name, '__') THEN 'public'
             WHEN starts_with(name, '__') THEN 'private'
             WHEN starts_with(name, '_') THEN 'internal' ELSE 'public' END
    """


class SymbolReach(Family):
    family = "SymbolReachFact"
    relations = _RELATIONS
