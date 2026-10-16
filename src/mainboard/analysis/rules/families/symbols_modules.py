"""mcmr's ModuleCouplingFact and DependencyComponentFact for Python, derived from sitting_duck's
syntax trees: how modules import each other.

Both are two views of the import arrows `symbols_common.import_relations` lands: an import counts
as an arrow from the importing module to the module it names (the module defining a re-exported
name, else the longest dotted prefix some module carries); imports outside the repository and
inside `if TYPE_CHECKING:` are no arrows. Coupling counts the distinct modules on each side;
the components are the strongly connected modules of the same arrows.
"""

from ...catalog import Family
from .symbols_common import chained, import_relations

_COUPLING = import_relations("module_coupling", arrows=True)
_COMPONENT = import_relations("dependency_component", arrows=True)

_CLASSES = """
    classes AS (
      SELECT c.fact_id, m.path,
             m.module || '.' || array_to_string(list_transform(c.qualified_name,
                                                               lambda q: q.name), '.') AS qualname,
             c.node_id
      FROM ast_nodes c JOIN modules m USING (fact_id)
      WHERE c.type = 'class_definition'
      QUALIFY row_number() OVER (PARTITION BY qualname ORDER BY m.path, c.node_id) = 1
    ), bases AS (
      SELECT l.fact_id, l.parent_id AS node_id,
             CASE b.type WHEN 'keyword_argument' THEN arg_max(v.name, v.sibling_index)
                         WHEN 'subscript' THEN arg_min(v.name, v.sibling_index)
                         ELSE any_value(b.name) END AS expression
      FROM ast_nodes l
      JOIN ast_nodes b ON b.fact_id = l.fact_id AND b.parent_id = l.node_id
      LEFT JOIN ast_nodes v ON v.fact_id = b.fact_id AND v.parent_id = b.node_id
      WHERE l.type = 'argument_list' GROUP BY l.fact_id, l.parent_id, b.node_id, b.type
    ), contracts AS (
      SELECT fact_id, node_id FROM bases
      WHERE regexp_extract(expression, '[^.]*$') IN ('ABC', 'ABCMeta', 'Protocol')
      UNION
      SELECT d.fact_id, k.node_id
      FROM ast_nodes d
      JOIN ast_nodes w ON w.fact_id = d.fact_id AND w.node_id = d.parent_id
      JOIN ast_nodes b ON b.fact_id = d.fact_id AND b.node_id = w.parent_id AND b.type = 'block'
      JOIN ast_nodes k ON k.fact_id = d.fact_id AND k.node_id = b.parent_id
        AND k.type = 'class_definition'
      JOIN ast_nodes t ON t.fact_id = d.fact_id AND t.parent_id = d.node_id AND t.sibling_index = 1
      WHERE d.type = 'decorator' AND starts_with(regexp_extract(t.name, '[^.]*$'), 'abstract')
    ), declared AS (
      SELECT c.path, count(*) AS declarations, count(k.node_id) AS abstracts
      FROM classes c LEFT JOIN (SELECT DISTINCT * FROM contracts) k USING (fact_id, node_id)
      GROUP BY c.path
    )
"""

_COUPLING_FACTS = f"""
    WITH {chained("module_coupling", _COUPLING)}, {_CLASSES},
    outward AS (SELECT DISTINCT importer, imported, imported_path
                FROM arrows WHERE importer <> imported),
    efferent AS (SELECT importer AS module, count(*) AS amount FROM outward GROUP BY importer),
    afferent AS (SELECT imported AS module, count(*) AS amount FROM outward GROUP BY imported)
    SELECT (row_number() OVER (ORDER BY m.path) - 1)::UBIGINT AS fact_order,
           'coupling:' || m.module AS fact_id, m.path, 1::UBIGINT AS start_line,
           0::UBIGINT AS start_column, 1::UBIGINT AS end_line, 0::UBIGINT AS end_column,
           'python' AS language, m.module, coalesce(a.amount, 0)::BIGINT AS afferent_count,
           coalesce(e.amount, 0)::BIGINT AS efferent_count,
           coalesce(d.declarations, 0)::BIGINT AS declaration_count,
           coalesce(d.abstracts, 0)::BIGINT AS abstract_declaration_count,
           coalesce((SELECT list(o.imported ORDER BY o.imported_path) FROM outward o
                     WHERE o.importer = m.module), []) AS imported
    FROM modules m
    LEFT JOIN afferent a ON a.module = m.module
    LEFT JOIN efferent e ON e.module = m.module
    LEFT JOIN declared d ON d.path = m.path
"""

_COUPLING_DEPENDENCIES = """
    SELECT f.fact_order, f.fact_id, f.fact_id AS parent_id,
           f.fact_id || '/dependencies:' || (n.ordinal - 1) AS record_id,
           (n.ordinal - 1)::UBIGINT AS ordinal, d.afferent_count, d.efferent_count,
           n.module
    FROM module_coupling_facts f
    CROSS JOIN LATERAL (
      SELECT unnest(range(1, len(f.imported) + 1)) AS ordinal,
             unnest(f.imported) AS module) n
    JOIN module_coupling_facts d ON d.module = n.module
"""

_COMPONENT_FACTS = """
    SELECT 0::UBIGINT AS fact_order, 'dependencies:repository' AS fact_id, '' AS path,
           1::UBIGINT AS start_line, 0::UBIGINT AS start_column, 1::UBIGINT AS end_line,
           0::UBIGINT AS end_column, NULL::VARCHAR AS language
"""

_COMPONENT_EDGES = """
    WITH RECURSIVE pairs AS (
      SELECT DISTINCT importer AS a, imported AS b FROM dependency_component_arrows
      WHERE importer <> imported
    ), reach(a, b) AS (
      SELECT a, b FROM pairs
      UNION SELECT r.a, p.b FROM reach r JOIN pairs p ON p.a = r.b
    ), mutual AS (
      SELECT r.a AS module, least(r.a, min(r.b)) AS leader
      FROM reach r JOIN reach s ON s.a = r.b AND s.b = r.a GROUP BY r.a
    ), members AS (
      SELECT importer AS module FROM dependency_component_arrows
      UNION SELECT imported FROM dependency_component_arrows
    ), components AS (
      SELECT m.module, dense_rank() OVER (ORDER BY coalesce(x.leader, m.module)) - 1 AS component
      FROM members m LEFT JOIN mutual x USING (module)
    )
    SELECT 0::UBIGINT AS fact_order, 'dependencies:repository' AS fact_id,
           'dependencies:repository' AS parent_id,
           'dependencies:repository/import_edges:' || (row_number() OVER w - 1) AS record_id,
           (row_number() OVER w - 1)::UBIGINT AS ordinal, a.line, a.importer_path AS path,
           a.importer AS source, s.component AS source_component, a.imported AS target,
           t.component AS target_component
    FROM dependency_component_arrows a
    JOIN components s ON s.module = a.importer
    JOIN components t ON t.module = a.imported
    WINDOW w AS (ORDER BY a.importer_path, a.imported_path)
"""


class ModuleCoupling(Family):
    family = "ModuleCouplingFact"
    verified = True
    relations = {"facts": _COUPLING_FACTS, "dependencies": _COUPLING_DEPENDENCIES}


class DependencyComponent(Family):
    family = "DependencyComponentFact"
    verified = True
    relations = {"facts": _COMPONENT_FACTS, **_COMPONENT, "import_edges": _COMPONENT_EDGES}
