"""mcmr's SymbolFact for Python, derived from sitting_duck's syntax trees: every name a file binds
at module scope by assignment (and its module-level predicates returning `bool`) and where the
file reads it.

mcmr lists class-scope and local names too, but no rule reads them, so they are left out. The
references are every read of the name in the file, found from the tree, not the list mcmr walks by
hand: it misses names in slices, `elif` tests and the right side of `:=`, which makes a rename
built on it unsafe.
"""

from ...catalog import Family
from .imports_common import PARSED
from .symbols_common import LOAD_FILTER, LOAD_JOINS

_FACTS = f"""
    WITH {PARSED}
    SELECT (row_number() OVER (ORDER BY f.path) - 1)::UBIGINT AS fact_order,
           'symbolfact:' || f.path AS fact_id, f.path, 1::UBIGINT AS start_line,
           0::UBIGINT AS start_column, coalesce(e.end_line, 1)::UBIGINT AS end_line,
           coalesce(e.end_column, 0)::UBIGINT AS end_column, 'python' AS language
    FROM parsed f LEFT JOIN ast_nodes e ON e.fact_id = f.fact_id AND e.node_id = 0
"""

_NAMES = (
    "WITH "
    + PARSED
    + r"""
    , assigned AS (
      SELECT a.fact_id, l.node_id AS declaration, l.name, l.start_line, l.start_column,
             l.end_line, l.end_column, NOT regexp_matches(l.name, '\p{Ll}') AS is_constant,
             false AS returns_boolean, 0 AS source_order
      FROM ast_nodes a
      JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
        AND l.sibling_index = 0 AND l.type = 'identifier'
      WHERE a.type = 'assignment' AND a.scope.function IS NULL AND a.scope.class IS NULL
        AND a.fact_id IN (SELECT fact_id FROM parsed)
    ), predicates AS (
      SELECT d.fact_id, n.node_id, n.name, n.start_line, n.start_column, n.end_line, n.end_column,
             false, true, 1
      FROM ast_nodes d
      JOIN ast_nodes n ON n.fact_id = d.fact_id AND n.parent_id = d.node_id
        AND n.type = 'identifier' AND n.sibling_index = 1
      JOIN ast_nodes r ON r.fact_id = d.fact_id AND r.parent_id = d.node_id AND r.type = 'type'
      WHERE d.type = 'function_definition' AND d.parent_id = 0
        AND d.fact_id IN (SELECT fact_id FROM parsed)
        AND regexp_extract(regexp_extract(trim(r.name, '"'''), '^[^\[]*'), '[^.]*$') = 'bool'
    )
    SELECT p.path, s.fact_id, s.declaration, s.name, s.start_line, s.start_column, s.end_line,
           s.end_column, s.is_constant, s.returns_boolean,
           (row_number() OVER (PARTITION BY s.fact_id ORDER BY s.source_order, s.start_line,
                               s.start_column) - 1)::UBIGINT AS ordinal
    FROM (SELECT * FROM assigned UNION ALL SELECT * FROM predicates) s
    JOIN parsed p USING (fact_id)
"""
)

_READS = (
    """
    SELECT i.fact_id, f.path, i.name, i.node_id, i.start_line, i.start_column, i.end_line,
           i.end_column
    FROM ast_nodes i
    JOIN ast_files f ON f.fact_id = i.fact_id
    """
    + LOAD_JOINS
    + """
    WHERE (i.fact_id, i.name) IN (SELECT fact_id, name FROM symbol_names) AND """
    + LOAD_FILTER
)

_DYNAMIC = r"""
    SELECT f.fact_id,
           NOT EXISTS (SELECT 1 FROM ast_nodes a
                       JOIN ast_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
                         AND l.sibling_index = 0 AND l.type = 'identifier' AND l.name = '__all__'
                       LEFT JOIN ast_nodes t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id
                         AND t.type = 'type'
                       WHERE a.fact_id = f.fact_id AND a.type = 'assignment' AND t.node_id IS NULL
                         AND a.scope.function IS NULL AND a.scope.class IS NULL
                         AND a.parent_id IN (SELECT e.node_id FROM ast_nodes e
                                             WHERE e.fact_id = f.fact_id AND e.parent_id = 0))
           AND NOT EXISTS (SELECT 1 FROM ast_nodes c
                           JOIN ast_nodes g ON g.fact_id = c.fact_id AND g.parent_id = c.node_id
                             AND g.sibling_index = 0 AND g.type = 'identifier'
                             AND g.name IN ('getattr', 'setattr', 'globals', 'vars', 'eval',
                                            'exec')
                           WHERE c.fact_id = f.fact_id AND c.type = 'call') AS reachable
    FROM ast_files f
"""

_SYMBOLS = """
    SELECT f.fact_order, f.fact_id, f.fact_id AS parent_id,
           f.fact_id || '/symbols:' || n.ordinal AS record_id, n.ordinal,
           n.is_constant AS is_constant_assignment, n.name, 'module' AS scope, n.returns_boolean,
           d.reachable AND NOT starts_with(n.name, '__') AS reference_are_references_complete,
           n.path || ':' || n.name AS reference_id, n.name AS reference_name,
           n.path || ':' || n.declaration || ':name' AS reference_declaration_id,
           'name' AS reference_declaration_kind, n.path AS reference_declaration_span_path,
           n.start_line::BIGINT AS reference_declaration_span_start_line,
           n.start_column::BIGINT AS reference_declaration_span_start_column,
           n.end_line::BIGINT AS reference_declaration_span_end_line,
           n.end_column::BIGINT AS reference_declaration_span_end_column,
           n.name AS reference_declaration_text, count(r.node_id)::BIGINT AS reference_count,
           NULL::VARCHAR AS path, NULL::BIGINT AS start_line, NULL::BIGINT AS start_column,
           NULL::BIGINT AS end_line, NULL::BIGINT AS end_column
    FROM symbol_names n
    JOIN symbol_facts f ON f.path = n.path
    JOIN symbol_dynamic d ON d.fact_id = n.fact_id
    LEFT JOIN symbol_reads r ON r.path = n.path AND r.name = n.name
    GROUP BY ALL
"""

_REFERENCES = """
    SELECT s.fact_order, s.fact_id, s.record_id AS parent_id,
           s.record_id || '/symbols.reference.references:'
             || (row_number() OVER w - 1) AS record_id,
           (row_number() OVER w - 1)::UBIGINT AS ordinal, s.reference_declaration_span_path
             || ':' || r.node_id || ':name' AS id, 'name' AS kind, r.name AS text,
           s.reference_declaration_span_path AS span_path, r.start_line::BIGINT AS span_start_line,
           r.start_column::BIGINT AS span_start_column, r.end_line::BIGINT AS span_end_line,
           r.end_column::BIGINT AS span_end_column
    FROM symbol_symbols s
    JOIN symbol_reads r ON r.path = s.reference_declaration_span_path AND r.name = s.name
    WINDOW w AS (PARTITION BY s.record_id ORDER BY r.start_line, r.start_column)
"""


class Symbol(Family):
    family = "SymbolFact"
    verified = True
    relations = {
        "facts": _FACTS,
        "names": _NAMES,
        "reads": _READS,
        "dynamic": _DYNAMIC,
        "symbols": _SYMBOLS,
        "symbols.reference.references": _REFERENCES,
    }
