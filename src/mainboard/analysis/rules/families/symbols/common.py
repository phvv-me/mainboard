"""SQL the symbol-side families share: which identifiers of a Python file read a name, and the
builtin names mcmr's resolver treats as external. Plain fragments, because a family's relations
are computed before the rules' macros exist."""

from collections.abc import Mapping

from ..constants.imports_common import MODULE_NAMES, PARSED
from ..constants.symbols_common import STORES
from ..imports.common import name_reads

LOAD_JOINS, LOAD_FILTER = name_reads()
LOAD_FILTER += STORES

_MODULES = f"""
    WITH {PARSED}, {MODULE_NAMES}
    SELECT p.fact_id, p.path, m.module,
           regexp_matches(p.path, '(^|/)__init__\\.pyi?$') AS is_package
    FROM parsed p JOIN module_names m USING (fact_id)
"""

_IMPORTS = """
    WITH statements AS (
      SELECT s.fact_id, m.module, m.is_package, s.node_id AS statement, s.type, s.name,
             s.start_line AS line
      FROM ast_nodes s JOIN @_modules m USING (fact_id)
      WHERE s.type IN ('import_statement', 'import_from_statement')
        AND NOT EXISTS (
          SELECT 1 FROM ast_nodes i
          JOIN ast_nodes t ON t.fact_id = i.fact_id AND t.parent_id = i.node_id
            AND t.sibling_index = 1 AND regexp_matches(t.name, '(^|\\.)TYPE_CHECKING$')
          JOIN ast_nodes b ON b.fact_id = i.fact_id AND b.parent_id = i.node_id
            AND b.sibling_index = 2
          WHERE i.fact_id = s.fact_id AND i.type = 'if_statement'
            AND s.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count)
    ), placed AS (
      SELECT s.*, length(s.name) - length(ltrim(s.name, '.')) AS level,
             ltrim(s.name, '.') AS written,
             list_slice(string_split(s.module, '.'), 1,
                        len(string_split(s.module, '.')) - 1 + s.is_package::INT) AS parts
      FROM statements s
    ), based AS (
      SELECT p.*, CASE WHEN p.type = 'import_statement' THEN ''
                       WHEN p.level = 0 THEN p.written
                       WHEN p.level > len(p.parts) THEN repeat('.', p.level) || p.written
                       ELSE array_to_string(list_filter(
                              p.parts[1:len(p.parts) + 1 - p.level] || [p.written],
                              lambda x: x <> ''), '.') END AS origin
      FROM placed p
    )
    SELECT b.fact_id, b.module, b.statement, b.line, b.type = 'import_from_statement' AS is_from,
           c.name AS alias, coalesce(i.name, c.name) AS imported,
           CASE WHEN b.type = 'import_statement' THEN coalesce(i.name, c.name)
                ELSE b.origin || '.' || coalesce(i.name, c.name) END AS target, c.sibling_index
    FROM based b
    JOIN ast_nodes k ON k.fact_id = b.fact_id AND k.parent_id = b.statement AND k.type = 'import'
    JOIN ast_nodes c ON c.fact_id = b.fact_id AND c.parent_id = b.statement
      AND c.sibling_index > k.sibling_index AND c.type IN ('dotted_name', 'aliased_import')
    LEFT JOIN ast_nodes i ON i.fact_id = c.fact_id AND i.parent_id = c.node_id
      AND c.type = 'aliased_import' AND i.type = 'dotted_name'
"""

_ALIASES = """
    SELECT fact_id, module, alias, arg_max(target, (statement, sibling_index)) AS target
    FROM @_imports GROUP BY ALL
"""


_REEXPORTS = """
    SELECT a.module || '.' || a.alias AS reached, a.target AS bound, d.defining,
           NOT EXISTS (SELECT 1 FROM @_aliases x WHERE x.module = d.defining AND x.alias = a.alias)
             AS ends
    FROM @_aliases a
    CROSS JOIN LATERAL (SELECT regexp_replace(a.target, '\\.[^.]*$', '') AS defining) d
    WHERE NOT contains(a.alias, '.') AND contains(a.target, '.') AND d.defining <> a.module
      AND a.module IN (SELECT module FROM @_modules)
      AND d.defining IN (SELECT module FROM @_modules)
"""

_ARROWS = """
    WITH RECURSIVE refs AS (
      SELECT row_number() OVER ()::BIGINT AS ref, fact_id, module, target, line FROM @_imports
    ), walk(ref, reached, depth) AS (
      SELECT ref, target, 0 FROM refs
      UNION ALL
      SELECT w.ref, r.bound, w.depth + 1
      FROM walk w JOIN @_reexports r ON r.reached = w.reached AND NOT r.ends
      WHERE w.depth < 8
    ), rerouted AS (
      SELECT w.ref, arg_min(r.defining, w.depth) AS defining
      FROM walk w JOIN @_reexports r ON r.reached = w.reached AND r.ends GROUP BY w.ref
    ), candidates AS (
      SELECT ref, 0 AS position, defining AS candidate FROM rerouted
      UNION ALL
      SELECT i.ref, n.size, array_to_string(list_slice(string_split(i.target, '.'), 1, n.size),
                                            '.')
      FROM refs i
      CROSS JOIN LATERAL (SELECT unnest(range(1, len(string_split(i.target, '.')) + 1)) AS size) n
    ), landed AS (
      SELECT c.ref, arg_min(m.module, CASE WHEN c.position = 0 THEN 0 ELSE 1000 - c.position END)
               AS imported
      FROM candidates c JOIN @_modules m ON m.module = c.candidate GROUP BY c.ref
    )
    SELECT r.fact_id, r.module AS importer, l.imported, i.path AS importer_path,
           j.path AS imported_path, min(r.line) AS line
    FROM refs r
    JOIN landed l USING (ref)
    JOIN @_modules i ON i.module = r.module
    JOIN @_modules j ON j.module = l.imported
    GROUP BY ALL
"""


_TEMPLATES = {
    "modules": _MODULES,
    "imports": _IMPORTS,
    "aliases": _ALIASES,
    "reexports": _REEXPORTS,
    "arrows": _ARROWS,
}


def import_relations(prefix: str, *, arrows: bool = False) -> dict[str, str]:
    """The relations every symbol-side family starts from, its views prefixed `prefix`.

    They are `modules` (each parsed file's qualified module name), `imports` (one row per name an
    import statement binds, outside `if TYPE_CHECKING:`, with the absolute dotted name it reaches)
    and `aliases` (what a module's name stands for: the last import binding it). With `arrows`,
    also `reexports` (what a module's from-import re-exports, one step) and `arrows` (every import
    statement landed on the module it names, mcmr's way: the module defining a re-exported name,
    else the longest dotted prefix some module carries; one row per importer and imported module,
    with the first line).
    """
    names = list(_TEMPLATES) if arrows else ["modules", "imports", "aliases"]
    return {name: _TEMPLATES[name].replace("@_", f"{prefix}_") for name in names}


def chained(prefix: str, relations: Mapping[str, str]) -> str:
    """`relations` as the common table expressions of one query.

    Each reads the earlier ones by its bare name instead of its view name.
    """
    return ", ".join(
        f"{name} AS ({sql.replace(f'{prefix}_', '')})" for name, sql in relations.items()
    )
