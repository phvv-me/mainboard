"""mcmr's ExportFact for Python, derived from the import bindings and the `__all__` strings: one
fact per explicitly exported name, how many other files consume its public route, and the
imports that reach its target around the shortest route.

mcmr resolves imports through a repository graph; here a reference is what a file states: the
absolute module an import statement names (`ImportExpression`) and the attribute chains it spells
(`pkg.name`, `pkg.name.attr`). The imports come from the import binding views, whichever engine
filled them, so a bypass names the same module node an `import_binding_facts` row does.
"""

from .....catalog import Family, Function
from ...constants.imports_common import ALL_STRINGS, MODULE_NAMES
from .. import common as imports_common


class ImportAbsoluteModule(Function):
    """The dotted module a `from` statement names once its leading dots are counted against the
    importing module (a package initializer counts them from itself); more dots than parents
    leave the dots in front, as mcmr does."""

    level: int
    module: str
    importer: str
    is_package: bool

    expression = """
        CASE WHEN level = 0 THEN module
             WHEN level > len(string_split(importer, '.')) - CASE WHEN is_package THEN 0 ELSE 1 END
             THEN repeat('.', level::BIGINT) || module
             ELSE array_to_string(list_filter(
                    [array_to_string(string_split(importer, '.')[1:len(string_split(importer, '.'))
                       - CASE WHEN is_package THEN 0 ELSE 1 END + 1 - level], '.'), module],
                    lambda part: part <> ''), '.') END
    """


class ImportExpression(Function):
    """The dotted name one import binding reaches: the module of `import a.b`, or the module of
    `from m import n` joined to `n`."""

    declaration_text: str
    relative_level: int
    module: str
    imported_name: str
    importer: str
    is_package: bool

    expression = """
        CASE WHEN starts_with(declaration_text, 'import') THEN imported_name
             ELSE import_absolute_module(relative_level, module, importer, is_package)
                  || '.' || imported_name END
    """


_IMPORTED = """
  imported AS (
    SELECT i.path, i.importer_module AS module, i.name, i.is_type_only, i.module_node_id,
           i.start_line, i.start_column, b.start_line AS binding_line,
           b.start_column AS binding_column,
           import_expression(i.declaration_text, i.relative_level, i.module, i.imported_name,
                             i.importer_module, ends_with(i.path, '__init__.py')) AS expression,
           count(*) OVER (PARTITION BY i.path, i.declaration_id) AS binding_count
    FROM import_binding_facts i
    JOIN import_binding_nodes b ON b.path = i.path AND b.node_id = i.binding_id
      AND b.role = 'binding'
    WHERE i.language = 'python'
  )
"""

_EXPORTED = """
  exported AS (
    SELECT DISTINCT s.path, m.module, s.value AS name
    FROM all_strings s
    JOIN python_files p ON p.fact_id = s.fact_id
    JOIN module_names m ON m.fact_id = s.fact_id
  ),
  targets AS (
    SELECT path, name,
           arg_max(expression, (start_line, start_column, binding_line, binding_column)) AS target
    FROM imported WHERE NOT is_type_only GROUP BY path, name
  ),
  exports AS (
    SELECT e.path, e.module, e.name, e.module || '.' || e.name AS public,
           coalesce(t.target, e.module || '.' || e.name) AS target
    FROM exported e LEFT JOIN targets t ON t.path = e.path AND t.name = e.name
  )
"""

_CONSUMERS = """
  prefixes AS (
    SELECT DISTINCT path, unnest(list_transform(range(1, len(string_split(expression, '.')) + 1),
             lambda k: array_to_string(string_split(expression, '.')[1:k], '.'))) AS route
    FROM imported
  ),
  chains AS (
    SELECT DISTINCT a.path, a.name AS route
    FROM ast_nodes a JOIN python_files p ON p.fact_id = a.fact_id
    WHERE a.type = 'attribute' AND a.name IN (SELECT public FROM exports)
  ),
  consumers AS (
    SELECT e.path, e.name, count(DISTINCT c.path) AS count
    FROM exports e
    JOIN (SELECT * FROM prefixes UNION SELECT * FROM chains) c ON c.route = e.public
      AND c.path <> e.path
    GROUP BY e.path, e.name
  )
"""

_BYPASSES = """
  WITH RECURSIVE {imported}, {names},
  named AS (
    -- An export renaming its target (`tracer as nvidia_tracer`) is no route: an import taking
    -- it would have to change the name it binds, not only the module it names.
    SELECT * FROM export_facts
    WHERE public_export_name = regexp_extract(public_export_target, '[^.]+$')
  ),
  chosen AS (
    SELECT e.* FROM named e
    JOIN (SELECT public_export_target AS target,
                 min_by(public_export_module || '.' || public_export_name,
                        (len(string_split(public_export_module || '.' || public_export_name, '.')),
                         public_export_module || '.' || public_export_name)) AS route
          FROM named GROUP BY ALL) p
      ON p.target = e.public_export_target
     AND p.route = e.public_export_module || '.' || e.public_export_name
  ),
  imports_of AS (
    SELECT DISTINCT module AS source, expression FROM imported WHERE NOT is_type_only
  ),
  edges AS (
    SELECT i.source, max_by(m.module, length(m.module)) AS target
    FROM (SELECT source, expression,
                 unnest(list_transform(range(1, len(string_split(expression, '.')) + 1),
                   lambda k: array_to_string(string_split(expression, '.')[1:k], '.'))) AS prefix
          FROM imports_of) i
    JOIN module_names m ON m.module = i.prefix
    GROUP BY i.source, i.expression
  ),
  reach(origin, module) AS (
    SELECT DISTINCT public_export_module, public_export_module FROM chosen
    UNION
    SELECT r.origin, e.target FROM reach r JOIN edges e ON e.source = r.module
  ),
  origins AS (
    SELECT c.*, regexp_replace(public_export_target, '\\.[^.]*$', '') AS home,
           public_export_module || '.' || public_export_name AS public
    FROM chosen c
  ),
  found AS (
    SELECT o.fact_order, o.fact_id, o.public, o.public_export_name, o.public_export_module,
           o.public_export_target, o.home, r.*, n.text AS node_text,
           n.start_line AS node_start_line, n.start_column AS node_start_column,
           n.end_line AS node_end_line, n.end_column AS node_end_column
    FROM origins o
    JOIN imported r ON r.expression = o.public_export_target AND NOT r.is_type_only
      AND r.path <> o.path
      AND NOT (r.module = o.public_export_module
               OR starts_with(r.module, o.public_export_module || '.'))
      AND NOT (contains(o.public_export_module, '.')
        AND instr(o.path, split_part(o.public_export_module, '.', 1) || '/') > 0
        AND starts_with(r.path, substr(o.path, 1, instr(o.path,
              split_part(o.public_export_module, '.', 1) || '/') - 1
              + length(split_part(o.public_export_module, '.', 1)) + 1)))
      AND NOT (r.module = regexp_replace(o.home, '\\.[^.]*$', '')
               OR starts_with(r.module, regexp_replace(o.home, '\\.[^.]*$', '') || '.'))
      AND r.expression <> o.public
    LEFT JOIN (SELECT DISTINCT path, node_id, text, start_line, start_column, end_line, end_column
               FROM import_binding_nodes WHERE role = 'module') n
      ON n.path = r.path AND n.node_id = r.module_node_id
  )
  SELECT f.fact_order, f.fact_id, f.fact_id AS parent_id,
         f.fact_id || '/bypasses:' || (row_number() OVER w - 1) AS record_id,
         (row_number() OVER w - 1)::UBIGINT AS ordinal, f.binding_count::BIGINT AS binding_count,
         f.expression, NOT EXISTS (SELECT 1 FROM reach x WHERE x.origin = f.public_export_module
                                     AND x.module = f.module) AS is_cycle_safe,
         f.module_node_id, 'module' AS module_node_kind,
         f.node_end_column AS module_node_span_end_column,
         f.node_end_line AS module_node_span_end_line,
         CASE WHEN f.node_text IS NOT NULL THEN f.path END AS module_node_span_path,
         f.node_start_column AS module_node_span_start_column,
         f.node_start_line AS module_node_span_start_line, f.node_text AS module_node_text,
         f.public_export_name AS name, f.public_export_module AS public_module,
         CASE WHEN f.node_text IS NULL THEN NULL
              WHEN f.node_text = f.home THEN f.public_export_module
              ELSE CASE WHEN starts_with(f.public_export_module, owner)
                        THEN ltrim(substr(f.public_export_module, length(owner) + 1), '.') END
              END AS replacement_module,
         0::BIGINT AS span_end_column, f.start_line::BIGINT AS span_end_line, f.path AS span_path,
         0::BIGINT AS span_start_column, f.start_line::BIGINT AS span_start_line,
         f.public_export_target AS target
  FROM (SELECT *, rtrim(coalesce(CASE WHEN ends_with(home, node_text)
                                      THEN left(home, length(home) - length(node_text)) END, ''),
                        '.') AS owner FROM found) f
  WINDOW w AS (PARTITION BY f.fact_id ORDER BY f.path, f.start_line, f.start_column,
                                              f.binding_line, f.binding_column)
"""


class Export(Family):
    family = "ExportFact"
    relations = {
        "facts": f"""
            WITH {imports_common.python_files("python_files")}, {MODULE_NAMES}, {ALL_STRINGS}, {_IMPORTED}, {_EXPORTED}, {_CONSUMERS}
            SELECT (row_number() OVER (ORDER BY e.path, e.name) - 1)::UBIGINT AS fact_order,
                   'export:' || e.public AS fact_id, e.path, 1::UBIGINT AS start_line,
                   0::UBIGINT AS start_column, 1::UBIGINT AS end_line, 0::UBIGINT AS end_column,
                   'python' AS language,
                   coalesce(c.count, 0)::BIGINT AS public_export_consumer_count,
                   e.module AS public_export_module, e.name AS public_export_name,
                   e.path AS public_export_span_path, 1::BIGINT AS public_export_span_start_line,
                   0::BIGINT AS public_export_span_start_column,
                   1::BIGINT AS public_export_span_end_line,
                   0::BIGINT AS public_export_span_end_column, e.target AS public_export_target
            FROM exports e LEFT JOIN consumers c ON c.path = e.path AND c.name = e.name
        """,
        "public_export_nodes": f"""
            WITH {imports_common.python_files("python_files")}, {MODULE_NAMES}, {ALL_STRINGS}
            SELECT 'export:' || m.module || '.' || s.value AS fact_id,
                   (row_number() OVER (PARTITION BY m.module, s.value ORDER BY s.node_id)
                    - 1)::UBIGINT AS ordinal,
                   s.path || ':' || s.node_id || ':sequence-item' AS id, s.path AS span_path,
                   s.start_line::BIGINT AS span_start_line, s.start_column::BIGINT
                     AS span_start_column, s.end_line::BIGINT AS span_end_line,
                   s.end_column::BIGINT AS span_end_column
            FROM all_strings s
            JOIN python_files p ON p.fact_id = s.fact_id
            JOIN module_names m ON m.fact_id = s.fact_id
        """,
        "bypasses": _BYPASSES.format(imported=_IMPORTED.strip(), names=MODULE_NAMES.strip()),
    }
