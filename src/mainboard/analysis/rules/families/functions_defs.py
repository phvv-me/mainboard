"""The identity and signature relations of FunctionFact: each function definition, a narrow copy of
the Python syntax nodes later relations read, the source text of the spans asked for, and a
function's imports, decorators, parameters and body statements.

Relation columns avoid names ending in `_id` (the engine narrows such a column by language
through the relation it matches): nodes are `ast` (the file's tree) and `node`, a function is its
`def` node.
"""

DEFS = r"""
    WITH files AS (
      SELECT fact_id AS ast, path FROM ast_files WHERE language = 'python'
    ), wrapper AS (
      SELECT fact_id AS ast, node_id::INT AS node, start_line::INT AS start_line,
             start_column::INT AS start_column, end_line::INT AS end_line,
             end_column::INT AS end_column
      FROM ast_nodes WHERE type = 'decorated_definition'
    ), fn AS (
      SELECT fact_id AS ast, node_id::INT AS node, parent_id::INT AS parent, name,
             scope.function::INT AS enclosing, scope.class::INT AS class_scope,
             start_line::INT AS start_line, start_column::INT AS start_column,
             end_line::INT AS end_line, end_column::INT AS end_column,
             descendant_count::INT AS size
      FROM ast_nodes WHERE type = 'function_definition'
    ), dangling AS (
      SELECT fn.ast, fn.node, fn.size FROM fn
      SEMI JOIN ast_nodes c ON c.fact_id = fn.ast AND c.type = 'comment'
        AND c.end_line = fn.end_line AND c.end_column = fn.end_column
    ), ending AS (
      SELECT t.ast, t.node, arg_max(n.end_line, n.node_id)::INT AS end_line,
             arg_max(n.end_column, n.node_id)::INT AS end_column
      FROM dangling t
      JOIN ast_nodes n ON n.fact_id = t.ast AND n.node_id BETWEEN t.node AND t.node + t.size
        AND n.children_count = 0 AND n.type <> 'comment'
      GROUP BY t.ast, t.node
    ), placed AS (
      SELECT f.path, fn.ast, fn.node, fn.name, fn.enclosing, w.node AS wrapper,
             coalesce(w.start_line, fn.start_line) AS start_line,
             coalesce(w.start_column, fn.start_column) AS start_column,
             coalesce(e.end_line, w.end_line, fn.end_line) AS end_line,
             coalesce(e.end_column, w.end_column, fn.end_column) AS end_column,
             CASE WHEN fn.enclosing IS NULL AND fn.class_scope IS NULL THEN 'module'
                  WHEN coalesce(fn.class_scope, -1) > coalesce(fn.enclosing, -1) THEN 'method'
                  ELSE 'nested' END AS scope,
             CASE WHEN coalesce(fn.class_scope, -1) > coalesce(fn.enclosing, -1)
                  THEN fn.class_scope END AS owner
      FROM fn JOIN files f ON f.ast = fn.ast
      LEFT JOIN wrapper w ON w.ast = fn.ast AND w.node = fn.parent
      LEFT JOIN ending e ON e.ast = fn.ast AND e.node = fn.node
    )
    SELECT 'function:' || path || ':' || start_line || ':' || start_column || ':' || name
             AS fact_id,
           path || ':' || node || ':function' AS entity_id, path, 'python' AS language,
           start_line, start_column, end_line, end_column, ast, node, wrapper, name, scope,
           owner, enclosing, functions_visibility(name, scope) AS visibility
    FROM placed
"""

NODES = r"""
    SELECT n.fact_id AS ast, n.node_id::INT AS node, n.parent_id::INT AS parent, n.type,
           CASE WHEN n.type IN ('string', 'string_content', 'concatenated_string', 'comment',
                                'interpolation') THEN NULL ELSE n.name END AS name,
           n.sibling_index::INT AS sibling, n.start_line::INT AS sl, n.start_column::INT AS sc,
           n.end_line::INT AS el, n.end_column::INT AS ec, n.descendant_count::INT AS size,
           n.children_count::INT AS kids, n.scope.function::INT AS function_scope,
           n.scope.class::INT AS class_scope
    FROM ast_nodes n SEMI JOIN ast_files f ON f.fact_id = n.fact_id AND f.language = 'python'
"""

FILES = "SELECT DISTINCT ast, path FROM function_defs"

LINES = r"""
    SELECT path, line, text, strlen(text) = length(text) AS plain, code,
           sum(code) OVER (PARTITION BY path ORDER BY line) AS code_through
    FROM (SELECT s.path, generate_subscripts(s.lines, 1) AS line, unnest(s.lines) AS text,
                 (NOT regexp_matches(unnest(s.lines), '^\s*(#.*)?$'))::INT AS code
          FROM (SELECT path, string_split(content, chr(10)) AS lines
                FROM source_files SEMI JOIN function_defs USING (path)) s)
"""

IMPORTS = r"""
    WITH statements AS (
      SELECT n.ast, n.node, n.type, n.name AS text, m.module, m.is_package
      FROM function_nodes n JOIN function_modules m ON m.ast_fact = n.ast
      WHERE n.type IN ('import_statement', 'import_from_statement')
    ), items AS (
      SELECT s.ast, s.node AS statement, s.type AS kind, s.text, s.module, s.is_package,
             c.node AS item, c.type, c.name, c.sibling,
             CASE WHEN c.type = 'aliased_import' THEN d.name ELSE c.name END AS imported
      FROM statements s
      JOIN function_nodes c ON c.ast = s.ast AND c.parent = s.node
        AND c.type IN ('dotted_name', 'aliased_import')
        AND (s.type = 'import_statement' OR c.sibling >= 3)
      LEFT JOIN function_nodes d ON d.ast = c.ast AND d.parent = c.node AND d.sibling = 0
        AND c.type = 'aliased_import'
    ), resolved AS (
      SELECT *, length(text) - length(ltrim(text, '.')) AS level, ltrim(text, '.') AS rest,
             CASE WHEN is_package THEN string_split(module, '.')
                  ELSE list_slice(string_split(module, '.'), 1, -2) END AS parts
      FROM items WHERE kind = 'import_from_statement'
    ), joined AS (
      SELECT *, CASE
        WHEN level = 0 THEN rest
        WHEN level > len(parts) THEN repeat('.', level) || rest
        WHEN len(parts) + 1 - level = 0 THEN rest
        WHEN rest = '' THEN array_to_string(list_slice(parts, 1, len(parts) + 1 - level), '.')
        ELSE array_to_string(list_slice(parts, 1, len(parts) + 1 - level), '.') || '.' || rest
        END AS source_module
      FROM resolved
    )
    SELECT ast, CASE WHEN type = 'aliased_import' THEN name ELSE imported END AS binding,
           imported AS origin,
           CASE WHEN type = 'aliased_import' THEN name ELSE split_part(imported, '.', 1) END
             AS bound,
           CASE WHEN type = 'aliased_import' THEN imported ELSE split_part(imported, '.', 1) END
             AS target
    FROM items WHERE kind = 'import_statement'
    UNION ALL
    SELECT ast, name AS binding, rest AS origin, name AS bound,
           source_module || '.' || imported AS target
    FROM joined
"""

CALLS = r"""
    SELECT c.ast, c.node, c.parent, k.name AS callee, k.type AS callee_type, c.sl, c.sc, c.el,
           c.ec, c.size, c.function_scope, c.class_scope
    FROM function_nodes c
    JOIN function_nodes k ON k.ast = c.ast AND k.parent = c.node AND k.sibling = 0
    WHERE c.type = 'call'
"""

BODIES = r"""
    WITH kids AS (
      SELECT d.ast, d.node AS def, k.node AS child, k.type, k.sl, k.el
      FROM function_defs d JOIN function_nodes k ON k.ast = d.ast AND k.parent = d.node
      WHERE k.type IN ('block', 'parameters', 'async', 'type')
    ), statements AS (
      SELECT b.ast, b.def, s.node, s.type, s.sl, s.el, s.kids,
             row_number() OVER (PARTITION BY b.ast, b.def ORDER BY s.sibling) AS position
      FROM kids b JOIN function_nodes s ON s.ast = b.ast AND s.parent = b.child
      WHERE b.type = 'block' AND s.type <> 'comment'
    ), opening AS (
      SELECT s.ast, s.def, s.node, g.name AS literal
      FROM statements s
      JOIN function_nodes c ON c.ast = s.ast AND c.parent = s.node AND c.sibling = 0
        AND c.type = 'string'
      JOIN ast_nodes g ON g.fact_id = c.ast AND g.node_id = c.node
      WHERE s.position = 1 AND s.type = 'expression_statement' AND s.kids = 1
        AND regexp_matches(g.name, '^[uUrR]*["'']')
    ), counted AS (
      SELECT s.ast, s.def, count(*) FILTER (WHERE s.position > (o.node IS NOT NULL)::INT)
               AS statements,
             min(s.node) FILTER (WHERE s.position = 1 + (o.node IS NOT NULL)::INT) AS first_node,
             min(s.type) FILTER (WHERE s.position = 1 + (o.node IS NOT NULL)::INT) AS first_type,
             min(s.sl) FILTER (WHERE s.position = 1 + (o.node IS NOT NULL)::INT) AS first_line,
             max(s.el) AS last_line
      FROM statements s LEFT JOIN opening o ON o.ast = s.ast AND o.def = s.def
      GROUP BY s.ast, s.def
    )
    SELECT d.ast, d.node AS def,
           max(k.child) FILTER (WHERE k.type = 'block') AS body,
           max(k.child) FILTER (WHERE k.type = 'parameters') AS params,
           max(k.child) FILTER (WHERE k.type = 'type') AS returns,
           bool_or(k.type = 'async') AS is_async,
           any_value(c.statements) AS statements, any_value(c.first_node) AS first_node,
           any_value(c.first_type) AS first_type,
           any_value(c.first_line) AS first_line, any_value(c.last_line) AS last_line,
           any_value(o.literal) AS literal
    FROM function_defs d
    LEFT JOIN kids k ON k.ast = d.ast AND k.def = d.node
    LEFT JOIN counted c ON c.ast = d.ast AND c.def = d.node
    LEFT JOIN opening o ON o.ast = d.ast AND o.def = d.node
    GROUP BY d.ast, d.node
"""
