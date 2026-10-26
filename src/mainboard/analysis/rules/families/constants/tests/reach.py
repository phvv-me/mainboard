"""Which production declarations a test reaches, from the repository's resolved calls.

mcmr walks its symbol graph from each test; here the walk is over `call_calls`: every call whose
target resolved to a first-party function, method or class (or to a first-party name that only a
package re-exports), from the innermost function holding it to the declaration its target names
(a declaration whose dotted module path ends with the target's module part and whose nesting names
are the rest). A target the call resolver left on a re-exporting package (`pkg.Name` where
`pkg/__init__.py` imports `Name` from `pkg.mod`) is followed through those module-level imports to
the module that declares it. A test reaches what its own calls name (`direct`) and, from those,
what the functions they call call in turn. References that are not calls (a function passed as a
value) are not edges. A declaration is named by its module (the path below the nearest directory
above its outermost package) and its nesting names.

TestFunctionFact's relations after the behavior's, each reading the earlier ones by their view
  names.
"""

REACH = {
    "modules": r"""
        WITH packages AS (
          SELECT DISTINCT regexp_replace(path, '/?[^/]*$', '') AS dir FROM ast_files
          WHERE language = 'python' AND regexp_matches(path, '(^|/)__init__\.pyi?$')
        ), roots AS (
          SELECT DISTINCT regexp_replace(p.dir, '/?[^/]*$', '') AS root FROM packages p
          WHERE p.dir <> '' AND regexp_replace(p.dir, '/?[^/]*$', '') <> ''
            AND NOT EXISTS (SELECT 1 FROM packages q WHERE q.dir <> ''
                            AND starts_with(p.dir, q.dir || '/'))
        )
        SELECT f.fact_id, f.path,
               replace(regexp_replace(regexp_replace(
                 CASE WHEN r.root IS NULL THEN f.path ELSE substr(f.path, length(r.root) + 2) END,
                 '\.pyi?$', ''), '(^|/)__init__$', ''), '/', '.') AS module,
               regexp_matches(f.path, '(^|/)__init__\.pyi?$') AS is_package
        FROM ast_files f LEFT JOIN roots r ON starts_with(f.path, r.root || '/')
        WHERE f.language = 'python'
        QUALIFY row_number() OVER (PARTITION BY f.fact_id
                                   ORDER BY length(r.root) DESC NULLS LAST) = 1
    """,
    "declarations": r"""
        SELECT n.path, n.node_id, m.module,
               array_to_string(list_transform(n.qualified_name, lambda q: q.name), '.')
                 AS qualname,
               classes_is_test(n.path) AS is_test
        FROM ast_nodes n JOIN test_function_modules m ON m.fact_id = n.fact_id
        WHERE n.type IN ('function_definition', 'class_definition')
    """,
    "lambdas": """
        SELECT fact_id, node_id, list_filter(scope.stack, lambda s: s.kind = 240)[-1].id AS home
        FROM ast_nodes WHERE type = 'lambda' AND scope.stack IS NOT NULL
    """,
    "call_edges": """
        SELECT DISTINCT c.node_path AS path, coalesce(l.home, n.scope['function']) AS caller,
               regexp_extract(c.target_id, '^python:[a-z-]+:(.*)$', 1) AS tail
        FROM call_calls c
        JOIN ast_nodes n ON n.path = c.node_path AND n.type = 'call'
          AND n.start_line = c.node_start_line AND n.start_column = c.node_start_column
          AND n.end_line = c.node_end_line AND n.end_column = c.node_end_column
        LEFT JOIN test_function_lambdas l ON l.fact_id = n.fact_id
          AND l.node_id = n.scope['function']
        WHERE (split_part(c.target_id, ':', 2) IN ('function', 'method', 'class')
               OR (split_part(c.target_id, ':', 2) = 'external-symbol' AND c.is_first_party))
          AND n.scope['function'] IS NOT NULL
    """,
    "reexports": """
        WITH imports AS (
          SELECT i.fact_id, i.node_id, length(i.name) - length(ltrim(i.name, '.')) AS level,
                 ltrim(i.name, '.') AS module, m.module AS file_module, m.is_package
          FROM ast_nodes i JOIN test_function_modules m ON m.fact_id = i.fact_id
          WHERE i.type = 'import_from_statement' AND tests_home(i.scope) = -1
        ), origins AS (
          SELECT *, CASE WHEN level = 0 THEN module
                         ELSE array_to_string(list_concat(
                                string_split(file_module, '.')[1:greatest(0,
                                  len(string_split(file_module, '.'))
                                  - CASE WHEN is_package THEN level - 1 ELSE level END)],
                                CASE WHEN module = '' THEN [] ELSE [module] END), '.') END
                       AS origin
          FROM imports
        )
        SELECT o.file_module || '.' || coalesce(a.name, n.name) AS key,
               o.origin || '.' || coalesce(d.name, n.name) AS target,
               coalesce(a.name, n.name) AS name
        FROM origins o
        JOIN ast_nodes n ON n.fact_id = o.fact_id AND n.parent_id = o.node_id
          AND n.sibling_index > 1 AND n.type IN ('dotted_name', 'aliased_import')
        LEFT JOIN ast_nodes d ON d.fact_id = o.fact_id AND d.parent_id = n.node_id
          AND n.type = 'aliased_import' AND d.sibling_index = 0
        LEFT JOIN ast_nodes a ON a.fact_id = o.fact_id AND a.parent_id = n.node_id
          AND n.type = 'aliased_import' AND a.sibling_index = 2
        WHERE o.origin <> ''
    """,
    "reads": r"""
        SELECT DISTINCT x.path, coalesce(l.home, x.scope['function']) AS caller,
               m.module || '.' || x.name AS tail
        FROM ast_nodes x
        JOIN test_function_modules m ON m.fact_id = x.fact_id
        JOIN ast_nodes p ON p.fact_id = x.fact_id AND p.node_id = x.parent_id
        LEFT JOIN test_function_lambdas l ON l.fact_id = x.fact_id
          AND l.node_id = x.scope['function']
        WHERE x.scope['function'] IS NOT NULL
          AND ((x.type = 'attribute' AND p.type <> 'attribute'
                AND regexp_matches(x.name, '^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$'))
               OR (x.type = 'identifier' AND NOT (
                     p.type IN ('attribute', 'dotted_name', 'aliased_import', 'parameters', 'type',
                                'import_from_statement', 'global_statement', 'nonlocal_statement')
                     OR (p.type IN ('function_definition', 'class_definition')
                         AND x.sibling_index = 1)
                     OR (p.type IN ('keyword_argument', 'default_parameter',
                                    'typed_default_parameter', 'typed_parameter')
                         AND x.sibling_index = 0))))
          AND m.module || '.' || split_part(x.name, '.', 1) IN (
            SELECT module || '.' || qualname FROM test_function_declarations
            WHERE NOT contains(qualname, '.')
            UNION SELECT key FROM test_function_reexports)
    """,
    "edges": """
        SELECT path, caller, tail FROM test_function_call_edges
        UNION SELECT path, caller, tail FROM test_function_reads
    """,
    "forwards": """
        WITH RECURSIVE walk(tail, current, depth) AS (
          SELECT DISTINCT tail, tail, 0 FROM test_function_edges
          UNION
          SELECT w.tail, e.target || substr(w.current, length(p.prefix) + 1), w.depth + 1
          FROM walk w,
               LATERAL (SELECT string_split(w.current, '.') AS parts) s,
               LATERAL (SELECT unnest(range(2, len(s.parts) + 1)) AS i) u,
               LATERAL (SELECT array_to_string(s.parts[1:u.i], '.') AS prefix,
                               s.parts[u.i] AS name) p
          JOIN test_function_reexports e
            ON e.name = p.name AND e.key = p.prefix
          WHERE w.depth < 5)
        SELECT DISTINCT tail, current FROM walk
    """,
    # A dotted name resolves to its longest declared prefix: `Model.model_validate_json`, an
    # inherited member with no declaration of its own, reaches the class `Model`.
    "resolutions": """
        WITH prefixes AS (
          SELECT f.tail, f.current, u.i,
                 array_to_string(string_split(f.current, '.')[1:u.i], '.') AS name
          FROM test_function_forwards f,
               LATERAL (SELECT unnest(range(1, len(string_split(f.current, '.')) + 1)) AS i) u)
        SELECT DISTINCT p.tail, d.path, d.node_id, d.is_test
        FROM prefixes p
        JOIN test_function_declarations d ON d.module || '.' || d.qualname = p.name
        QUALIFY p.i = max(p.i) OVER (PARTITION BY p.tail, p.current)
    """,
    "links": """
        SELECT e.path, e.caller, r.path AS callee_path, r.node_id AS callee, e.tail, r.is_test
        FROM test_function_edges e JOIN test_function_resolutions r ON r.tail = e.tail
    """,
    "called": """
        SELECT t.fact_id, t.test, l.callee_path, l.callee, l.tail, l.is_test
        FROM test_function_declared t
        JOIN test_function_links l ON l.path = t.path AND l.caller = t.test
        UNION ALL
        SELECT t.fact_id, t.test, t.path, f.fn, f.name, true
        FROM test_function_declared t
        JOIN test_function_reached r ON r.fact_id = t.fact_id AND r.test = t.test
        JOIN test_function_fixtures f ON f.fact_id = r.fact_id AND f.name = r.name
    """,
    "closure": """
        WITH RECURSIVE walk(root_path, root, path, node_id) AS (
          SELECT DISTINCT callee_path, callee, callee_path, callee FROM test_function_called
          UNION
          SELECT w.root_path, w.root, l.callee_path, l.callee
          FROM walk w JOIN test_function_links l ON l.path = w.path AND l.caller = w.node_id)
        SELECT * FROM walk
    """,
    "named": """
        SELECT path, node_id, module || '.' || qualname AS name
        FROM test_function_declarations WHERE NOT is_test
    """,
}
