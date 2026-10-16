"""What each collected test declares and calls: decorators, parameters, fixtures, parametrization,
quarantine markers and calls.

TestFunctionFact's relations after the bodies', each reading the earlier ones by their view names.
A fixture is the nearest `fixture`-decorated function of its name above a test; the fixtures a test
reaches are those its parameters name and those they request in turn.
"""

BEHAVIOR = {
    "imports": """
        SELECT * FROM (
          SELECT i.fact_id, i.end_line, i.end_column,
                 split_part(d.name, '.', 1) AS local, split_part(d.name, '.', 1) AS origin
          FROM test_function_nodes i
          JOIN test_function_nodes d ON d.fact_id = i.fact_id AND d.parent_id = i.node_id
            AND d.type = 'dotted_name'
          WHERE i.type = 'import_statement'
          UNION ALL
          SELECT i.fact_id, i.end_line, i.end_column, a.name, d.name
          FROM test_function_nodes i
          JOIN test_function_nodes a ON a.fact_id = i.fact_id AND a.parent_id = i.node_id
            AND a.type = 'aliased_import'
          JOIN test_function_nodes d ON d.fact_id = a.fact_id AND d.parent_id = a.node_id
            AND d.sibling_index = 0
          WHERE i.type = 'import_statement'
          UNION ALL
          SELECT i.fact_id, i.end_line, i.end_column, n.name, m.name || '.' || n.name
          FROM test_function_nodes i
          JOIN test_function_nodes m ON m.fact_id = i.fact_id AND m.parent_id = i.node_id
            AND m.sibling_index = 1 AND m.type = 'dotted_name'
          JOIN test_function_nodes n ON n.fact_id = i.fact_id AND n.parent_id = i.node_id
            AND n.sibling_index > 1 AND n.type = 'dotted_name'
          WHERE i.type = 'import_from_statement'
          UNION ALL
          SELECT i.fact_id, i.end_line, i.end_column, a.name, m.name || '.' || d.name
          FROM test_function_nodes i
          JOIN test_function_nodes m ON m.fact_id = i.fact_id AND m.parent_id = i.node_id
            AND m.sibling_index = 1 AND m.type = 'dotted_name'
          JOIN test_function_nodes a ON a.fact_id = i.fact_id AND a.parent_id = i.node_id
            AND a.sibling_index > 1 AND a.type = 'aliased_import'
          JOIN test_function_nodes d ON d.fact_id = a.fact_id AND d.parent_id = a.node_id
            AND d.sibling_index = 0
          WHERE i.type = 'import_from_statement')
        QUALIFY count(*) OVER (PARTITION BY fact_id, local) = 1
    """,
    "decorators": """
        SELECT f.fact_id, f.node_id AS fn,
               row_number() OVER (PARTITION BY f.fact_id, f.node_id ORDER BY d.sibling_index) - 1
                 AS ordinal,
               e.node_id AS expr, e.type AS expr_type,
               CASE WHEN e.type IN ('identifier', 'attribute') THEN e.name
                    WHEN e.type = 'call' AND c.type IN ('identifier', 'attribute') THEN c.name
                    ELSE '' END AS mark
        FROM test_function_nodes f
        JOIN test_function_nodes w ON w.fact_id = f.fact_id AND w.node_id = f.parent_id
          AND w.type = 'decorated_definition'
        JOIN test_function_nodes d ON d.fact_id = f.fact_id AND d.parent_id = w.node_id
          AND d.type = 'decorator'
        JOIN test_function_nodes e ON e.fact_id = f.fact_id AND e.parent_id = d.node_id
          AND e.sibling_index = 1
        LEFT JOIN test_function_nodes c ON c.fact_id = f.fact_id AND c.parent_id = e.node_id
          AND c.sibling_index = 0 AND e.type = 'call'
        WHERE f.type = 'function_definition'
    """,
    "fixtures": """
        SELECT d.fact_id, d.fn, f.name,
               row_number() OVER (PARTITION BY d.fact_id, f.name ORDER BY d.fn DESC) AS recency
        FROM test_function_decorators d
        JOIN test_function_nodes f ON f.fact_id = d.fact_id AND f.node_id = d.fn
        WHERE ends_with(d.mark, 'fixture')
        QUALIFY recency = 1
    """,
    "params": """
        SELECT c.fact_id, c.fn,
               row_number() OVER (PARTITION BY c.fact_id, c.fn ORDER BY p.sibling_index) - 1 AS
                 ordinal,
               ltrim(coalesce(nullif(p.name, ''), q.name), '*') AS name
        FROM (SELECT fact_id, test AS fn FROM test_function_declared
              UNION SELECT fact_id, fn FROM test_function_fixtures) c
        JOIN test_function_nodes s ON s.fact_id = c.fact_id AND s.parent_id = c.fn
          AND s.type = 'parameters'
        JOIN test_function_nodes p ON p.fact_id = c.fact_id AND p.parent_id = s.node_id
          AND p.type IN ('identifier', 'typed_parameter', 'default_parameter',
                         'typed_default_parameter', 'list_splat_pattern',
                           'dictionary_splat_pattern')
        LEFT JOIN test_function_nodes q ON q.fact_id = p.fact_id AND q.parent_id = p.node_id
          AND q.sibling_index = 0 AND p.name = ''
    """,
    "reached": """
        WITH RECURSIVE walk AS (
          SELECT p.fact_id, p.fn AS test, p.name FROM test_function_params p
          JOIN test_function_declared t ON t.fact_id = p.fact_id AND t.test = p.fn
          UNION
          SELECT r.fact_id, r.test, q.name FROM walk r
          JOIN test_function_fixtures f ON f.fact_id = r.fact_id AND f.name = r.name
          JOIN test_function_params q ON q.fact_id = f.fact_id AND q.fn = f.fn)
        SELECT * FROM walk
    """,
    "arguments": """
        SELECT d.fact_id, d.fn, d.ordinal AS decorator, a.node_id, a.type, a.name,
               row_number() OVER (PARTITION BY d.fact_id, d.fn, d.ordinal ORDER BY a.sibling_index)
                 AS position
        FROM test_function_decorators d
        JOIN test_function_nodes l ON l.fact_id = d.fact_id AND l.parent_id = d.expr
          AND l.type = 'argument_list'
        JOIN test_function_nodes a ON a.fact_id = d.fact_id AND a.parent_id = l.node_id
          AND a.type NOT IN ('(', ')', 'keyword_argument', 'dictionary_splat', 'comment')
        WHERE d.expr_type = 'call' AND ends_with(d.mark, 'parametrize')
    """,
    "sizes": """
        WITH ranges AS (
          SELECT a.fact_id, a.fn, a.decorator,
                 list(try_cast(i.name AS BIGINT) ORDER BY i.sibling_index)
                   FILTER (WHERE i.type = 'integer') AS bounds
          FROM test_function_arguments a
          JOIN test_function_nodes c ON c.fact_id = a.fact_id AND c.parent_id = a.node_id
            AND c.sibling_index = 0 AND c.type = 'identifier' AND c.name = 'range'
          JOIN test_function_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
            AND l.type = 'argument_list'
          JOIN test_function_nodes i ON i.fact_id = a.fact_id AND i.parent_id = l.node_id
          WHERE a.position = 2 AND a.type = 'call'
          GROUP BY a.fact_id, a.fn, a.decorator)
        SELECT fact_id, fn AS test, decorator,
               CASE len(bounds) WHEN 1 THEN bounds[1] WHEN 2 THEN bounds[2] - bounds[1] END AS size
        FROM ranges
    """,
    "generations": """
        SELECT a.fact_id, a.fn AS test, count(*) AS n
        FROM test_function_arguments a
        JOIN test_function_modular g ON g.fact_id = a.fact_id AND g.name = a.name AND g.generated
        WHERE a.position = 2 AND a.type = 'identifier'
        GROUP BY a.fact_id, a.fn
    """,
    "keywords": """
        SELECT d.fact_id, d.fn, d.ordinal AS decorator, k.name AS key, v.type AS value_type,
               CASE WHEN v.type = 'string' THEN
                      (SELECT coalesce(string_agg(c.name, '' ORDER BY c.node_id), '')
                       FROM test_function_nodes c
                       WHERE c.fact_id = v.fact_id AND c.parent_id = v.node_id
                         AND c.type IN ('string_content', 'escape_sequence'))
                    ELSE v.name END AS value
        FROM test_function_decorators d
        JOIN test_function_declared t ON t.fact_id = d.fact_id AND t.test = d.fn
        JOIN test_function_nodes l ON l.fact_id = d.fact_id AND l.parent_id = d.expr
          AND l.type = 'argument_list'
        JOIN test_function_nodes k ON k.fact_id = d.fact_id AND k.parent_id = l.node_id
          AND k.type = 'keyword_argument'
        JOIN test_function_nodes v ON v.fact_id = d.fact_id AND v.parent_id = k.node_id
          AND v.sibling_index = 2
        WHERE d.expr_type = 'call'
    """,
    "quarantines": """
        SELECT d.fact_id, t.name, d.ordinal, t.test,
               coalesce(CASE WHEN a.value_type = 'integer' THEN try_cast(a.value AS BIGINT) END,
                        CASE WHEN date_diff('day', try_cast(s.value AS DATE), current_date) >= 0
                             THEN date_diff('day', try_cast(s.value AS DATE), current_date) END)
                 AS age_days,
               coalesce(o.value, '') AS owner,
               trim(coalesce(r.value, '')) <> '' AS has_remediation_evidence,
               coalesce(c.value_type = 'true', false) AS recurred_after_repair
        FROM test_function_decorators d
        JOIN test_function_declared t ON t.fact_id = d.fact_id AND t.test = d.fn AND t.collected
        LEFT JOIN test_function_keywords a ON a.fact_id = d.fact_id AND a.fn = d.fn
          AND a.decorator = d.ordinal AND a.key = 'age_days'
        LEFT JOIN test_function_keywords s ON s.fact_id = d.fact_id AND s.fn = d.fn
          AND s.decorator = d.ordinal AND s.key = 'since' AND s.value_type = 'string'
        LEFT JOIN test_function_keywords o ON o.fact_id = d.fact_id AND o.fn = d.fn
          AND o.decorator = d.ordinal AND o.key = 'owner' AND o.value_type = 'string'
        LEFT JOIN test_function_keywords r ON r.fact_id = d.fact_id AND r.fn = d.fn
          AND r.decorator = d.ordinal AND r.key = 'remediation' AND r.value_type = 'string'
        LEFT JOIN test_function_keywords c ON c.fact_id = d.fact_id AND c.fn = d.fn
          AND c.decorator = d.ordinal AND c.key = 'recurred_after_repair'
        WHERE list_contains(['flaky', 'quarantine', 'quarantined'], regexp_extract(d.mark,
          '[^.]*$'))
    """,
    "calls": """
        SELECT t.fact_id, t.test,
               row_number() OVER (PARTITION BY t.fact_id, t.test ORDER BY c.node_id) - 1 AS
                 ordinal,
               coalesce(CASE WHEN f.type IN ('identifier', 'attribute') THEN
                 coalesce(i.origin, split_part(f.name, '.', 1))
                 || substr(f.name, length(split_part(f.name, '.', 1)) + 1) END, '') AS
                   qualified_name
        FROM test_function_inside c
        JOIN test_function_declared t ON t.fact_id = c.fact_id AND t.test = c.in_test
        JOIN test_function_nodes f ON f.fact_id = c.fact_id AND f.parent_id = c.node_id
          AND f.sibling_index = 0
        LEFT JOIN test_function_imports i ON i.fact_id = c.fact_id
          AND i.local = split_part(f.name, '.', 1)
          AND (i.end_line, i.end_column) <= (c.start_line, c.start_column)
        WHERE c.type = 'call' AND c.owner = c.in_test
    """,
}
