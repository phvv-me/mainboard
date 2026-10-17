"""The relations of FunctionFact that hang off a function's source: the text of the spans asked
for, decorators, parameters, control structures, name loads and the call sites of its name."""

TEXTS = r"""
    WITH wanted AS (
      SELECT ast, coalesce(wrapper, node) AS node, start_line AS sl, start_column AS sc,
             end_line AS el, end_column AS ec
      FROM function_defs
      UNION
      SELECT e.ast, e.node, e.sl, e.sc, e.el, e.ec
      FROM function_defs d
      JOIN function_nodes m ON m.ast = d.ast AND m.parent = d.wrapper AND m.type = 'decorator'
      JOIN function_nodes e ON e.ast = m.ast AND e.parent = m.node AND e.sibling = 1
      UNION
      SELECT a.ast, a.node, a.sl, a.sc, a.el, a.ec
      FROM function_bodies b
      JOIN function_nodes p ON p.ast = b.ast AND p.parent = b.params
        AND p.type IN ('typed_parameter', 'typed_default_parameter')
      JOIN function_nodes a ON a.ast = p.ast AND a.parent = p.node AND a.type = 'type'
      UNION
      SELECT c.ast, c.node, c.sl, c.sc, c.el, c.ec FROM function_calls c
      SEMI JOIN function_defs d ON d.ast = c.ast AND d.name = c.callee
      WHERE c.callee_type = 'identifier'
    ), placed AS (
      SELECT w.*, f.path, unnest(range(w.sl, w.el + 1)) AS line
      FROM wanted w JOIN function_files f ON f.ast = w.ast
    )
    SELECT p.ast, p.node,
           string_agg(
             CASE WHEN l.line = p.sl AND l.line = p.el
                  THEN functions_cut(l.text, l.plain, p.sc, p.ec)
                  WHEN l.line = p.sl THEN functions_cut(l.text, l.plain, p.sc, strlen(l.text))
                  WHEN l.line = p.el THEN functions_cut(l.text, l.plain, 0, p.ec)
                  ELSE l.text END, chr(10) ORDER BY p.line) AS text
    FROM placed p JOIN function_lines l ON l.path = p.path AND l.line = p.line
    GROUP BY p.ast, p.node
"""


DECORATORS = r"""
    SELECT d.entity_id AS function_id,
           row_number() OVER (PARTITION BY d.entity_id ORDER BY m.sibling) - 1 AS ordinal,
           t.text AS decorator
    FROM function_defs d
    JOIN function_nodes m ON m.ast = d.ast AND m.parent = d.wrapper AND m.type = 'decorator'
    JOIN function_nodes e ON e.ast = m.ast AND e.parent = m.node AND e.sibling = 1
    JOIN function_texts t ON t.ast = e.ast AND t.node = e.node
"""

PARAMETERS = r"""
    WITH items AS (
      SELECT b.ast, b.def, k.node, k.type, k.name, k.sibling, k.kids,
             f.type IN ('list_splat_pattern', 'dictionary_splat_pattern') AS splat
      FROM function_bodies b
      JOIN function_nodes k ON k.ast = b.ast AND k.parent = b.params
      LEFT JOIN function_nodes f ON f.ast = k.ast AND f.parent = k.node AND f.sibling = 0
        AND k.type = 'typed_parameter'
    ), marks AS (
      SELECT ast, def,
             min(sibling) FILTER (WHERE type IN ('list_splat_pattern', 'keyword_separator', '*')
                                  OR (splat AND type = 'typed_parameter')) AS star,
             min(sibling) FILTER (WHERE type IN ('positional_separator', '/')) AS slash
      FROM items GROUP BY ast, def
    ), declared AS (
      SELECT i.*, m.star, m.slash,
             row_number() OVER (PARTITION BY i.ast, i.def ORDER BY i.sibling) - 1 AS ordinal
      FROM items i JOIN marks m ON m.ast = i.ast AND m.def = i.def
      WHERE i.type IN ('identifier', 'typed_parameter', 'default_parameter',
                       'typed_default_parameter') AND NOT coalesce(i.splat, false)
    ), shaped AS (
      SELECT p.*, coalesce(p.slash IS NOT NULL AND p.sibling < p.slash, false)
               AS is_positional_only,
             coalesce(p.star IS NOT NULL AND p.sibling > p.star, false) AS is_keyword_only,
             regexp_replace(t.text, '\s', '', 'g') AS type_name, v.type AS default_type,
             p.type IN ('default_parameter', 'typed_default_parameter') AS has_default
      FROM declared p
      LEFT JOIN function_nodes a ON a.ast = p.ast AND a.parent = p.node AND a.type = 'type'
      LEFT JOIN function_texts t ON t.ast = a.ast AND t.node = a.node
      LEFT JOIN function_nodes v ON v.ast = p.ast AND v.parent = p.node AND v.sibling = p.kids - 1
        AND p.type IN ('default_parameter', 'typed_default_parameter')
    )
    SELECT d.entity_id AS function_id, p.ordinal, p.name, coalesce(p.type_name, '') AS type_name,
           p.is_positional_only, p.is_keyword_only, r.is_receiver,
           NOT r.is_receiver AND NOT p.has_default AS is_required_by_external_contract,
           coalesce(p.type_name, '') = 'bool' AS has_boolean_annotation,
           coalesce(p.default_type IN ('true', 'false'), false) AS has_boolean_default
    FROM shaped p
    JOIN function_defs d ON d.ast = p.ast AND d.node = p.def,
    LATERAL (SELECT p.ordinal = 0 AND NOT p.is_keyword_only AND p.name IN ('self', 'cls')
               AS is_receiver) r
"""

CONTROLS = r"""
    WITH raw AS (
      SELECT n.ast, n.node, n.size, n.function_scope AS def,
             CASE WHEN n.type = 'if_statement' THEN 'conditional'
                  WHEN n.type IN ('for_statement', 'while_statement') THEN 'loop'
                  WHEN n.type = 'match_statement' THEN 'switch'
                  WHEN n.type = 'try_statement' THEN 'catch'
                  WHEN n.type IN ('break_statement', 'continue_statement') THEN 'jump'
                  WHEN p.type = 'if_statement' THEN 'alternative' END AS kind
      FROM function_nodes n JOIN function_nodes p ON p.ast = n.ast AND p.node = n.parent
      WHERE n.type IN ('if_statement', 'for_statement', 'while_statement', 'match_statement',
                       'try_statement', 'break_statement', 'continue_statement', 'elif_clause',
                       'else_clause')
        AND n.function_scope IS NOT NULL AND coalesce(n.class_scope, -1) < n.function_scope
    ), kept AS (
      SELECT * FROM raw WHERE kind IS NOT NULL
    )
    SELECT d.entity_id AS function_id,
           row_number() OVER (PARTITION BY d.entity_id ORDER BY c.node) - 1 AS ordinal,
           c.kind, count(m.node)::INT - (c.kind = 'alternative')::INT AS nesting_depth
    FROM kept c
    JOIN function_defs d ON d.ast = c.ast AND d.node = c.def
    LEFT JOIN kept m ON m.ast = c.ast AND m.def = c.def
      AND m.kind IN ('conditional', 'loop', 'switch', 'catch')
      AND m.node < c.node AND c.node <= m.node + m.size
    GROUP BY d.entity_id, c.node, c.kind
"""

LOADS = r"""
    WITH names AS (
      SELECT DISTINCT ast, name FROM function_defs
    ), candidates AS (
      SELECT n.ast, n.node, n.parent, n.sibling, n.name, p.type AS parent_type, i.sibling AS later,
             g.type AS grand_type
      FROM function_nodes n
      SEMI JOIN names USING (ast, name)
      JOIN function_nodes p ON p.ast = n.ast AND p.node = n.parent
      LEFT JOIN function_nodes g ON g.ast = p.ast AND g.node = p.parent
      LEFT JOIN function_nodes i ON i.ast = n.ast AND i.parent = n.parent AND i.type = 'in'
        AND p.type IN ('for_statement', 'for_in_clause')
      WHERE n.type = 'identifier'
    )
    SELECT ast, name, count(*) AS loads
    FROM candidates
    WHERE NOT functions_store_position(parent_type, sibling, later, grand_type)
    GROUP BY ast, name
"""

REFERENCES = r"""
    WITH sites AS (
      SELECT d.entity_id AS function_id, d.path, c.ast, c.node, c.sl, c.sc, c.el, c.ec,
             coalesce(c.class_scope, -1) > coalesce(w.def, -1) AS in_class_body,
             o.scope AS owner_scope, k.name AS method_class, kc.name AS body_class,
             o.start_line, o.start_column, o.end_line, o.end_column
      FROM function_defs d
      JOIN function_calls c ON c.ast = d.ast AND c.callee = d.name AND c.callee_type = 'identifier'
      LEFT JOIN function_owners w ON w.ast = c.ast AND w.node = c.node
      LEFT JOIN function_defs o ON o.ast = w.ast AND o.node = w.def AND o.scope <> 'nested'
      LEFT JOIN function_nodes k ON k.ast = o.ast AND k.node = o.owner
      LEFT JOIN function_nodes kc ON kc.ast = c.ast AND kc.node = c.class_scope
    )
    SELECT s.function_id, row_number() OVER (PARTITION BY s.function_id ORDER BY s.node) - 1
             AS ordinal, s.path, s.sl AS start_line, s.sc AS start_column, s.el AS end_line,
           s.ec AS end_column, 'reference' AS kind, t.text,
           CASE WHEN s.in_class_body THEN s.body_class WHEN s.owner_scope = 'method'
                THEN s.method_class ELSE '' END AS owner_class,
           CASE WHEN NOT s.in_class_body THEN s.start_line END AS owner_start_line,
           CASE WHEN NOT s.in_class_body THEN s.start_column END AS owner_start_column,
           CASE WHEN NOT s.in_class_body THEN s.end_line END AS owner_end_line,
           CASE WHEN NOT s.in_class_body THEN s.end_column END AS owner_end_column
    FROM sites s JOIN function_texts t ON t.ast = s.ast AND t.node = s.node
"""
