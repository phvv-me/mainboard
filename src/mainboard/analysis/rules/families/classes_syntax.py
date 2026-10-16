"""What the syntax trees say about each class: its header, decorators, members and fields.

ClassFact's relations, in dependency order, each reading the earlier ones by their view names
(`class_nodes` for the relation `nodes`). Every class and method is named by a stable id
(`classes:<path>:class:<n>`), and a node by its `node_id` in its file; `fact_id` is the file's.
mcmr reads only the statements directly in a class body (a method under an `if` is not a method)
and a method's statements at any depth, which the relations below follow.
"""

from ...catalog import Function


class ClassesText(Function):
    """The source text of a span, cut from the file's `lines` by UTF-8 byte columns."""

    lines: list[str]
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    expression = """array_to_string(list_concat(
        [decode(array_slice(encode(lines[start_line]), start_column + 1,
                            CASE WHEN start_line = end_line THEN end_column
                                 ELSE strlen(lines[start_line]) END))],
        lines[start_line + 1:end_line - 1],
        CASE WHEN end_line > start_line
             THEN [decode(array_slice(encode(lines[end_line]), 1, end_column))] ELSE [] END),
      chr(10))"""


class ClassesLastName(Function):
    """The last dotted segment of a decorator or base written as text, without call or subscript:
    `abc.abstractmethod` and `register(x)` read `abstractmethod` and `register`."""

    text: str

    expression = (
        r"""regexp_extract(rtrim(trim(regexp_extract(text, '^[^\[\(]*')), '.'), '[^.]*$')"""
    )


FILES = r"""
    SELECT 'classes:' || f.path AS fact_id, f.path, 'python' AS language,
           1 AS start_line, 0 AS start_column, m.end_line, m.end_column,
           string_split(s.content, chr(10)) AS lines
    FROM ast_files f
    JOIN ast_nodes m ON m.fact_id = f.fact_id AND m.node_id = 0
    JOIN source_files s ON s.path = f.path
    WHERE f.language = 'python'
"""

NODES = """
    SELECT 'classes:' || n.path AS fact_id, n.path, n.node_id, n.type, n.name, n.parent_id,
           n.sibling_index, n.children_count::BIGINT AS children_count,
           n.descendant_count::BIGINT AS descendant_count, n.depth::BIGINT AS depth,
           n.start_line::BIGINT AS start_line, n.start_column::BIGINT AS start_column,
           n.end_line::BIGINT AS end_line, n.end_column::BIGINT AS end_column,
           len(n.scope.stack) AS scope_depth, n.scope.current AS scope_node,
           n.scope."class" AS scope_class, list_contains(n.modifiers, 'async') AS is_async,
           CASE WHEN n.type IN ('class_definition', 'function_definition')
                THEN array_to_string([q.name for q in n.qualified_name], '.') END AS qualified
    FROM ast_nodes n SEMI JOIN ast_files f ON f.fact_id = n.fact_id AND f.language = 'python'
"""

BLOCK_ENDS = """
    SELECT b.fact_id, b.parent_id AS owner_node, arg_max(s.end_line, s.node_id) AS end_line,
           arg_max(s.end_column, s.node_id) AS end_column
    FROM class_nodes b
    JOIN class_nodes s ON s.fact_id = b.fact_id AND s.parent_id = b.node_id AND s.type <> 'comment'
    WHERE b.type = 'block'
    GROUP BY b.fact_id, b.parent_id
"""

DEFS = """
    SELECT c.fact_id, c.path, c.node_id AS class_node, c.name,
           coalesce(e.end_line, c.end_line) AS end_line,
           coalesce(e.end_column, c.end_column) AS end_column,
           coalesce(d.node_id, c.node_id) AS statement_node,
           coalesce(d.start_line, c.start_line) AS start_line,
           coalesce(d.start_column, c.start_column) AS start_column,
           c.scope_depth > 2 AS nested,
           row_number() OVER (PARTITION BY c.fact_id ORDER BY c.node_id) - 1 AS ordinal,
           c.fact_id || ':class:' || (row_number() OVER (PARTITION BY c.fact_id ORDER BY c.node_id)
                                      - 1) AS class_id
    FROM class_nodes c
    LEFT JOIN class_nodes d ON d.fact_id = c.fact_id AND d.node_id = c.parent_id
                           AND d.type = 'decorated_definition'
    LEFT JOIN class_block_ends e ON e.fact_id = c.fact_id AND e.owner_node = c.node_id
    WHERE c.type = 'class_definition'
"""

BLOCKS = """
    SELECT c.fact_id, c.class_node, b.node_id AS block_node
    FROM class_defs c
    JOIN class_nodes b ON b.fact_id = c.fact_id AND b.parent_id = c.class_node AND b.type = 'block'
"""

MEMBER_DEFS = """
    SELECT f.fact_id, f.path, f.node_id AS method_node, f.name, b.class_node, c.class_id,
           coalesce(e.end_line, f.end_line) AS end_line,
           coalesce(e.end_column, f.end_column) AS end_column, f.descendant_count,
           coalesce(d.node_id, f.node_id) AS statement_node,
           coalesce(d.start_line, f.start_line) AS start_line,
           coalesce(d.start_column, f.start_column) AS start_column,
           row_number() OVER (PARTITION BY f.fact_id, b.class_node ORDER BY f.node_id) - 1
             AS ordinal
    FROM class_nodes f
    LEFT JOIN class_nodes d ON d.fact_id = f.fact_id AND d.node_id = f.parent_id
                           AND d.type = 'decorated_definition'
    JOIN class_blocks b ON b.fact_id = f.fact_id
                       AND b.block_node = coalesce(d.parent_id, f.parent_id)
    JOIN class_defs c ON c.fact_id = b.fact_id AND c.class_node = b.class_node
    LEFT JOIN class_block_ends e ON e.fact_id = f.fact_id AND e.owner_node = f.node_id
    WHERE f.type = 'function_definition'
"""

HEADER_ITEMS = """
    SELECT a.fact_id, a.path, c.class_id, c.class_node, a.node_id, a.type, a.name,
           a.start_line, a.start_column, a.end_line, a.end_column,
           a.type IN ('keyword_argument', 'dictionary_splat') AS is_keyword,
           row_number() OVER (
             PARTITION BY a.fact_id, c.class_node, a.type IN ('keyword_argument',
                   'dictionary_splat')
             ORDER BY a.node_id) - 1 AS ordinal,
           classes_text(f.lines, a.start_line, a.start_column, a.end_line, a.end_column) AS text
    FROM class_defs c
    JOIN class_nodes l ON l.fact_id = c.fact_id AND l.parent_id = c.class_node
                      AND l.type = 'argument_list'
    JOIN class_nodes a ON a.fact_id = c.fact_id AND a.parent_id = l.node_id
                      AND a.type NOT IN ('(', ')', ',', 'comment')
    JOIN class_files f ON f.fact_id = c.fact_id
"""

DECORATIONS = """
    SELECT *, classes_last_name(split_part(text, '(', 1)) AS name,
           trim(split_part(text, '(', 1)) AS dotted
    FROM (SELECT e.fact_id, e.path, d.parent_id AS decorated_node,
                 row_number() OVER (PARTITION BY d.fact_id, d.parent_id ORDER BY d.node_id) - 1
                   AS ordinal,
                 classes_text(f.lines, e.start_line, e.start_column, e.end_line, e.end_column)
                   AS text
          FROM class_nodes d
          JOIN class_nodes e ON e.fact_id = d.fact_id AND e.parent_id = d.node_id
                            AND e.sibling_index = 1
          JOIN class_files f ON f.fact_id = d.fact_id
          WHERE d.type = 'decorator')
"""

DEF_DECORATIONS = """
    SELECT c.class_id, d.fact_id, c.class_node, d.ordinal, d.text, d.name, d.dotted
    FROM class_decorations d
    JOIN class_defs c ON c.fact_id = d.fact_id AND c.statement_node = d.decorated_node
                     AND c.statement_node <> c.class_node
"""

BODY_ASSIGNS = """
    SELECT b.fact_id, b.class_node, a.node_id AS assign_node, a.name AS target_name,
           t.name AS annotation, v.type AS value_type
    FROM class_blocks b
    JOIN class_nodes s ON s.fact_id = b.fact_id AND s.parent_id = b.block_node
                      AND s.type = 'expression_statement'
    JOIN class_nodes a ON a.fact_id = b.fact_id AND a.parent_id = s.node_id AND
          a.type = 'assignment'
    LEFT JOIN class_nodes t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id AND
          t.type = 'type'
    LEFT JOIN class_nodes e ON e.fact_id = a.fact_id AND e.parent_id = a.node_id AND e.type = '='
    LEFT JOIN class_nodes v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
                           AND v.sibling_index = a.children_count - 1 AND e.node_id IS NOT NULL
"""

NESTED_DEFS = """
    SELECT b.fact_id, b.class_node, c.name
    FROM class_blocks b
    JOIN class_nodes d ON d.fact_id = b.fact_id AND d.type = 'class_definition'
    JOIN class_defs c ON c.fact_id = d.fact_id AND c.class_node = d.node_id
    LEFT JOIN class_nodes p ON p.fact_id = d.fact_id AND p.node_id = d.parent_id
                           AND p.type = 'decorated_definition'
    WHERE coalesce(p.parent_id, d.parent_id) = b.block_node
"""

REGIONS = """
    SELECT n.path, n.start_line AS line
    FROM class_nodes n JOIN class_files f ON f.fact_id = n.fact_id
    WHERE n.type = 'comment'
      AND starts_with(lower(ltrim(ltrim(
            classes_text(f.lines, n.start_line, n.start_column, n.end_line, n.end_column), '#'),
            ' ' || chr(9))), 'region')
"""

METHOD_DEFS = """
    SELECT m.*, m.class_id || ':method:' || m.ordinal AS method_id,
           coalesce(d.names, []) AS decorator_names,
           CASE WHEN m.name IN ('__init__', '__new__') THEN 'constructor'
                WHEN list_has_any(d.names, ['property', 'cached_property']) THEN 'property'
                WHEN list_contains(d.names, 'staticmethod') THEN 'static_method'
                WHEN list_contains(d.names, 'classmethod') THEN 'class_method'
                ELSE 'method' END AS kind,
           CASE WHEN starts_with(m.name, '__') AND ends_with(m.name, '__') THEN 'public'
                WHEN starts_with(m.name, '__') THEN 'private'
                WHEN starts_with(m.name, '_') THEN 'protected' ELSE 'public' END AS visibility,
           starts_with(m.name, '__') AND ends_with(m.name, '__') AS is_protocol_name
    FROM class_member_defs m
    LEFT JOIN (SELECT fact_id, decorated_node, list(name ORDER BY ordinal) AS names
               FROM class_decorations GROUP BY fact_id, decorated_node) d
      ON d.fact_id = m.fact_id AND d.decorated_node = m.statement_node
"""

SELF_FIELDS = r"""
    SELECT DISTINCT m.fact_id, m.class_node, substr(f0.full_name, 6) AS field
    FROM class_member_defs m
    JOIN (SELECT t.fact_id, t.node_id AS target_node, arg_min(x.name, x.node_id) AS full_name
          FROM class_nodes a
          JOIN class_nodes t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id
                            AND t.sibling_index = 0
          JOIN class_nodes x ON x.fact_id = t.fact_id AND x.type = 'attribute'
                            AND x.node_id BETWEEN t.node_id AND t.node_id + t.descendant_count
                            AND regexp_matches(x.name, '^self\.\w+$')
          WHERE a.type IN ('assignment', 'augmented_assignment')
          GROUP BY t.fact_id, t.node_id) f0
      ON f0.fact_id = m.fact_id
     AND f0.target_node BETWEEN m.method_node AND m.method_node + m.descendant_count
"""

FIELD_COUNTS = r"""
    SELECT fact_id, class_node, count(DISTINCT field) AS field_count
    FROM (SELECT fact_id, class_node, target_name AS field FROM class_body_assigns
          WHERE annotation IS NOT NULL AND regexp_matches(target_name, '\p{Ll}')
            AND NOT regexp_matches(annotation, '^(?:[\w.]*\.)?ClassVar(\[|$)')
          UNION ALL
          SELECT fact_id, class_node, field FROM class_self_fields)
    GROUP BY fact_id, class_node
"""

MEMBER_PARAMS = """
    SELECT * FROM (
      SELECT m.fact_id, m.class_node, m.method_node, p.node_id AS param_node,
             p.type AS param_type, coalesce(e.name, p.name) AS name, p.sibling_index AS idx,
             CASE WHEN coalesce(e.type, p.type) = 'list_splat_pattern' THEN 'vararg'
                  WHEN coalesce(e.type, p.type) = 'dictionary_splat_pattern' THEN 'kwarg'
                  WHEN coalesce(bool_or(coalesce(e.type, p.type) IN ('list_splat_pattern', '*'))
                         OVER (PARTITION BY m.fact_id, m.method_node ORDER BY p.sibling_index
                               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), false)
                       THEN 'kwonly'
                  ELSE 'positional' END AS kind
      FROM class_member_defs m
      JOIN class_nodes l ON l.fact_id = m.fact_id AND l.parent_id = m.method_node
                        AND l.type = 'parameters'
      JOIN class_nodes p ON p.fact_id = m.fact_id AND p.parent_id = l.node_id
                        AND p.type NOT IN ('(', ')', ',', 'comment')
      LEFT JOIN class_nodes e ON e.fact_id = p.fact_id AND e.parent_id = p.node_id
                             AND p.type = 'typed_parameter' AND e.sibling_index = 0
                             AND e.type IN ('list_splat_pattern', 'dictionary_splat_pattern')
    ) WHERE param_type NOT IN ('/', '*')
"""

RELATIONS = {
    "files": FILES,
    "nodes": NODES,
    "block_ends": BLOCK_ENDS,
    "defs": DEFS,
    "blocks": BLOCKS,
    "member_defs": MEMBER_DEFS,
    "header_items": HEADER_ITEMS,
    "decorations": DECORATIONS,
    "def_decorations": DEF_DECORATIONS,
    "body_assigns": BODY_ASSIGNS,
    "nested_defs": NESTED_DEFS,
    "regions": REGIONS,
    "method_defs": METHOD_DEFS,
    "self_fields": SELF_FIELDS,
    "field_counts": FIELD_COUNTS,
    "member_params": MEMBER_PARAMS,
}
