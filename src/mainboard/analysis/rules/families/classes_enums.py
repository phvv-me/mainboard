"""mcmr's Enum family, derived from ClassFact's relations and sitting_duck's trees.

Per file, the enumerations (a class whose first standard base, by the last word of the name as
written, is `StrEnum`, `IntEnum`, `IntFlag`, `Flag` or `Enum`) with their members: each member's
explicit value and what `auto()` would generate at that position. `auto()` is simulated as the
standard library does: a `StrEnum` takes the member's lowercase name, an `IntEnum` the largest
integer so far plus one, a `Flag` the next power of two, and the simulation stops knowing the
moment the class body holds a statement it cannot follow (a nested class, a loop, an unpacking
assignment, a value that is not an integer or `auto()` in a numeric enum), or binds
`_generate_next_value_` or `_ignore_`.

One more fact, `enums:repository`, answers what no file can: the shared modules that would hold
the enums imported across modules (`scopes`) and the shape of every file under an `enums`
directory (`files`). A scope is the deepest directory holding an enum and all its importers.

Differences from the kernel, kept on purpose: `is_shared_across_unrelated_branches` of a file,
which no rule reads, is not stated.
"""

from ...catalog import Family

_KINDS = (
    "CASE regexp_extract(dotted, '[^.]*$') WHEN 'StrEnum' THEN 'str_enum' "
    "WHEN 'IntEnum' THEN 'int_enum' WHEN 'IntFlag' THEN 'int_flag' WHEN 'Flag' THEN 'flag' "
    "WHEN 'Enum' THEN 'enum' END"
)

FACTS = """
    SELECT fact_order, fact_id, path, start_line, start_column, end_line, end_column, language
    FROM (
      SELECT row_number() OVER (ORDER BY path) - 1 AS fact_order, 'enum:' || path AS fact_id, path,
             start_line, start_column, greatest(end_line, 1) AS end_line, end_column, language
      FROM class_files
      UNION ALL
      SELECT (SELECT count(*) FROM class_files), 'enums:repository', '', 1, 0, 1, 0, 'python')
"""

CLASSES = f"""
    SELECT c.class_id, c.fact_id AS file_id, c.path, c.class_node, c.name, c.nested, k.kind,
           'enum:' || c.path || '/enums:'
             || (row_number() OVER (PARTITION BY c.fact_id ORDER BY c.class_node) - 1) AS
                   record_id,
           row_number() OVER (PARTITION BY c.fact_id ORDER BY c.class_node) - 1 AS ordinal
    FROM class_defs c
    JOIN (SELECT class_id, arg_min(kind, ordinal) AS kind
          FROM (SELECT class_id, ordinal, {_KINDS} AS kind
                FROM (SELECT class_id, ordinal,
                             CASE WHEN type IN ('identifier', 'attribute', 'call')
                                  THEN trim(regexp_extract(text, '^[^\\(]*')) END AS dotted
                      FROM class_bases))
          WHERE kind IS NOT NULL
          GROUP BY class_id) k ON k.class_id = c.class_id
"""

STATEMENTS = r"""
    WITH direct AS (
      SELECT e.class_id, e.file_id, e.path, e.kind, s.node_id AS stmt, s.type AS stmt_type,
             row_number() OVER (PARTITION BY e.class_id ORDER BY s.node_id) AS pos
      FROM enum_classes e
      JOIN class_blocks b ON b.fact_id = e.file_id AND b.class_node = e.class_node
      JOIN class_nodes s ON s.fact_id = e.file_id AND s.parent_id = b.block_node
                        AND s.type <> 'comment'
    ), shaped AS (
      SELECT d.*, a.node_id AS assign, l.type AS left_type, l.name AS left_name,
             v.node_id AS value_node, v.type AS value_type, v.name AS value_text,
             v.start_line AS value_start_line, v.start_column AS value_start_column,
             v.end_line AS value_end_line, v.end_column AS value_end_column,
             a.children_count AS assign_size, f.name AS function_name, sc.name AS string_content,
             v.type = 'string' AND NOT regexp_matches(v.name, '^[a-zA-Z]*[bBfF][a-zA-Z]*[''"]')
               AS is_plain_string,
             EXISTS (SELECT 1 FROM class_nodes t
                     WHERE t.fact_id = d.file_id AND t.parent_id = a.node_id
                       AND t.type = 'type') AS annotated,
             EXISTS (SELECT 1 FROM class_nodes t
                     WHERE t.fact_id = d.file_id AND t.parent_id = a.node_id
                       AND t.type = '=') AS has_value
      FROM direct d
      LEFT JOIN class_nodes a ON a.fact_id = d.file_id AND a.parent_id = d.stmt
                             AND a.type IN ('assignment', 'augmented_assignment')
                             AND d.stmt_type = 'expression_statement'
      LEFT JOIN class_nodes l ON l.fact_id = d.file_id AND l.parent_id = a.node_id
                             AND l.sibling_index = 0 AND a.type = 'assignment'
      LEFT JOIN class_nodes v ON v.fact_id = d.file_id AND v.parent_id = a.node_id
                             AND v.sibling_index = a.children_count - 1 AND a.type = 'assignment'
      LEFT JOIN class_nodes sc ON sc.fact_id = d.file_id AND sc.parent_id = v.node_id
                              AND sc.type = 'string_content' AND v.type = 'string'
      LEFT JOIN class_nodes f ON f.fact_id = d.file_id
                             AND ((f.parent_id = d.stmt AND f.type = 'function_definition'
                                   AND d.stmt_type = 'decorated_definition')
                                  OR (f.node_id = d.stmt AND f.type = 'function_definition'))
    ), classified AS (
      SELECT *,
             CASE WHEN stmt_type = 'function_definition' OR (stmt_type = 'decorated_definition'
                                                             AND function_name IS NOT NULL)
                       OR stmt_type = 'pass_statement' THEN 'neutral'
                  WHEN stmt_type = 'expression_statement' AND assign IS NULL THEN 'neutral'
                  WHEN assign IS NULL THEN 'break'
                  WHEN left_type IS NULL THEN 'break'
                  WHEN NOT has_value THEN CASE WHEN left_type = 'identifier' THEN 'neutral'
                                               ELSE 'break' END
                  WHEN left_type <> 'identifier' OR value_type = 'assignment' THEN 'break'
                  WHEN starts_with(left_name, '__')
                       OR (starts_with(left_name, '_') AND ends_with(left_name, '_'))
                       THEN 'neutral'
                  ELSE 'member' END AS action,
             coalesce(CASE WHEN assign IS NOT NULL AND left_type = 'identifier' THEN left_name END,
                      function_name) AS bound_name
      FROM shaped
    )
    SELECT c.*,
           CASE WHEN c.value_type = 'integer' THEN try_cast(replace(c.value_text, '_', '') AS
                 BIGINT)
                END AS literal_int,
           c.value_type = 'call' AND c.value_text IN ('auto', 'enum.auto')
             AND EXISTS (SELECT 1 FROM class_nodes q
                         WHERE q.fact_id = c.file_id AND q.parent_id = c.value_node
                           AND q.type = 'argument_list' AND q.children_count = 2) AS is_auto
    FROM classified c
"""

STEPS = """
    WITH RECURSIVE walk AS (
      SELECT e.class_id, 0 AS pos, 0 AS stmt_pos, NULL::BIGINT AS automatic_int,
             NULL::VARCHAR AS automatic_str,
             NOT EXISTS (SELECT 1 FROM enum_statements s
                         WHERE s.class_id = e.class_id
                           AND s.bound_name IN ('_generate_next_value_', '_ignore_')) AS known,
             NULL::BIGINT AS maxv
      FROM enum_classes e
      UNION ALL
      SELECT s.class_id, s.pos, s.pos, step.automatic_int, step.automatic_str,
             CASE s.action WHEN 'break' THEN false WHEN 'neutral' THEN w.known
                  ELSE CASE WHEN s.kind = 'str_enum' THEN w.known
                            ELSE w.known AND step.actual IS NOT NULL END END,
             CASE WHEN s.action = 'member' AND s.kind <> 'str_enum' AND w.known
                       AND step.actual IS NOT NULL
                  THEN greatest(coalesce(w.maxv, step.actual), step.actual) ELSE w.maxv END
      FROM walk w
      JOIN enum_statements s ON s.class_id = w.class_id AND s.pos = w.pos + 1,
      LATERAL (
        SELECT CASE WHEN s.action <> 'member' OR NOT w.known OR s.kind = 'str_enum' THEN NULL
                    WHEN w.maxv IS NULL THEN 1
                    WHEN s.kind IN ('flag', 'int_flag')
                    THEN CASE WHEN w.maxv < 0 THEN NULL
                              WHEN w.maxv = 0 THEN 1
                              ELSE (1::BIGINT << (floor(log2(w.maxv))::INT + 1)) END
                    ELSE w.maxv + 1 END AS automatic_int,
               CASE WHEN s.action = 'member' AND w.known AND s.kind = 'str_enum'
                    THEN lower(s.left_name) END AS automatic_str,
               CASE WHEN s.literal_int IS NOT NULL THEN s.literal_int
                    WHEN s.is_auto THEN
                      CASE WHEN s.action <> 'member' OR NOT w.known OR
                            s.kind = 'str_enum' THEN NULL
                           WHEN w.maxv IS NULL THEN 1
                           WHEN s.kind IN ('flag', 'int_flag')
                           THEN CASE WHEN w.maxv < 0 THEN NULL WHEN w.maxv = 0 THEN 1
                                     ELSE (1::BIGINT << (floor(log2(w.maxv))::INT + 1)) END
                           ELSE w.maxv + 1 END END AS actual) step
    )
    SELECT * FROM walk WHERE pos > 0
"""

ENUMS = """
    SELECT 'enum:' || e.path AS fact_id, 'enum:' || e.path AS parent_id, e.record_id,
           e.ordinal, e.name, e.kind,
           EXISTS (SELECT 1 FROM enum_statements s
                   WHERE s.class_id = e.class_id AND s.bound_name = '_generate_next_value_')
             AS overrides_generate_next_value, e.class_id
    FROM enum_classes e
"""

ENUMS_MEMBERS = """
    SELECT 'enum:' || e.path AS fact_id, e.record_id AS parent_id,
           e.record_id || '/enums.members:' || (row_number() OVER w - 1) AS record_id,
           row_number() OVER w - 1 AS ordinal, s.left_name AS name,
           CASE WHEN s.value_type = 'integer' THEN s.literal_int END AS explicit_value_integer,
           CASE WHEN s.value_type = 'integer' THEN
                  CASE WHEN s.literal_int IS NULL THEN s.value_text END
                WHEN s.is_plain_string THEN coalesce(s.string_content, '')
                WHEN s.value_type = 'float' THEN 'Float(' || s.value_text || ')'
                ELSE '' END AS explicit_value_string,
           t.automatic_int AS standard_auto_value_integer,
           t.automatic_str AS standard_auto_value_string,
           e.path || ':' || s.value_node || ':expression' AS value_node_id,
           'expression' AS value_node_kind, e.path AS value_node_span_path,
           s.value_start_line AS value_node_span_start_line,
           s.value_start_column AS value_node_span_start_column,
           s.value_end_line AS value_node_span_end_line,
           s.value_end_column AS value_node_span_end_column
    FROM enum_statements s
    JOIN enum_classes e ON e.class_id = s.class_id
    JOIN enum_steps t ON t.class_id = s.class_id AND t.pos = s.pos
    WHERE s.action = 'member'
    WINDOW w AS (PARTITION BY s.class_id ORDER BY s.pos)
"""

DEFINITIONS = """
    SELECT k.class_id, k.module, k.name, k.path, classes_is_test(k.path) AS is_test
    FROM enum_classes e
    JOIN class_keys k ON k.class_id = e.class_id
    JOIN class_defs d ON d.class_id = e.class_id
    JOIN class_nodes s ON s.fact_id = d.fact_id AND s.node_id = d.statement_node AND s.depth = 1
"""

IMPORTS = """
    SELECT d.class_id, d.module, d.path, d.is_test, m.path AS importer_path
    FROM enum_definitions d
    JOIN class_from_imports i ON i.target = d.module AND i.imported_name = d.name
    JOIN class_modules m ON m.path = i.path
    WHERE m.module <> d.module AND classes_is_test(m.path) = d.is_test
      AND EXISTS (SELECT 1 FROM class_nodes n
                  WHERE n.path = i.path AND n.type = 'import_from_statement' AND n.depth = 1
                    AND n.start_line = i.start_line)
    GROUP BY d.class_id, d.module, d.path, d.is_test, m.path
"""

REUSE = r"""
    WITH paths AS (
      SELECT class_id, is_test, path FROM enum_imports
      UNION ALL
      SELECT class_id, is_test, importer_path FROM enum_imports
    ), parts AS (
      SELECT class_id, is_test, generate_subscripts(dir, 1) AS pos, unnest(dir) AS comp
      FROM (SELECT class_id, is_test,
                   string_split(CASE WHEN contains(path, '/')
                                     THEN regexp_replace(path, '/[^/]*$', '') ELSE '' END, '/')
                     AS dir
            FROM paths)
    ), agreement AS (
      SELECT class_id, is_test, pos, count(*) AS n, count(DISTINCT comp) AS distinct_comps,
             any_value(comp) AS comp
      FROM parts GROUP BY class_id, is_test, pos
    ), totals AS (
      SELECT class_id, count(*) AS n FROM paths GROUP BY class_id
    ), shared AS (
      SELECT a.class_id, a.is_test,
             coalesce(min(a.pos) FILTER (WHERE a.n < t.n OR a.distinct_comps > 1) - 1,
                      max(a.pos)) AS length
      FROM agreement a JOIN totals t USING (class_id)
      GROUP BY a.class_id, a.is_test
    )
    SELECT s.class_id, s.is_test,
           coalesce(array_to_string(list(a.comp ORDER BY a.pos) FILTER (WHERE a.pos <= s.length),
                                    '/'), '') AS directory,
           (SELECT count(*) FROM enum_imports i WHERE i.class_id = s.class_id) AS importers
    FROM shared s JOIN agreement a ON a.class_id = s.class_id
    GROUP BY s.class_id, s.is_test, s.length
"""

SCOPE_KEYS = """
    SELECT directory, is_test, count(*) AS reused_enum_count, sum(importers) AS
          cross_module_import_count
    FROM enum_reuse GROUP BY directory, is_test
"""

ASSIGNED = """
    SELECT d.class_id, coalesce(r.directory, h.directory) AS directory, d.is_test
    FROM enum_definitions d
    LEFT JOIN enum_reuse r ON r.class_id = d.class_id
    LEFT JOIN (
      SELECT d2.class_id, arg_max(k.directory, length(k.directory)) AS directory
      FROM enum_definitions d2
      JOIN enum_scope_keys k ON k.is_test = d2.is_test
                            AND (k.directory = ''
                                 OR CASE WHEN contains(d2.path, '/')
                                         THEN regexp_replace(d2.path, '/[^/]*$', '') ELSE '' END
                                    = k.directory
                                 OR starts_with(CASE WHEN contains(d2.path, '/')
                                                     THEN regexp_replace(d2.path, '/[^/]*$', '')
                                                     ELSE '' END, k.directory || '/'))
      GROUP BY d2.class_id) h ON h.class_id = d.class_id
    WHERE coalesce(r.directory, h.directory) IS NOT NULL
"""

SCOPES = """
    SELECT 'enums:repository' AS fact_id, 'enums:repository' AS parent_id,
           'enums:repository/scopes:'
             || (row_number() OVER (ORDER BY k.directory, k.is_test) - 1) AS record_id,
           row_number() OVER (ORDER BY k.directory, k.is_test) - 1 AS ordinal,
           CASE WHEN k.directory = '' THEN 'enums.py' ELSE k.directory || '/enums.py' END
             AS destination,
           (SELECT count(*) FROM enum_assigned a
            WHERE a.directory = k.directory AND a.is_test = k.is_test) AS enum_count,
           k.reused_enum_count, k.cross_module_import_count
    FROM enum_scope_keys k
"""

FILES = """
    SELECT 'enums:repository' AS fact_id, 'enums:repository' AS parent_id,
           'enums:repository/files:'
             || (row_number() OVER (ORDER BY f.path) - 1) AS record_id,
           row_number() OVER (ORDER BY f.path) - 1 AS ordinal, f.path,
           (SELECT count(*) FROM class_defs c
            JOIN class_nodes s ON s.fact_id = c.fact_id AND s.node_id = c.statement_node
                              AND s.depth = 1
            WHERE c.fact_id = f.fact_id) AS top_level_class_count,
           (SELECT count(*) FROM enum_classes e
            JOIN class_defs c ON c.class_id = e.class_id
            JOIN class_nodes s ON s.fact_id = c.fact_id AND s.node_id = c.statement_node
                              AND s.depth = 1
            WHERE e.file_id = f.fact_id) AS enum_class_count,
           ends_with(f.path, '__init__.py') AS is_package_initializer
    FROM class_files f
    WHERE list_contains(string_split(CASE WHEN contains(f.path, '/')
                                          THEN regexp_replace(f.path, '/[^/]*$', '') ELSE '' END,
                                     '/'), 'enums')
"""


class Enums(Family):
    family = "Enum"
    verified = True
    relations = {
        "facts": FACTS,
        "classes": CLASSES,
        "statements": STATEMENTS,
        "steps": STEPS,
        "enums": ENUMS,
        "enums_members": ENUMS_MEMBERS,
        "definitions": DEFINITIONS,
        "imports": IMPORTS,
        "reuse": REUSE,
        "scope_keys": SCOPE_KEYS,
        "assigned": ASSIGNED,
        "scopes": SCOPES,
        "files": FILES,
    }
