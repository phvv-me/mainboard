"""Rules over enum values (mcmr's PY-ENUM values family)."""

from ...catalog import Example, Rule


class RedundantEnumValue(Rule, id="PY-ENUM0001", applicability="safe"):
    """Explicit enum values that standard `auto()` would preserve."""

    query = """
            SELECT * FROM enum_facts f WHERE EXISTS (
              SELECT 1 FROM enum_enums_members m
              JOIN enum_enums e ON e.fact_id = m.fact_id AND e.record_id = m.parent_id
              WHERE m.fact_id = f.fact_id AND NOT e.overrides_generate_next_value
                AND (m.explicit_value_integer = m.standard_auto_value_integer
                     OR m.explicit_value_string = m.standard_auto_value_string))
        """
    message = "`{fact_id}` satisfies redundant enum value"
    # Every member of the file whose literal equals what the standard `auto()` generates after
    # the member before it, which keeps its own value. The finding is the file, so its members
    # are the edits of one group, and `auto` is imported unless the file already imports it.
    fix = """
        SELECT f.finding_key, m.value_node_span_path AS path,
               m.value_node_span_start_line AS start_line,
               m.value_node_span_start_column AS start_column,
               m.value_node_span_end_line AS end_line,
               m.value_node_span_end_column AS end_column, 'auto()' AS text,
               if(EXISTS (SELECT 1 FROM import_binding_facts i
                          WHERE i.path = f.path AND i.module = 'enum'
                            AND i.imported_name = 'auto' AND i.name = 'auto'),
                  NULL, 'from enum import auto') AS "import"
        FROM finding f
        JOIN enum_enums_members m ON m.fact_id = f.fact_id
        JOIN enum_enums e ON e.fact_id = m.fact_id AND e.record_id = m.parent_id
        WHERE NOT e.overrides_generate_next_value AND m.value_node_id IS NOT NULL
          AND (m.explicit_value_integer = m.standard_auto_value_integer
               OR m.explicit_value_string = m.standard_auto_value_string)
    """
    examples = (
        Example(
            bad=(
                "from enum import StrEnum\n\n\nclass Mode(StrEnum):\n"
                '    FAST = "fast"\n    SLOW = "slow"\n'
            ),
            good=(
                "from enum import StrEnum\nfrom enum import auto\n\n\nclass Mode(StrEnum):\n"
                "    FAST = auto()\n    SLOW = auto()\n"
            ),
        ),
    )


class ParallelEnumMetadata(Rule, id="PY-ENUM0002"):
    """Static string dictionaries that mirror one enum."""

    query = """
        WITH python AS (
          SELECT n.file_id, n.node_id AS id, n.parent_id AS pid, n.sibling_index AS sib, n.type,
                 n.name, n.children_count AS cc, n.descendant_count AS dc
          FROM core_node n JOIN core_file f USING (file_id)
          WHERE f.language = 'python'
            AND n.type IN ('class_definition', 'argument_list', 'identifier', 'attribute',
                           'dictionary', 'pair', 'dictionary_splat', 'string',
                           'concatenated_string', 'import_from_statement', 'aliased_import',
                           'dotted_name', 'module', 'block', 'function_definition',
                           'decorated_definition')
        ), aliases AS (
          SELECT a.file_id, alias.name AS alias, orig.name AS original
          FROM python imp
          JOIN python src ON src.file_id = imp.file_id AND src.pid = imp.id AND src.sib = 1
            AND src.type = 'dotted_name' AND src.name = 'enum'
          JOIN python a ON a.file_id = imp.file_id AND a.pid = imp.id AND a.type = 'aliased_import'
          JOIN python orig ON orig.file_id = a.file_id AND orig.pid = a.id AND orig.sib = 0
          JOIN python alias ON alias.file_id = a.file_id AND alias.pid = a.id AND alias.sib = 1
          JOIN python m ON m.file_id = imp.file_id AND m.id = imp.pid AND m.type = 'module'
          WHERE imp.type = 'import_from_statement'
        ), bases AS (
          SELECT c.file_id, c.id AS class,
                 (SELECT x.name FROM python x
                  WHERE x.file_id = c.file_id AND x.pid = c.id AND x.type = 'identifier'
                    AND x.sib = 1) AS name,
                 CASE WHEN b.type = 'identifier' THEN b.name
                      ELSE (SELECT t.name FROM python t
                            WHERE t.file_id = b.file_id AND t.pid = b.id AND t.type = 'identifier'
                            ORDER BY t.id DESC LIMIT 1) END AS tail
          FROM python c
          JOIN python al ON al.file_id = c.file_id AND al.pid = c.id AND al.type = 'argument_list'
          JOIN python b ON b.file_id = c.file_id AND b.pid = al.id
            AND b.type IN ('identifier', 'attribute')
          WHERE c.type = 'class_definition'
        ), conversions AS (
          SELECT DISTINCT cb.file_id, cb.pid AS class
          FROM python f
          JOIN python n ON n.file_id = f.file_id AND n.pid = f.id AND n.type = 'identifier'
            AND n.name IN ('__str__', '__int__') AND n.sib = 1
          JOIN python up ON up.file_id = f.file_id AND up.id = f.pid
          JOIN python cb ON cb.file_id = f.file_id AND cb.type = 'block'
            AND cb.id = CASE WHEN up.type = 'decorated_definition' THEN up.pid ELSE up.id END
          WHERE f.type = 'function_definition'
        ), enums AS (
          SELECT DISTINCT b.file_id, b.class, b.name
          FROM bases b LEFT JOIN aliases a ON a.file_id = b.file_id AND a.alias = b.tail
          WHERE coalesce(a.original, b.tail) IN ('StrEnum', 'IntEnum', 'IntFlag', 'Flag', 'Enum')
            AND NOT EXISTS (SELECT 1 FROM conversions c
                            WHERE c.file_id = b.file_id AND c.class = b.class)
        ), entries AS (
          SELECT d.file_id, d.id AS dict, e.id AS entry
          FROM python d JOIN python e ON e.file_id = d.file_id AND e.pid = d.id
            AND e.type IN ('pair', 'dictionary_splat')
          WHERE d.type = 'dictionary'
        ), keyed AS (
          SELECT n.file_id, n.dict, count(*) AS items, count(m.owner) AS keys_length,
                 count(*) FILTER (WHERE v.id IS NOT NULL) AS values_length,
                 count(DISTINCT m.owner) AS owners
          FROM entries n
          LEFT JOIN (
            SELECT k.file_id, k.pid AS entry, o.name AS owner
            FROM python k
            JOIN python o ON o.file_id = k.file_id AND o.pid = k.id AND o.sib = 0
              AND o.type = 'identifier'
            JOIN enums e ON e.file_id = o.file_id AND e.name = o.name
            WHERE k.type = 'attribute' AND k.sib = 0 AND k.cc = 2) m
            ON m.file_id = n.file_id AND m.entry = n.entry
          LEFT JOIN python v ON v.file_id = n.file_id AND v.pid = n.entry AND v.sib = 1
            AND v.type IN ('string', 'concatenated_string')
            AND NOT regexp_matches(v.name, '^[bBfFtT]|^[rRuU][bBfFtT]')
          GROUP BY n.file_id, n.dict
          HAVING count(m.owner) > 0
        )
        SELECT f.path, 1 AS start_line, 0 AS start_column,
               'literalgroupfact:' || f.path AS fact_id
        FROM core_file f
        WHERE f.language = 'python' AND EXISTS (
          SELECT 1 FROM keyed k WHERE k.file_id = f.file_id AND k.keys_length = k.items
            AND k.values_length = k.items AND k.owners = 1 AND k.keys_length >= 2)
    """
    message = "`{fact_id}` satisfies parallel enum metadata"


class PreferEnumConversion(Rule, id="PY-ENUM0005", applicability="safe"):
    """`.value` reads of `StrEnum` and `IntEnum` members that a public conversion replaces."""

    query = """
        WITH aliases AS (
          SELECT s.file_id, a.name AS alias, d.name AS original
          FROM core_node s
          JOIN core_node a ON a.file_id = s.file_id AND a.parent_id = s.node_id
            AND a.type = 'aliased_import'
          JOIN core_node d ON d.file_id = s.file_id AND d.parent_id = a.node_id
            AND d.sibling_index = 0
          WHERE s.type = 'import_from_statement' AND s.parent_id = 0 AND s.name = 'enum'
        ), based AS (
          SELECT c.file_id, c.node_id AS class, c.name,
                 coalesce(al.original, regexp_extract(b.name, '[^.]+$')) AS base
          FROM core_node c
          JOIN core_node l ON l.file_id = c.file_id AND l.parent_id = c.node_id
            AND l.type = 'argument_list'
          JOIN core_node b ON b.file_id = c.file_id AND b.parent_id = l.node_id
            AND b.type IN ('identifier', 'attribute')
          LEFT JOIN aliases al ON al.file_id = c.file_id
            AND al.alias = regexp_extract(b.name, '[^.]+$')
          WHERE c.type = 'class_definition'
        ), enums AS (
          SELECT file_id, name, list(base) AS bases
          FROM based b
          GROUP BY file_id, class, name
          HAVING list_has_any(list(base), ['Enum', 'Flag', 'IntFlag', 'IntEnum', 'StrEnum'])
             AND NOT EXISTS (SELECT 1 FROM core_node f
                             WHERE f.file_id = b.file_id AND f.type = 'function_definition'
                               AND f.owner_id = b.class AND f.name IN ('__str__', '__int__'))
        ), heads AS (
          SELECT p.file_id, p.node_id AS node, c.name AS head
          FROM core_node p
          JOIN core_node c ON c.file_id = p.file_id AND c.parent_id = p.node_id
            AND c.sibling_index = 0 AND c.type = 'identifier'
          WHERE p.type IN ('attribute', 'call', 'subscript')
        ), finalized AS (
          WITH RECURSIVE walk(file_id, assignment, value, value_type) AS (
            SELECT a.file_id, a.node_id, v.node_id, v.type
            FROM core_node a
            JOIN core_node v ON v.file_id = a.file_id AND v.parent_id = a.node_id
              AND v.sibling_index = a.children_count - 1
            WHERE a.type = 'assignment'
            UNION ALL
            SELECT f.file_id, f.assignment, v.node_id, v.type
            FROM walk f
            JOIN core_node p ON p.file_id = f.file_id AND p.node_id = f.value
            JOIN core_node v ON v.file_id = p.file_id AND v.parent_id = p.node_id
              AND v.sibling_index = p.children_count - 1
            WHERE f.value_type = 'assignment'
          )
          SELECT file_id, assignment, value, value_type FROM walk WHERE value_type <> 'assignment'
        ), events AS (
          SELECT p.file_id, coalesce(c.function_id, -1) AS func,
                 CASE WHEN c.type IN ('list_splat_pattern', 'dictionary_splat_pattern')
                      THEN ltrim(c.name, '*') ELSE c.name END AS name,
                 calls_annotation_name(t.name) AS held
          FROM core_node p
          JOIN core_node c ON c.file_id = p.file_id AND c.parent_id = p.node_id
            AND c.type IN ('identifier', 'typed_parameter', 'default_parameter',
                           'typed_default_parameter', 'list_splat_pattern',
                           'dictionary_splat_pattern')
          LEFT JOIN core_node t ON t.file_id = c.file_id AND t.parent_id = c.node_id
            AND t.type = 'type'
          WHERE p.type = 'parameters'
          UNION ALL
          SELECT a.file_id, coalesce(a.function_id, -1), l.name,
                 CASE WHEN t.node_id IS NOT NULL THEN calls_annotation_name(t.name)
                      WHEN f.value_type IN ('attribute', 'call', 'subscript') THEN h.head END
          FROM core_node a
          JOIN core_node l ON l.file_id = a.file_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'identifier'
          LEFT JOIN core_node t ON t.file_id = a.file_id AND t.parent_id = a.node_id
            AND t.type = 'type'
          LEFT JOIN finalized f ON f.file_id = a.file_id AND f.assignment = a.node_id
          LEFT JOIN heads h ON h.file_id = a.file_id AND h.node = f.value
          WHERE a.type = 'assignment' AND a.owner_id IS DISTINCT FROM a.class_id
          UNION ALL
          SELECT a.file_id, coalesce(a.function_id, -1), l.name, NULL
          FROM core_node a
          JOIN core_node l ON l.file_id = a.file_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'identifier'
          WHERE a.type = 'augmented_assignment' AND a.owner_id IS DISTINCT FROM a.class_id
          UNION ALL
          SELECT f.file_id, coalesce(f.function_id, -1), l.name,
                 CASE r.type WHEN 'identifier' THEN r.name WHEN 'call' THEN h.head END
          FROM core_node f
          JOIN core_node l ON l.file_id = f.file_id AND l.parent_id = f.node_id
            AND l.sibling_index = 1 AND l.type = 'identifier'
          LEFT JOIN core_node r ON r.file_id = f.file_id AND r.parent_id = f.node_id
            AND r.sibling_index = 3
          LEFT JOIN heads h ON h.file_id = f.file_id AND h.node = r.node_id
          WHERE f.type IN ('for_statement', 'for_in_clause')
            AND f.owner_id IS DISTINCT FROM f.class_id
          UNION ALL
          SELECT t.file_id, coalesce(t.function_id, -1), i.name, NULL
          FROM core_node t
          JOIN core_node i ON i.file_id = t.file_id AND i.parent_id = t.node_id
            AND i.sibling_index = 0 AND i.type = 'identifier'
          JOIN core_node w ON w.file_id = t.file_id AND w.node_id = t.parent_id
          JOIN core_node u ON u.file_id = t.file_id AND u.node_id = w.parent_id
            AND u.type = 'with_item'
          WHERE t.type = 'as_pattern_target' AND t.owner_id IS DISTINCT FROM t.class_id
        ), proven AS (
          SELECT e.file_id, e.func, e.name, e.held
          FROM (SELECT file_id, func, name, any_value(held) AS held
                FROM events GROUP BY file_id, func, name HAVING count(*) = 1) e
          SEMI JOIN enums n ON n.file_id = e.file_id AND n.name = e.held
        ), sites AS (
          SELECT a.file_id, f.path, a.node_id AS node, a.start_line, a.start_column, a.end_line,
                 a.end_column, coalesce(a.function_id, -1) AS func,
                 coalesce(i2.node_id, i1.node_id, r.node_id) AS receiver,
                 coalesce(i2.type, i1.type, r.type) AS receiver_type,
                 coalesce(i2.name, i1.name, r.name) AS receiver_name,
                 coalesce(i2.start_line, i1.start_line, r.start_line) AS rsl,
                 coalesce(i2.start_column, i1.start_column, r.start_column) AS rsc,
                 coalesce(i2.end_line, i1.end_line, r.end_line) AS rel,
                 coalesce(i2.end_column, i1.end_column, r.end_column) AS rec
          FROM core_node a
          JOIN core_file f USING (file_id)
          JOIN core_node r ON r.file_id = a.file_id AND r.parent_id = a.node_id
            AND r.sibling_index = 0
          JOIN core_node m ON m.file_id = a.file_id AND m.parent_id = a.node_id
            AND m.sibling_index = 1 AND m.name = 'value'
          LEFT JOIN core_node i1 ON i1.file_id = a.file_id AND i1.parent_id = r.node_id
            AND r.type = 'parenthesized_expression' AND i1.sibling_index = 1
          LEFT JOIN core_node i2 ON i2.file_id = a.file_id AND i2.parent_id = i1.node_id
            AND i1.type = 'parenthesized_expression' AND i2.sibling_index = 1
          WHERE a.type = 'attribute' AND f.language = 'python'
        )
        SELECT s.path, s.start_line, s.start_column, s.end_line, s.end_column,
               syntax_slice(src.content, s.rsl, s.rsc, s.rel, s.rec) AS receiver_text,
               CASE WHEN list_contains(n.bases, 'StrEnum') THEN 'str' ELSE 'int' END AS conversion
        FROM sites s
        JOIN source_files src ON src.path = s.path
        LEFT JOIN heads h ON h.file_id = s.file_id AND h.node = s.receiver
        LEFT JOIN proven p ON p.file_id = s.file_id AND p.func = s.func
          AND p.name = s.receiver_name AND s.receiver_type = 'identifier'
        JOIN enums n ON n.file_id = s.file_id
          AND n.name = CASE WHEN s.receiver_type = 'identifier' THEN p.held ELSE h.head END
        WHERE list_has_any(n.bases, ['StrEnum', 'IntEnum'])
    """
    message = (
        "`{receiver_text}.value` exposes an enum representation that public `{conversion}` "
        "conversion already provides"
    )
    fix = """
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               conversion || '(' || receiver_text || ')' AS text
        FROM finding
    """
    examples = (
        Example(
            bad=(
                'from enum import StrEnum\n\n\nclass Mode(StrEnum):\n    FAST = "fast"\n\n\n'
                "def name(mode: Mode) -> str:\n    return mode.value\n"
            ),
            good=(
                'from enum import StrEnum\n\n\nclass Mode(StrEnum):\n    FAST = "fast"\n\n\n'
                "def name(mode: Mode) -> str:\n    return str(mode)\n"
            ),
        ),
    )
