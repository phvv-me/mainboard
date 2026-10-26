"""ClassFact's file-level groups: attribute projections, shared model files, coupled type groups.

Projections are dict, call and tuple/list-of-pairs literals that write one key per attribute read
off the same object (`{"a": p.a, "b": p.b}`, `dict(a=p.a)`, `[("a", p.a)]`). A model file is a file
directly inside a `models` directory that holds a declarative model somewhere. A coupled group is
the top-level classes of one file that share a leading camel-case word and are imported together.
"""

_PROJECTION_ENTRIES = """
    SELECT p.fact_id, p.path, p.parent_id AS container, p.node_id AS entry,
           CASE WHEN k.type = 'string' THEN coalesce(c.name, '') END AS key, v.node_id AS
                 value_node
    FROM class_nodes p
    JOIN class_nodes d ON d.fact_id = p.fact_id AND d.node_id = p.parent_id AND
          d.type = 'dictionary'
    JOIN class_nodes k ON k.fact_id = p.fact_id AND k.parent_id = p.node_id AND k.sibling_index = 0
    JOIN class_nodes v ON v.fact_id = p.fact_id AND v.parent_id = p.node_id
                      AND v.sibling_index = p.children_count - 1
    LEFT JOIN class_nodes c ON c.fact_id = k.fact_id AND c.parent_id = k.node_id
                           AND c.type = 'string_content'
    WHERE p.type = 'pair'
    UNION ALL
    SELECT s.fact_id, s.path, s.parent_id, s.node_id, NULL, v.node_id
    FROM class_nodes s
    JOIN class_nodes d ON d.fact_id = s.fact_id AND d.node_id = s.parent_id AND
          d.type = 'dictionary'
    JOIN class_nodes v ON v.fact_id = s.fact_id AND v.parent_id = s.node_id AND v.sibling_index = 1
    WHERE s.type = 'dictionary_splat'
    UNION ALL
    SELECT kw.fact_id, kw.path, l.parent_id, kw.node_id, k.name, v.node_id
    FROM class_nodes kw
    JOIN class_nodes l ON l.fact_id = kw.fact_id AND l.node_id = kw.parent_id
                      AND l.type = 'argument_list'
    JOIN class_nodes c ON c.fact_id = l.fact_id AND c.node_id = l.parent_id AND c.type = 'call'
    JOIN class_nodes k ON k.fact_id = kw.fact_id AND k.parent_id = kw.node_id AND
          k.sibling_index = 0
    JOIN class_nodes v ON v.fact_id = kw.fact_id AND v.parent_id = kw.node_id
                      AND v.sibling_index = kw.children_count - 1
    WHERE kw.type = 'keyword_argument'
    UNION ALL
    SELECT s.fact_id, s.path, l.parent_id, s.node_id, NULL, v.node_id
    FROM class_nodes s
    JOIN class_nodes l ON l.fact_id = s.fact_id AND l.node_id = s.parent_id
                      AND l.type = 'argument_list'
    JOIN class_nodes c ON c.fact_id = l.fact_id AND c.node_id = l.parent_id AND c.type = 'call'
    JOIN class_nodes v ON v.fact_id = s.fact_id AND v.parent_id = s.node_id AND v.sibling_index = 1
    WHERE s.type = 'dictionary_splat'
    UNION ALL
    SELECT t.fact_id, t.path, t.parent_id, t.node_id, coalesce(c.name, ''), v.node_id
    FROM class_nodes t
    JOIN class_nodes o ON o.fact_id = t.fact_id AND o.node_id = t.parent_id
                      AND o.type IN ('list', 'tuple')
    JOIN class_nodes k ON k.fact_id = t.fact_id AND k.parent_id = t.node_id
                      AND k.sibling_index = 1 AND k.type = 'string'
    JOIN class_nodes v ON v.fact_id = t.fact_id AND v.parent_id = t.node_id
                      AND v.sibling_index = 2 AND v.type NOT IN ('(', ')', ',', 'comment')
    LEFT JOIN class_nodes c ON c.fact_id = k.fact_id AND c.parent_id = k.node_id
                           AND c.type = 'string_content'
    WHERE t.type = 'tuple' AND t.children_count = 4
"""

_PROJECTION_READS = """
    SELECT e.fact_id, e.path, e.container, e.entry, e.key, e.value_node,
           r.name AS root, a.name AS attribute
    FROM class_projection_entries e
    JOIN class_nodes w ON w.fact_id = e.fact_id AND w.node_id = e.value_node
    JOIN class_nodes x ON x.fact_id = e.fact_id AND x.type = 'attribute' AND x.children_count = 2
                      AND (x.node_id = w.node_id
                           OR (w.type = 'parenthesized_expression' AND x.parent_id = w.node_id))
    JOIN class_nodes r ON r.fact_id = x.fact_id AND r.parent_id = x.node_id
                      AND r.sibling_index = 0 AND r.type = 'identifier'
    JOIN class_nodes a ON a.fact_id = x.fact_id AND a.parent_id = x.node_id AND a.sibling_index = 1
"""

_PROJECTION_GROUPS = """
    SELECT g.*, g.fact_id || ':projection:' || (row_number() OVER w - 1) AS projection_id,
           row_number() OVER w - 1 AS ordinal
    FROM (SELECT r.fact_id, r.path, r.container, r.root, list(r.attribute ORDER BY r.value_node)
                   AS attributes
          FROM class_projection_reads r GROUP BY r.fact_id, r.path, r.container, r.root) g
    WINDOW w AS (PARTITION BY g.fact_id ORDER BY g.container, g.root)
"""

_PROJECTION_KEYS = """
    SELECT fact_id, container, list(key ORDER BY entry) AS keys
    FROM class_projection_entries WHERE key IS NOT NULL GROUP BY fact_id, container
"""

_MODEL_PACKAGE_DIRS = """
    SELECT DISTINCT regexp_replace(path, '/[^/]*$', '') AS directory
    FROM class_staged
    WHERE (foundation_base OR is_dataclass) AND NOT nested AND contains(path, '/')
      AND regexp_extract(regexp_replace(path, '/[^/]*$', ''), '[^/]*$') = 'models'
"""

_MODEL_FILE_ROWS = """
    SELECT f.path, f.fact_id, f.end_line, f.end_column,
           count(s.class_id) AS top_level_class_count,
           count(s.class_id) FILTER (
             WHERE NOT s.is_model_configuration_base AND (s.foundation_base OR s.is_dataclass)
           ) AS model_class_count,
           ends_with(f.path, '__init__.py') AS is_package_initializer
    FROM class_files f
    LEFT JOIN class_staged s ON s.path = f.path AND NOT s.nested
    WHERE regexp_replace(f.path, '/[^/]*$', '') IN (SELECT directory FROM class_model_package_dirs)
    GROUP BY f.path, f.fact_id, f.end_line, f.end_column
"""

_CAMEL_WORDS = r"""
    SELECT k.class_id, k.fact_id, k.path, k.module, k.name, k.ordinal,
           regexp_extract_all(k.name, '(?:^[^A-Z]+)|[A-Z][^A-Z]*') AS words
    FROM class_keys k WHERE NOT k.nested
"""

_COUPLED_CANDIDATES = """
    SELECT w.path, w.module, w.words[1] AS first_word, w.class_id, w.name, w.words, w.ordinal,
           c.start_line, c.start_column, c.end_line, c.end_column,
           c.end_line - c.start_line + 1 AS lines
    FROM class_camel_words w JOIN class_defs c ON c.class_id = w.class_id
    WHERE len(w.words) >= 2
"""

_COUPLED_SIZED = """
    SELECT path, module, first_word, count(*) AS type_count,
           list(name ORDER BY ordinal) AS names, arg_min(words, ordinal) AS lead_words,
           arg_min(start_line, ordinal) AS start_line, arg_min(start_column, ordinal) AS
                 start_column,
           arg_min(end_line, ordinal) AS end_line, arg_min(end_column, ordinal) AS end_column,
           max(lines) AS maximum_type_lines
    FROM class_coupled_candidates GROUP BY path, module, first_word HAVING count(*) >= 2
"""

_COUPLED_PREFIXES = """
    SELECT g.*, coalesce(f.failed - 1, 39) AS prefix_words
    FROM class_coupled_sized g
    LEFT JOIN (
      SELECT c.path, c.module, c.first_word, min(p.p) AS failed
      FROM class_coupled_candidates c
      JOIN class_coupled_sized h ON h.path = c.path AND h.module = c.module
                                AND h.first_word = c.first_word
      CROSS JOIN (SELECT unnest(range(2, 40)) AS p) p
      WHERE NOT coalesce(len(c.words) > p.p AND c.words[p.p] = h.lead_words[p.p], false)
      GROUP BY c.path, c.module, c.first_word) f
      ON f.path = g.path AND f.module = g.module AND f.first_word = g.first_word
"""

_COUPLED_GROUPS = """
    SELECT p.path, p.module, p.first_word, p.type_count, p.start_line, p.start_column,
          p.end_line, p.end_column,
           p.maximum_type_lines, p.names,
           array_to_string(list_slice(p.lead_words, 1, p.prefix_words), '') AS prefix,
           row_number() OVER (PARTITION BY p.path ORDER BY p.first_word) - 1 AS ordinal,
           (SELECT count(*) FROM (
              SELECT m.module AS importer
              FROM class_from_imports i JOIN class_modules m ON m.path = i.path AND
                    NOT m.is_package
              WHERE i.target = p.module AND m.module <> p.module
              GROUP BY m.module
              HAVING count(*) FILTER (WHERE list_contains(p.names, i.imported_name)) >= 2))
             AS coimporting_module_count
    FROM class_coupled_prefixes p
"""

RELATIONS = {
    "projection_entries": _PROJECTION_ENTRIES,
    "projection_reads": _PROJECTION_READS,
    "projection_groups": _PROJECTION_GROUPS,
    "projection_keys": _PROJECTION_KEYS,
    "model_package_dirs": _MODEL_PACKAGE_DIRS,
    "model_file_rows": _MODEL_FILE_ROWS,
    "camel_words": _CAMEL_WORDS,
    "coupled_candidates": _COUPLED_CANDIDATES,
    "coupled_sized": _COUPLED_SIZED,
    "coupled_prefixes": _COUPLED_PREFIXES,
    "coupled_groups_rows": _COUPLED_GROUPS,
}
