"""mcmr's ExceptionFact, derived from sitting_duck's Python trees."""

from ....catalog import Family


class ExceptionClasses(Family):
    """mcmr's ExceptionFact: the exception classes each module declares and the modules that import
    them. A module's name is its path below the nearest root: the directory above the outermost
    package (a directory holding an `__init__.py`), none when the outermost package sits at the
    top. A class is an exception when its bases, followed through the project's classes, include a
    name containing `Error` or `Exception`; modules that only gather names (a package
    `__init__` or a module holding nothing but imports, `__all__` and a docstring) are not
    importers of the exception, they re-export it."""

    family = "ExceptionFact"
    relations = {
        "facts": r"""
            WITH RECURSIVE nodes AS (
              SELECT fact_id, node_id, parent_id, type, name, sibling_index, children_count
              FROM ast_nodes
              WHERE fact_id IN (SELECT fact_id FROM ast_files WHERE language = 'python') AND depth
                <= 6
            ), packages AS (
              SELECT DISTINCT regexp_replace(path, '/?[^/]*$', '') AS dir FROM ast_files
              WHERE language = 'python' AND is_initializer_path(path)
            ), roots AS (
              SELECT DISTINCT regexp_replace(p.dir, '/?[^/]*$', '') AS root FROM packages p
              WHERE p.dir <> '' AND regexp_replace(p.dir, '/?[^/]*$', '') <> ''
                AND NOT EXISTS (SELECT 1 FROM packages q WHERE q.dir <> ''
                                AND starts_with(p.dir, q.dir || '/'))
            ), modules AS (
              SELECT f.fact_id, f.path,
                     replace(regexp_replace(regexp_replace(
                       CASE WHEN r.root IS NULL THEN f.path ELSE substr(f.path, length(r.root) +
                         2) END,
                       '\.pyi?$', ''), '/?__init__$', ''), '/', '.') AS module,
                     is_initializer_path(f.path) AS is_package
              FROM ast_files f LEFT JOIN roots r ON starts_with(f.path, r.root || '/')
              WHERE f.language = 'python'
              QUALIFY row_number() OVER (PARTITION BY f.fact_id
                                         ORDER BY length(r.root) DESC NULLS LAST) = 1
            ), classes AS (
              SELECT c.fact_id, m.module, m.path, c.name, c.node_id,
                     row_number() OVER (PARTITION BY c.fact_id ORDER BY c.node_id) - 1 AS ordinal
              FROM nodes c
              JOIN nodes p ON p.fact_id = c.fact_id AND p.node_id = c.parent_id
              JOIN modules m ON m.fact_id = c.fact_id
              WHERE c.type = 'class_definition'
                AND (p.type = 'module' OR (p.type = 'decorated_definition' AND p.parent_id = 0))
            ), bases AS (
              SELECT c.fact_id, c.name, regexp_extract(coalesce(f.name, b.name), '[^.]*$') AS base
              FROM classes c
              JOIN nodes a ON a.fact_id = c.fact_id AND a.parent_id = c.node_id
                AND a.type = 'argument_list'
              JOIN nodes b ON b.fact_id = c.fact_id AND b.parent_id = a.node_id
                AND b.type IN ('identifier', 'attribute', 'call')
              LEFT JOIN nodes f ON f.fact_id = c.fact_id AND f.parent_id = b.node_id
                AND f.sibling_index = 0 AND b.type = 'call'
              WHERE coalesce(f.name, b.name) <> ''
            ), errors AS (
              SELECT name FROM bases WHERE contains(base, 'Error') OR contains(base, 'Exception')
              UNION
              SELECT b.name FROM bases b JOIN errors e ON b.base = e.name
            ), statements AS (
              SELECT fact_id, node_id, type FROM nodes WHERE parent_id = 0 AND type <> 'comment'
            ), seams AS (
              SELECT m.fact_id FROM modules m
              WHERE m.is_package OR NOT EXISTS (
                SELECT 1 FROM statements s
                WHERE s.fact_id = m.fact_id
                  AND s.type NOT IN ('import_statement', 'import_from_statement',
                                     'future_import_statement')
                  AND NOT (s.type = 'expression_statement' AND EXISTS (
                    SELECT 1 FROM nodes a JOIN nodes l ON l.fact_id = a.fact_id
                      AND l.parent_id = a.node_id AND l.sibling_index = 0
                    WHERE a.fact_id = s.fact_id AND a.parent_id = s.node_id AND a.type =
                      'assignment'
                      AND a.children_count = 3 AND l.name = '__all__'))
                  AND NOT (s.type = 'expression_statement' AND s.node_id = (
                    SELECT min(x.node_id) FROM statements x WHERE x.fact_id = s.fact_id)
                    AND EXISTS (SELECT 1 FROM nodes q WHERE q.fact_id = s.fact_id
                                AND q.parent_id = s.node_id AND q.type = 'string')))
            ), imported AS (
              SELECT m.fact_id, m.module AS importer, coalesce(a.name, n.name) AS name,
                     CASE WHEN r.node_id IS NULL THEN d.name
                          ELSE array_to_string(list_concat(
                                 list_slice(string_split(m.module, '.'), 1,
                                            greatest(0, len(string_split(m.module, '.'))
                                                     - CASE WHEN m.is_package THEN 0 ELSE 1 END
                                                     - (length(i.name) - length(ltrim(i.name, '.'))
                                                        - 1))),
                                 CASE WHEN d.name IS NULL THEN [] ELSE [d.name] END), '.') END
                       AS target
              FROM nodes i
              JOIN modules m ON m.fact_id = i.fact_id
              LEFT JOIN nodes r ON r.fact_id = i.fact_id AND r.parent_id = i.node_id
                AND r.type = 'relative_import'
              LEFT JOIN nodes d ON d.fact_id = i.fact_id AND d.type = 'dotted_name'
                AND ((r.node_id IS NULL AND d.parent_id = i.node_id AND d.sibling_index = 1)
                     OR d.parent_id = r.node_id)
              JOIN nodes n ON n.fact_id = i.fact_id AND n.parent_id = i.node_id
                AND n.sibling_index > 1 AND n.type IN ('dotted_name', 'aliased_import')
              LEFT JOIN nodes a ON a.fact_id = i.fact_id AND a.parent_id = n.node_id
                AND n.type = 'aliased_import' AND a.sibling_index = 0
              WHERE i.type = 'import_from_statement' AND i.parent_id = 0
            ), importers AS (
              SELECT DISTINCT target, name, importer FROM imported i
              WHERE target <> importer AND name IN (SELECT name FROM errors)
                AND NOT EXISTS (SELECT 1 FROM seams s WHERE s.fact_id = i.fact_id)
            ), declared AS (
              SELECT c.module, c.path, c.name,
                     row_number() OVER (PARTITION BY c.fact_id ORDER BY c.ordinal) - 1 AS ordinal
              FROM classes c WHERE c.name IN (SELECT name FROM errors)
            )
            SELECT row_number() OVER (ORDER BY min(d.path)) - 1 AS fact_order,
                   'exceptions:' || d.module AS fact_id, any_value(d.path) AS path, 1 AS
                     start_line,
                   0 AS start_column, 1 AS end_line, 0 AS end_column, 'python' AS language,
                   d.module,
                   list(struct_pack(name := d.name,
                                    importers := coalesce(i.importers, []::VARCHAR[]))
                        ORDER BY d.ordinal) AS declared
            FROM declared d
            LEFT JOIN (SELECT target, name, list(importer ORDER BY importer) AS importers
                       FROM importers GROUP BY ALL) i ON i.target = d.module AND i.name = d.name
            GROUP BY d.module
        """,
        "exceptions": """
            SELECT fact_id, fact_id AS parent_id, fact_id || '/exceptions:' || (i - 1) AS
              record_id,
                   i - 1 AS ordinal, d.name, module AS defining_module, d.importers
            FROM (SELECT fact_id, module, generate_subscripts(declared, 1) AS i,
                         unnest(declared) AS d FROM exception_facts)
        """,
        "values": """
            SELECT fact_id, 'exceptions.importing_modules' AS relation, record_id AS parent_id,
                   'value' AS entry_kind, generate_subscripts(importers, 1) - 1 AS ordinal,
                   unnest(importers) AS string_value
            FROM exception_exceptions
        """,
    }
