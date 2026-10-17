"""The closed-world part of ClassFact: which class each written base is, and who inherits whom.

mcmr resolves a base by its bare name: a top-level class of that name in the same module, else
the class a `from` import of that name reaches (through a package's `__all__` re-export), else a
nested class of that name. The relations here do the same over `import_binding_facts` and
`class_defs`, then close the edges transitively. A class is named by its `class_id`.
"""

BINDINGS = """
    SELECT path, name, any_value(module)::VARCHAR AS origin,
           any_value(CASE WHEN starts_with(ltrim(declaration_text), 'from')
                          THEN module || '.' || imported_name ELSE module END)::VARCHAR AS target
    FROM import_binding_facts WHERE language = 'python' GROUP BY path, name
"""

MODULES = r"""
    SELECT path, importer_module AS module, ends_with(path, '__init__.py') AS is_package
    FROM (SELECT DISTINCT path, importer_module FROM import_binding_facts
          WHERE language = 'python')
    UNION ALL
    SELECT f.path, regexp_replace(replace(regexp_replace(f.path, '^src/', ''), '/', '.'),
                                  '(\.__init__)?\.pyi?$', '') AS module,
           ends_with(f.path, '__init__.py') AS is_package
    FROM class_files f
    WHERE f.path NOT IN (SELECT path FROM import_binding_facts WHERE language = 'python')
"""

KEYS = """
    SELECT c.class_id, c.fact_id, c.path, c.class_node, c.name, c.nested, c.ordinal,
           m.module, m.is_package, m.module || '.' || n.qualified AS qualname
    FROM class_defs c
    JOIN class_modules m ON m.path = c.path
    JOIN class_nodes n ON n.fact_id = c.fact_id AND n.node_id = c.class_node
"""

REGISTRY = """
    SELECT module, name, arg_min(class_id, (nested, ordinal)) AS class_id,
           arg_min(nested, (nested, ordinal)) AS nested
    FROM class_keys GROUP BY module, name
"""

FROM_IMPORTS = r"""
    SELECT path, name, imported_name, target, start_line
    FROM (SELECT i.path, i.name, i.imported_name, i.start_line,
                 CASE WHEN i.relative_level = 0 THEN i.module
                      ELSE concat_ws('.', nullif(array_to_string(list_slice(parts, 1,
                             len(parts) - cut), '.'), ''), nullif(i.module, '')) END AS target,
                 i.is_wildcard
          FROM (SELECT *, string_split(importer_module, '.') AS parts,
                       relative_level - ends_with(path, '__init__.py')::INT AS cut
                FROM import_binding_facts
                WHERE language = 'python' AND starts_with(ltrim(declaration_text), 'from')) i)
    WHERE NOT is_wildcard
"""

PACKAGE_EXPORTS = """
    SELECT m.module, m.is_package, s.name AS exported
    FROM class_nodes a
    JOIN class_nodes t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id
                      AND t.sibling_index = 0 AND t.name = '__all__'
    JOIN class_nodes s ON s.fact_id = a.fact_id AND s.type = 'string_content'
                      AND s.node_id BETWEEN a.node_id AND a.node_id + a.descendant_count
    JOIN class_modules m ON m.path = a.path
    WHERE a.type = 'assignment' AND a.depth = 2
"""

REEXPORTS = """
    SELECT m.module, i.imported_name AS name, i.target AS origin_module,
           i.imported_name AS origin_name
    FROM class_from_imports i
    JOIN class_modules m ON m.path = i.path AND m.is_package
    JOIN class_package_exports e ON e.module = m.module AND e.exported = i.imported_name
"""

REACHED = """
    SELECT m.module, m.is_package, i.name AS bound, coalesce(d1.class_id, d2.class_id) AS class_id,
           d1.class_id IS NOT NULL AS direct
    FROM class_from_imports i
    JOIN class_modules m ON m.path = i.path
    LEFT JOIN class_registry d1 ON d1.module = i.target AND d1.name = i.imported_name
    LEFT JOIN class_reexports r ON r.module = i.target AND r.name = i.imported_name
    LEFT JOIN class_registry d2 ON d2.module = r.origin_module AND d2.name = r.origin_name
    WHERE coalesce(d1.class_id, d2.class_id) IS NOT NULL
"""

BASES = """
    SELECT b.class_id, b.fact_id, b.path, b.class_node, b.ordinal, b.text, b.type,
           classes_last_name(b.text) AS base_name
    FROM class_header_items b WHERE NOT b.is_keyword
"""

EDGES = """
    SELECT b.class_id AS child, coalesce(own.class_id, imp.class_id, any_nested.class_id) AS
          parent,
           b.ordinal
    FROM class_bases b
    JOIN class_keys c ON c.class_id = b.class_id
    LEFT JOIN class_registry own ON own.module = c.module AND own.name = b.base_name
                                AND NOT own.nested
    LEFT JOIN class_reached imp ON imp.module = c.module AND imp.bound = b.base_name
    LEFT JOIN class_registry any_nested ON any_nested.module = c.module
                                       AND any_nested.name = b.base_name
    WHERE coalesce(own.class_id, imp.class_id, any_nested.class_id) <> b.class_id
"""

LINEAGE = """
    WITH RECURSIVE ancestors AS (
      SELECT child AS class_id, parent AS ancestor, 1 AS depth FROM class_edges
      UNION
      SELECT a.class_id, e.parent, a.depth + 1
      FROM ancestors a JOIN class_edges e ON e.child = a.ancestor WHERE a.depth < 30
    )
    SELECT class_id, ancestor, min(depth) AS depth FROM ancestors GROUP BY class_id, ancestor
"""

RELATIONS = {
    "bindings": BINDINGS,
    "modules": MODULES,
    "keys": KEYS,
    "registry": REGISTRY,
    "from_imports": FROM_IMPORTS,
    "package_exports": PACKAGE_EXPORTS,
    "reexports": REEXPORTS,
    "reached": REACHED,
    "bases": BASES,
    "edges": EDGES,
    "lineage": LINEAGE,
}
