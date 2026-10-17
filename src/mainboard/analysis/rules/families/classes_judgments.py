"""One row per class with every judgment ClassFact states.

`class_rows` joins what each class says about itself (bases, decorators, body) to what the
repository says about it (who inherits it, who imports and builds it, what it exports).
"""

_DECLARATIVE = (
    "'BaseModel', 'Component', 'DeclarativeBase', 'FlexModel', 'FrozenFlexModel', "
    "'FrozenModel', 'Model', 'RootModel', 'SQLModel'"
)
_APPROVED = (
    "(g.origin = 'patos' OR starts_with(g.origin, 'patos.') OR g.origin = 'common.bases' "
    "OR ends_with(g.origin, '.common.bases'))"
)

REEXPORT_ONLY = """
    SELECT f.path FROM class_files f
    WHERE NOT EXISTS (
      SELECT 1 FROM (
        SELECT s.fact_id, s.type, coalesce(e.type, '') AS head, coalesce(e.name, '') AS head_name,
               row_number() OVER (PARTITION BY s.fact_id ORDER BY s.node_id) AS pos
        FROM class_nodes s
        LEFT JOIN class_nodes e ON e.fact_id = s.fact_id AND e.parent_id = s.node_id
                               AND e.sibling_index = 0 AND s.type = 'expression_statement'
        WHERE s.depth = 1 AND s.type <> 'comment') t
      WHERE t.fact_id = f.fact_id
        AND NOT (t.type IN ('import_statement', 'import_from_statement', 'future_import_statement')
                 OR (t.pos = 1 AND t.head = 'string')
                 OR (t.head = 'assignment' AND t.head_name = '__all__')))
"""

CALLED_NAMES = """
    SELECT DISTINCT m.module, f.name
    FROM class_nodes c
    JOIN class_nodes f ON f.fact_id = c.fact_id AND f.parent_id = c.node_id
                      AND f.sibling_index = 0 AND f.type = 'identifier'
    JOIN class_modules m ON m.path = c.path
    WHERE c.type = 'call'
"""

BASE_INFO = f"""
    SELECT b.class_id,
           bool_or(b.base_name = 'BaseModel'
                   AND (g.origin = 'pydantic' OR starts_with(g.origin, 'pydantic.')))
             AS pydantic_base,
           bool_or(coalesce({_APPROVED}, false)) AS approved_base,
           bool_or(b.base_name IN ({_DECLARATIVE})) AS foundation_base,
           bool_or(b.text IN ('typing.Protocol', 'typing_extensions.Protocol')
                   OR (b.base_name = 'Protocol'
                       AND g.origin IN ('typing', 'typing_extensions'))) AS protocol_base,
           count(*) AS base_count
    FROM class_bases b
    LEFT JOIN class_bindings g ON g.path = b.path AND g.name = b.base_name
    GROUP BY b.class_id
"""

BODY_FLAGS = """
    SELECT c.class_id,
           coalesce(bool_or(a.target_name = 'name' AND a.value_type = 'string'), false)
             AS has_explicit_registry_name,
           coalesce(bool_or(a.target_name = 'model_config'), false)
           OR coalesce(bool_or(n.name = 'Config'), false) AS states_model_configuration
    FROM class_defs c
    LEFT JOIN class_body_assigns a ON a.fact_id = c.fact_id AND a.class_node = c.class_node
    LEFT JOIN class_nested_defs n ON n.fact_id = c.fact_id AND n.class_node = c.class_node
    GROUP BY c.class_id
"""

EXPORTS = """
    SELECT k.class_id,
           EXISTS (SELECT 1 FROM class_package_exports e
                   WHERE e.module = k.module AND e.exported = k.name) AS in_own_all,
           EXISTS (SELECT 1 FROM class_reached r
                   WHERE r.class_id = k.class_id AND r.is_package AND r.direct) AS reexported,
           EXISTS (SELECT 1 FROM class_package_exports e
                   WHERE e.is_package AND e.exported = k.name) AS name_exported
    FROM class_keys k
"""

IMPORTERS = """
    SELECT r.class_id, count(DISTINCT r.module) AS importer_count, min(r.module) AS first_importer
    FROM class_reached r
    JOIN class_keys k ON k.class_id = r.class_id AND NOT k.nested AND r.module <> k.module
    JOIN class_modules m ON m.module = r.module AND NOT m.is_package
    WHERE r.direct AND m.path NOT IN (SELECT path FROM class_reexport_only)
    GROUP BY r.class_id
"""

BUILT = """
    SELECT d.class_id FROM class_called_names c
    JOIN class_registry d ON d.module = c.module AND d.name = c.name
    UNION
    SELECT r.class_id FROM class_reached r
    JOIN class_called_names c ON c.module = r.module AND c.name = r.bound
    WHERE r.direct
"""

CHILDREN = """
    SELECT e.parent AS class_id, count(DISTINCT e.child) AS subclass_count,
           arg_min(k.module, k.ordinal) AS only_child_module,
           arg_min(k.name, k.ordinal) AS only_child_name,
           arg_min(k.path, k.ordinal) AS only_child_path
    FROM class_edges e JOIN class_keys k ON k.class_id = e.child
    GROUP BY e.parent
"""

READS = """
    SELECT DISTINCT p.class_id
    FROM class_children p
    JOIN class_nodes i ON i.path = p.only_child_path AND i.type = 'identifier'
    JOIN class_keys k ON k.class_id = p.class_id AND i.name = k.name
    JOIN class_nodes q ON q.fact_id = i.fact_id AND q.node_id = i.parent_id
    LEFT JOIN class_nodes g ON g.fact_id = q.fact_id AND g.node_id = q.parent_id
    WHERE p.subclass_count = 1
      AND q.type NOT IN ('dotted_name', 'aliased_import', 'import_from_statement',
                         'import_statement', 'future_import_statement', 'function_definition',
                         'class_definition', 'parameters', 'typed_parameter', 'default_parameter',
                         'typed_default_parameter', 'global_statement', 'nonlocal_statement',
                         'list_splat_pattern', 'dictionary_splat_pattern', 'lambda_parameters')
      AND NOT (q.type = 'attribute' AND i.sibling_index > 0)
      AND NOT (q.type = 'keyword_argument' AND i.sibling_index = 0)
      AND NOT (q.type = 'argument_list' AND g.type = 'class_definition')
"""

MEMBER_INFO = """
    SELECT m.class_id, m.name,
           NOT (list_has_any(m.decorator_names, ['abstractmethod', 'abstractproperty', 'overload'])
                OR coalesce(s.is_stub, true)) AS is_concrete,
           d.method_node IS NOT NULL AS delegates
    FROM class_method_defs m
    LEFT JOIN class_stubs s ON s.fact_id = m.fact_id AND s.method_node = m.method_node
    LEFT JOIN class_super_delegates d ON d.fact_id = m.fact_id AND d.method_node = m.method_node
"""

COLLISIONS = """
    SELECT DISTINCT e.child AS class_id
    FROM class_edges e
    JOIN (
      SELECT ch.root, mi.name, arg_min(mi.delegates, (ch.depth, ch.member_class)) AS delegates
      FROM (SELECT class_id AS root, class_id AS member_class, 0 AS depth FROM class_keys
            UNION ALL SELECT class_id, ancestor, depth FROM class_lineage) ch
      JOIN class_member_info mi ON mi.class_id = ch.member_class AND mi.is_concrete
      GROUP BY ch.root, mi.name) s ON s.root = e.parent
    GROUP BY e.child, s.name
    HAVING count(DISTINCT e.parent) >= 2 AND bool_or(NOT s.delegates)
"""

REDUNDANCY = """
    SELECT DISTINCT e1.child AS class_id
    FROM class_edges e1
    JOIN class_edges e2 ON e2.child = e1.child AND e2.parent <> e1.parent
    JOIN class_lineage l ON l.class_id = e1.parent AND l.ancestor = e2.parent
"""

PRELIM = """
    SELECT k.class_id, k.fact_id, k.class_node, k.path, k.name, k.nested, k.ordinal,
           k.module, k.is_package, coalesce(f.field_count, 0) AS field_count,
           coalesce(bi.base_count, 0) AS base_count,
           coalesce(kw.n, 0) AS keyword_count,
           coalesce(dc.decorator_count, 0) AS decorator_count,
           coalesce(dc.is_dataclass, false) AS is_dataclass,
           coalesce(bf.has_explicit_registry_name, false) AS has_explicit_registry_name,
           coalesce(bf.states_model_configuration, false) AS states_model_configuration,
           coalesce(bi.protocol_base, false) AS is_protocol,
           coalesce(pt.is_pass_through, false) AS is_pass_through_layer,
           coalesce(al.alias_count, 0) AS duplicate_component_alias_count,
           coalesce(bh.has_ordinary_behavior, false) AS has_ordinary_behavior,
           coalesce(bi.pydantic_base, false) AS pydantic_base,
           coalesce(bi.approved_base, false) AS inherits_approved_model_foundation,
           coalesce(bi.foundation_base, false) AS foundation_base,
           coalesce(ch.subclass_count, 0) AS subclass_count,
           coalesce(ds.descendant_count, 0) AS descendant_count,
           coalesce(eb.resolved_bases, 0) AS resolved_base_count,
           x.class_id IS NOT NULL AS built,
           coalesce(ex.in_own_all, false) AS in_own_all,
           coalesce(ex.reexported, false) AS reexported,
           coalesce(ex.name_exported, false) AS name_exported,
           coalesce(im.importer_count, 0) AS importer_count,
           im.first_importer, ch.only_child_module,
           rd.class_id IS NOT NULL AS child_reads_name,
           coalesce(inh.has_inherited_fields, false) AS has_inherited_fields,
           co.class_id IS NOT NULL AS has_noncooperative_concrete_collision,
           rr.class_id IS NOT NULL AS has_redundant_direct_base
    FROM class_keys k
    LEFT JOIN class_field_counts f ON f.fact_id = k.fact_id AND f.class_node = k.class_node
    LEFT JOIN class_base_info bi ON bi.class_id = k.class_id
    LEFT JOIN (SELECT class_id, count(*) AS n FROM class_header_items WHERE is_keyword
               GROUP BY class_id) kw ON kw.class_id = k.class_id
    LEFT JOIN (SELECT class_id, count(*) AS decorator_count, bool_or(name = 'dataclass') AS
          is_dataclass
               FROM class_def_decorations GROUP BY class_id) dc ON dc.class_id = k.class_id
    LEFT JOIN class_body_flags bf ON bf.class_id = k.class_id
    LEFT JOIN class_pass_through pt ON pt.fact_id = k.fact_id AND pt.class_node = k.class_node
    LEFT JOIN class_aliases al ON al.fact_id = k.fact_id AND al.class_node = k.class_node
    LEFT JOIN class_behaviour bh ON bh.fact_id = k.fact_id AND bh.class_node = k.class_node
    LEFT JOIN class_exports ex ON ex.class_id = k.class_id
    LEFT JOIN class_importers im ON im.class_id = k.class_id
    LEFT JOIN class_built x ON x.class_id = k.class_id
    LEFT JOIN class_children ch ON ch.class_id = k.class_id
    LEFT JOIN (SELECT ancestor AS class_id, count(DISTINCT class_id) AS descendant_count
               FROM class_lineage GROUP BY ancestor) ds ON ds.class_id = k.class_id
    LEFT JOIN (SELECT child AS class_id, count(*) AS resolved_bases FROM class_edges
               GROUP BY child) eb ON eb.class_id = k.class_id
    LEFT JOIN class_reads rd ON rd.class_id = k.class_id
    LEFT JOIN (SELECT DISTINCT l.class_id, true AS has_inherited_fields FROM class_lineage l
               JOIN class_keys ak ON ak.class_id = l.ancestor
               JOIN class_field_counts af ON af.fact_id = ak.fact_id AND
                     af.class_node = ak.class_node
               WHERE af.field_count > 0) inh ON inh.class_id = k.class_id
    LEFT JOIN class_collisions co ON co.class_id = k.class_id
    LEFT JOIN class_redundancy rr ON rr.class_id = k.class_id
"""

STAGED = """
    SELECT p.*, p.field_count > 0 AS has_instance_fields,
           p.field_count = 0 AND p.states_model_configuration AS is_model_configuration_base,
           p.field_count = 0 AND NOT p.has_inherited_fields
             AND (p.states_model_configuration OR p.subclass_count > 0) AS is_foundation,
           p.class_id IN (SELECT l.class_id FROM class_lineage l
                          JOIN class_prelim a ON a.class_id = l.ancestor
                          WHERE a.foundation_base OR a.is_dataclass) AS ancestor_is_declarative,
           p.in_own_all OR (NOT p.nested AND (p.reexported OR p.name_exported)) AS is_exported,
           NOT p.nested AND (p.in_own_all OR p.reexported OR p.name_exported) AS named_export,
           p.subclass_count = 1 AND p.importer_count = 1
             AND p.first_importer = p.only_child_module AND NOT p.child_reads_name
             AS only_cross_module_reference_is_subclass
    FROM class_prelim p
"""

REMOVABLE_BASES = """
    SELECT class_id FROM class_staged
    WHERE decorator_count = 0 AND keyword_count = 0 AND resolved_base_count = 0
      AND subclass_count = 1 AND descendant_count = 1 AND NOT built AND NOT named_export
      AND only_cross_module_reference_is_subclass
"""

ROWS = """
    SELECT s.*,
           NOT s.is_model_configuration_base AND s.pydantic_base AND NOT s.is_foundation
             AS directly_inherits_pydantic_base_model,
           (NOT s.is_model_configuration_base AND (s.foundation_base OR s.is_dataclass)
            OR s.ancestor_is_declarative) AND NOT s.is_foundation AS is_declarative_model,
           s.resolved_base_count = 1 AND e.parent IN (SELECT class_id FROM class_removable_bases)
             AS base_is_removable_overlap
    FROM class_staged s LEFT JOIN class_edges e ON e.child = s.class_id AND
          s.resolved_base_count = 1
"""

RELATIONS = {
    "reexport_only": REEXPORT_ONLY,
    "called_names": CALLED_NAMES,
    "base_info": BASE_INFO,
    "body_flags": BODY_FLAGS,
    "exports": EXPORTS,
    "importers": IMPORTERS,
    "built": BUILT,
    "children": CHILDREN,
    "reads": READS,
    "member_info": MEMBER_INFO,
    "collisions": COLLISIONS,
    "redundancy": REDUNDANCY,
    "prelim": PRELIM,
    "staged": STAGED,
    "removable_bases": REMOVABLE_BASES,
    "rows": ROWS,
}
