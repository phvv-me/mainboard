"""The classes of a project as the PY-CUDA facts use them: what each inherits, and which fields
and methods an instance answers to, so `batch.tokens` and `stage.queued(...)` find the
annotation or the return type that says what they hold.

A field is `f:<class key>.<name>`, a method's return `r:<fact>:<function node>`; both are
keys in the type tables (cuda_types.py).
"""

from ..base import CudaRelation


class CudaAncestors(CudaRelation):
    """The classes a class inherits from, itself first (`distance` 0), up to four levels."""

    stored = True
    query = """
        WITH RECURSIVE reach(class_key, ancestor_key, distance) AS (
          SELECT key, key, 0 FROM cuda_definitions() WHERE kind = 'class'
          UNION
          SELECT r.class_key, b.definition, r.distance + 1
          FROM reach r
          JOIN cuda_bases() b ON b.class_key = r.ancestor_key
          WHERE r.distance < 4 AND b.definition IS NOT NULL
        )
        SELECT * FROM reach
    """


class CudaBases(CudaRelation):
    """The base classes each class writes, resolved to the classes of the project."""

    stored = True
    query = """
        SELECT d.key AS class_key, b.name AS written, c.qualified, c.definition
        FROM cuda_definitions() d
        JOIN (SELECT fact_id, node_id, unnest(parameters) AS b FROM cuda_nodes()
              WHERE type = 'class_definition') n ON n.fact_id = d.fact_id AND n.node_id = d.node_id
        LEFT JOIN cuda_chains() c ON c.fact_id = d.fact_id AND c.chain = b.name
        WHERE d.kind = 'class' AND b.type = 'extends'
    """


class CudaMethods(CudaRelation):
    """The methods each class defines: `kind` is `property` for one a decorator makes an
    attribute, `method` otherwise; `ret` the key of its return type."""

    stored = True
    query = """
        WITH decorated AS (
          SELECT f.fact_id, f.node_id,
                 bool_or(regexp_matches(coalesce(e.name, ''),
                         '^(property|cached_property|functools\\.cached_property|[\\w.]*\\.setter)$'))
                   AS is_property
          FROM cuda_nodes() f
          JOIN cuda_nodes() w ON w.fact_id = f.fact_id AND w.node_id = f.parent_id
            AND w.type = 'decorated_definition'
          JOIN cuda_nodes() d ON d.fact_id = w.fact_id AND d.parent_id = w.node_id
            AND d.type = 'decorator'
          JOIN cuda_nodes() e ON e.fact_id = d.fact_id AND e.parent_id = d.node_id
            AND e.sibling_index = 1
          WHERE f.type = 'function_definition'
          GROUP BY f.fact_id, f.node_id
        )
        SELECT c.key AS class_key, m.name, m.fact_id, m.node_id,
               'r:' || m.fact_id || ':' || m.node_id AS ret,
               CASE WHEN coalesce(d.is_property, false) THEN 'property' ELSE 'method' END AS kind
        FROM cuda_definitions() m
        JOIN cuda_definitions() c ON c.fact_id = m.fact_id AND c.node_id = m.class_id
          AND c.kind = 'class'
        LEFT JOIN decorated d ON d.fact_id = m.fact_id AND d.node_id = m.node_id
        WHERE m.kind = 'function' AND m.outer_id IS NULL
    """


class CudaFields(CudaRelation):
    """The attributes an instance of a class holds, by the key (`f:<class>.<name>`) the type
    tables name them with: those annotated in the class body or on `self`, and those a method
    assigns on `self`."""

    stored = True
    query = """
        SELECT DISTINCT c.key AS class_key, a.name,
                        'f:' || c.key || '.' || a.name AS field
        FROM cuda_annotations() a
        JOIN cuda_definitions() c ON c.fact_id = a.fact_id AND c.node_id = a.class_id
          AND c.kind = 'class'
        WHERE a.site = 'field' OR (a.site = 'local' AND a.is_attribute)
        UNION
        SELECT DISTINCT c.key, f.name, 'f:' || c.key || '.' || f.name
        FROM cuda_field_stores() f
        JOIN cuda_definitions() c ON c.fact_id = f.fact_id AND c.node_id = f.class_id
          AND c.kind = 'class'
    """


class CudaFieldStores(CudaRelation):
    """Each `self.name = value` in a method (a class-body `name = value` too, and the elements of
    `self.a, self.b = a, b`): the field and the value's node, with the class holding it."""

    stored = True
    query = r"""
        WITH rhs AS (
          SELECT a.fact_id, a.node_id AS assignment_id, r.node_id AS rhs_id, r.type AS rhs_type,
                 r.children_count AS rhs_count
          FROM cuda_nodes() a JOIN cuda_nodes() r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
          WHERE a.type = 'assignment'
          QUALIFY row_number() OVER (PARTITION BY a.fact_id, a.node_id
                                     ORDER BY r.sibling_index DESC) = 1
        )
        SELECT a.fact_id, a.scope."class" AS class_id, t.node_id AS target_id,
               regexp_replace(t.name, '^(self|cls)\.', '') AS name,
               CASE WHEN x.rhs_type = 'type' THEN NULL ELSE x.rhs_id END AS source_id
        FROM cuda_nodes() a
        JOIN cuda_nodes() t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id
          AND t.sibling_index = 0 AND t.type = 'attribute'
          AND regexp_matches(t.name, '^(self|cls)\.\w+$')
        JOIN rhs x ON x.fact_id = a.fact_id AND x.assignment_id = a.node_id
        WHERE a.type = 'assignment' AND a.scope."class" IS NOT NULL
        UNION ALL
        SELECT a.fact_id, a.scope."class", t.node_id, regexp_replace(t.name, '^(self|cls)\.', ''),
               (SELECT e.node_id FROM cuda_nodes() e WHERE e.fact_id = x.fact_id
                  AND e.parent_id = x.rhs_id AND e.sibling_index = t.sibling_index)
        FROM cuda_nodes() a
        JOIN cuda_nodes() p ON p.fact_id = a.fact_id AND p.parent_id = a.node_id
          AND p.sibling_index = 0 AND p.type IN ('pattern_list', 'tuple_pattern')
        JOIN cuda_nodes() t ON t.fact_id = a.fact_id AND t.parent_id = p.node_id
          AND t.type = 'attribute' AND regexp_matches(t.name, '^(self|cls)\.\w+$')
        JOIN rhs x ON x.fact_id = a.fact_id AND x.assignment_id = a.node_id
          AND x.rhs_type IN ('expression_list', 'tuple') AND x.rhs_count = p.children_count
        WHERE a.type = 'assignment' AND a.scope."class" IS NOT NULL
    """


class CudaMembers(CudaRelation):
    """What an instance of a class answers to, the class's own members and those it inherits
    (the nearest definition wins): a `field` or a `property` (read as an attribute) with the
    key of its type, a `method` with the key of its return type."""

    stored = True
    query = """
        WITH members AS (
          SELECT class_key, name, field AS member, 'field' AS kind FROM cuda_fields()
          UNION ALL
          SELECT class_key, name, ret, kind FROM cuda_methods()
        )
        SELECT a.class_key AS receiver, m.name, m.member, m.kind
        FROM cuda_ancestors() a JOIN members m ON m.class_key = a.ancestor_key
        QUALIFY row_number() OVER (PARTITION BY a.class_key, m.name, m.kind
                                   ORDER BY a.distance) = 1
    """


class CudaFieldOrder(CudaRelation):
    """The annotated fields of a class in the order its body writes them, which is the order a
    tuple-like class (a `NamedTuple`, a record) unpacks them in."""

    stored = True
    query = """
        SELECT c.key AS class_key, f.field,
               row_number() OVER (PARTITION BY c.key ORDER BY a.start_line, a.start_column) - 1
                 AS position
        FROM cuda_annotations() a
        JOIN cuda_definitions() c ON c.fact_id = a.fact_id AND c.node_id = a.class_id
          AND c.kind = 'class'
        JOIN cuda_fields() f ON f.class_key = c.key AND f.name = a.name
        WHERE a.site = 'field'
    """
