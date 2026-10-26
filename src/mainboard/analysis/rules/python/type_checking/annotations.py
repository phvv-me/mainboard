"""Rules over how annotations are written (mcmr's PY-TYPE annotations group)."""

from collections.abc import Sequence

from ....catalog import Rule

_CONSTRAINTS = (
    "'AfterValidator', 'BeforeValidator', 'Field', 'Ge', 'Gt', 'Interval', 'Le', 'Len', 'Lt', "
    "'MaxLen', 'MinLen', 'MultipleOf', 'PlainSerializer', 'PlainValidator', 'Predicate', "
    "'StringConstraints', 'UrlConstraints', 'WrapSerializer', 'WrapValidator'"
)
_DESCRIBING = (
    "'description', 'title', 'alias', 'validation_alias', 'serialization_alias', 'examples', "
    "'json_schema_extra', 'deprecated'"
)


class ProhibitedAnnotation(Rule, id="PY-TYPE0004"):
    """Annotations that resolve to a name too vague to constrain anything."""

    prohibited: tuple[str, ...] = ("Any", "object")

    query = r"""
        SELECT path, 1 AS start_line, 0 AS start_column,
               'typeannotationfact:' || path AS fact_id, count(*) AS value
        FROM core_annotation
        WHERE list_contains($prohibited::VARCHAR[], regexp_replace(head, '^.*\.', '', 'g'))
        GROUP BY path
    """
    message = "prohibited annotation is {value} for `{fact_id}`"


class RepeatedAnnotatedConstraint(Rule, id="PY-TYPE0007"):
    """`Annotated` constraints written out again and again instead of named once."""

    minimum_repetitions: int = 3
    minimum_files: int = 2
    preferred_modules: Sequence[str] = ("typings.py",)

    query = f"""
        WITH constrained AS (
          SELECT s.file_id, s.path, s.type_node,
                 arg_min(syntax_slice(src.content, n.start_line, n.start_column, n.end_line,
                                      n.end_column), n.node_id) AS recipe
          FROM core_annotation s
          JOIN core_node n ON n.file_id = s.file_id AND n.type IN ('generic_type', 'subscript')
                          AND n.node_id BETWEEN s.type_node AND s.type_node + s.size
          JOIN core_node h ON h.file_id = n.file_id AND h.parent_id = n.node_id
                          AND h.sibling_index = 0
          JOIN source_files src ON src.path = s.path
          WHERE regexp_extract(h.name, '[^.]*$') = 'Annotated' AND EXISTS (
            SELECT 1 FROM core_node tp
            JOIN core_node e ON e.file_id = n.file_id AND e.parent_id = tp.node_id
                            AND e.type = 'type' AND e.sibling_index >= 2
            JOIN core_node m ON m.file_id = n.file_id AND m.parent_id = e.node_id
                            AND m.sibling_index = 0
            WHERE tp.file_id = n.file_id AND tp.parent_id = n.node_id
              AND tp.type = 'type_parameter'
              AND n.type = 'generic_type' AND m.type IN ('identifier', 'attribute', 'call')
              AND regexp_extract(m.name, '[^.]*$') IN ({_CONSTRAINTS})
            UNION ALL
            SELECT 1 FROM core_node m
            WHERE m.file_id = n.file_id AND m.parent_id = n.node_id AND n.type = 'subscript'
              AND m.sibling_index >= 3 AND m.type IN ('identifier', 'attribute', 'call')
              AND regexp_extract(m.name, '[^.]*$') IN ({_CONSTRAINTS}))
          GROUP BY s.file_id, s.path, s.type_node
        ), placed AS (
          SELECT a.path, c.recipe,
                 row_number() OVER (PARTITION BY a.file_id
                                    ORDER BY a.owner, CASE a.role WHEN 'return' THEN 0 ELSE 1 END,
                                             a.type_node) AS ordinal
          FROM core_annotation a
          LEFT JOIN constrained c ON c.file_id = a.file_id AND c.type_node = a.type_node
          WHERE NOT EXISTS (
            SELECT 1 FROM core_node k
            WHERE k.file_id = a.file_id AND k.type = 'keyword_argument'
              AND k.node_id BETWEEN a.type_node AND a.type_node + a.size
              AND k.name IN ({_DESCRIBING}))
        ), recipes AS (
          SELECT arg_min(path, (path, ordinal)) AS path, count(*) AS repetitions,
                 count(DISTINCT path) AS file_count,
                 bool_or(len(list_filter($preferred_modules::VARCHAR[],
                                         lambda m: ends_with(path, m))) > 0)
                   AS has_preferred_module
          FROM placed WHERE recipe IS NOT NULL GROUP BY recipe
        )
        SELECT path, 1 AS start_line, 0 AS start_column, 'typeannotationfact:' || path AS fact_id,
               count(*) AS value
        FROM recipes
        WHERE repetitions >= $minimum_repetitions AND file_count >= $minimum_files
          AND NOT has_preferred_module
        GROUP BY path
    """
    message = "repeated annotated constraint is {value} for `{fact_id}`"
