"""mcmr's PydanticModelFact, derived from ClassFact's relations and sitting_duck's trees.

One fact per Python file, holding the classes that are models (a base whose name, resolved through
the file's imports, contains `Model`) or plain classes (no decorator, no base) that a constructor
might turn into one. For each: its annotated fields (direct statements of the class body, class
variables aside), its `field_validator` and `model_validator` methods with what they read off
their receiver, and the measures of its first `__init__`. The shape is mcmr's. Differences kept on
purpose:

- a subscripted base (`Model[T]`) is read by its name, where the kernel read it as nameless;
- imports resolve through every import of the file, not only its module-level ones;
- the one validator pattern read is `sum(<tuple or list of presence checks>) > 1` guarding a
  `raise ValueError`, written inline or through a variable assigned just before; the kernel also
  read a generator over a named tuple, which no code here writes.
"""

from .....catalog import Family

_TUPLES = "'tuple', 'builtins.tuple', 'typing.Tuple'"
_IDENTITY_METHODS = (
    "'__init__', '__eq__', '__ne__', '__hash__', '__repr__', '__str__', '__lt__', '__le__', "
    "'__gt__', '__ge__', '__post_init__'"
)

_FACTS = """
    SELECT 'pydanticmodelfact:' || path AS fact_id, path, start_line, start_column,
           greatest(end_line, 1) AS end_line, end_column, language
    FROM class_files
"""

_BASES = r"""
    SELECT h.class_id, h.fact_id AS file_id, h.ordinal, h.path, h.start_line, h.start_column,
           h.end_line, h.end_column,
           CASE WHEN g.target IS NULL THEN d.dotted
                ELSE g.target || substr(d.dotted, length(split_part(d.dotted, '.', 1)) + 1)
                END AS resolved
    FROM class_header_items h,
         LATERAL (SELECT trim(regexp_extract(h.text, '^[^\[\(]*')) AS dotted) d
    LEFT JOIN class_bindings g ON g.path = h.path AND g.name = split_part(d.dotted, '.', 1)
    WHERE NOT h.is_keyword
"""

_CANDIDATES = """
    WITH shaped AS (
      SELECT c.class_id, c.fact_id AS file_id, c.path, c.class_node, c.name,
             coalesce(bool_or(contains(b.resolved, 'Model')), false) AS is_pydantic_model,
             min(b.ordinal) FILTER (WHERE regexp_extract(b.resolved, '[^.]*$') = 'FrozenFlexModel')
               AS flexible_ordinal,
             count(b.ordinal) = 0
               AND NOT EXISTS (SELECT 1 FROM class_def_decorations d WHERE d.class_id = c.class_id)
               AND NOT EXISTS (SELECT 1 FROM class_header_items k WHERE k.class_id = c.class_id)
               AS is_undecorated_plain_class
      FROM class_defs c LEFT JOIN pydantic_model_bases b ON b.class_id = c.class_id
      GROUP BY c.class_id, c.fact_id, c.path, c.class_node, c.name
    )
    SELECT 'pydanticmodelfact:' || s.path AS fact_id,
           'pydanticmodelfact:' || s.path || '/models:'
             || (row_number() OVER w - 1) AS record_id,
           row_number() OVER w - 1 AS ordinal, s.class_id, s.file_id, s.path, s.class_node, s.name,
           s.is_pydantic_model, s.flexible_ordinal IS NOT NULL AS uses_flexible_model,
           f.path AS flexible_base_span_path, f.start_line AS flexible_base_span_start_line,
           f.start_column AS flexible_base_span_start_column,
           f.end_line AS flexible_base_span_end_line,
           f.end_column AS flexible_base_span_end_column, s.is_undecorated_plain_class
    FROM shaped s
    LEFT JOIN pydantic_model_bases f ON f.class_id = s.class_id AND f.ordinal = s.flexible_ordinal
    WHERE s.is_pydantic_model OR s.is_undecorated_plain_class
    WINDOW w AS (PARTITION BY s.file_id ORDER BY s.class_node)
"""

_ANNOTATED = """
    SELECT m.fact_id, m.record_id, m.class_id, m.file_id, m.path, l.name,
           t.node_id AS type_node, t.descendant_count AS type_size, t.start_line, t.start_column,
           t.end_line, t.end_column, a.node_id AS assign_node, coalesce(v.type = 'none', false)
             AS defaults_to_none,
           classes_text(f.lines, t.start_line, t.start_column, t.end_line, t.end_column)
             AS annotation
    FROM pydantic_model_candidates m
    JOIN class_blocks b ON b.fact_id = m.file_id AND b.class_node = m.class_node
    JOIN class_nodes s ON s.fact_id = m.file_id AND s.parent_id = b.block_node
                      AND s.type = 'expression_statement'
    JOIN class_nodes a ON a.fact_id = m.file_id AND a.parent_id = s.node_id
                      AND a.type = 'assignment'
    JOIN class_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
                      AND l.sibling_index = 0 AND l.type = 'identifier'
    JOIN class_nodes t ON t.fact_id = a.fact_id AND t.parent_id = a.node_id AND t.type = 'type'
    JOIN class_files f ON f.fact_id = m.file_id
    LEFT JOIN class_nodes v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
                           AND v.sibling_index = a.children_count - 1 AND v.sibling_index > 2
"""

_SHAPES = """
    SELECT x.fact_id, x.type_node, x.outermost, x.variadic_shape,
           CASE WHEN g.target IS NULL THEN x.head
                ELSE g.target || substr(x.head, length(split_part(x.head, '.', 1)) + 1)
                END AS head
    FROM (
      SELECT m.file_id AS fact_id, m.path, m.type_node, n.parent_id = m.type_node AS outermost,
             h.name AS head,
             coalesce(n.type = 'generic_type' AND tp.children_count = 4 AND e.type = 'ellipsis'
                      OR n.type = 'subscript' AND n.children_count = 5 AND z.type = 'ellipsis',
                      false) AS variadic_shape
      FROM pydantic_model_annotated m
      JOIN class_nodes n ON n.fact_id = m.file_id AND n.type IN ('generic_type', 'subscript')
                        AND n.node_id BETWEEN m.type_node AND m.type_node + m.type_size
      JOIN class_nodes h ON h.fact_id = n.fact_id AND h.parent_id = n.node_id
                        AND h.sibling_index = 0
      LEFT JOIN class_nodes tp ON tp.fact_id = n.fact_id AND tp.parent_id = n.node_id
                              AND tp.type = 'type_parameter' AND n.type = 'generic_type'
      LEFT JOIN class_nodes x ON x.fact_id = n.fact_id AND x.parent_id = tp.node_id
                             AND x.sibling_index = 2 AND x.type = 'type'
      LEFT JOIN class_nodes e ON e.fact_id = n.fact_id AND e.parent_id = x.node_id
                             AND e.sibling_index = 0
      LEFT JOIN class_nodes z ON z.fact_id = n.fact_id AND z.parent_id = n.node_id
                             AND z.sibling_index = 3 AND n.type = 'subscript'
    ) x
    LEFT JOIN class_bindings g ON g.path = x.path AND g.name = split_part(x.head, '.', 1)
"""

_FIELDS = r"""
    SELECT m.fact_id, m.record_id AS parent_id,
           m.record_id || '/models.fields:' || (row_number() OVER w - 1) AS record_id,
           row_number() OVER w - 1 AS ordinal, m.name, m.annotation, m.path AS span_path,
           m.start_line AS span_start_line, m.start_column AS span_start_column,
           m.end_line AS span_end_line, m.end_column AS span_end_column,
           coalesce(bool_or(s.variadic_shape AND s.head IN (__TUPLES__)), false)
             AS contains_variadic_tuple,
           m.defaults_to_none AND (regexp_matches(m.annotation, '^(?:\w+\.)*Optional\b')
                                   OR regexp_matches(m.annotation, '(^|[^\w.])None($|\W)'))
             AS is_optional
    FROM pydantic_model_annotated m
    LEFT JOIN pydantic_model_shapes s ON s.fact_id = m.file_id AND s.type_node = m.type_node
    WHERE NOT EXISTS (SELECT 1 FROM pydantic_model_shapes c
                      WHERE c.fact_id = m.file_id AND c.type_node = m.type_node
                        AND c.outermost AND c.head = 'typing.ClassVar')
    GROUP BY m.fact_id, m.record_id, m.path, m.name, m.annotation, m.start_line, m.start_column,
             m.end_line, m.end_column, m.assign_node, m.defaults_to_none
    WINDOW w AS (PARTITION BY m.record_id ORDER BY m.assign_node)
""".replace("__TUPLES__", _TUPLES)

_VALIDATOR_DEFS = r"""
    WITH marked AS (
      SELECT m.fact_id, m.record_id, m.file_id, m.class_id, d.method_node, d.start_line,
             c.dotted, c.text, c.ordinal
      FROM pydantic_model_candidates m
      JOIN class_member_defs d ON d.class_id = m.class_id
      JOIN class_decorations c ON c.fact_id = d.fact_id AND c.decorated_node = d.statement_node
      WHERE regexp_extract(c.dotted, '[^.]*$') IN ('field_validator', 'model_validator')
    ), first AS (
      SELECT fact_id, record_id, file_id, class_id, method_node,
             arg_min(CASE WHEN regexp_extract(dotted, '[^.]*$') = 'field_validator' THEN 'field'
                          WHEN regexp_matches(text, 'mode\s*=\s*["'']after["'']')
                          THEN 'model_after' ELSE 'other' END, ordinal) AS kind
      FROM marked GROUP BY fact_id, record_id, file_id, class_id, method_node
    )
    SELECT f.*, f.record_id || '/models.validators:'
                  || (row_number() OVER w - 1) AS validator_id,
           row_number() OVER w - 1 AS ordinal,
           (SELECT p.name FROM class_member_params p
            WHERE p.fact_id = f.file_id AND p.method_node = f.method_node
            ORDER BY p.idx LIMIT 1) AS receiver,
           (SELECT p.name FROM class_member_params p
            WHERE p.fact_id = f.file_id AND p.method_node = f.method_node
            ORDER BY p.idx LIMIT 1 OFFSET 1) AS value
    FROM first f WINDOW w AS (PARTITION BY f.record_id ORDER BY f.method_node)
"""

_READS = r"""
    SELECT v.validator_id, a.name AS attribute_text, c.node_id AS call_node,
           substr(a.name, length(v.receiver) + 2) AS field
    FROM pydantic_model_validator_defs v
    JOIN class_nodes b ON b.fact_id = v.file_id AND b.parent_id = v.method_node
                      AND b.type = 'block'
    JOIN class_nodes a ON a.fact_id = v.file_id AND a.type = 'attribute'
                      AND a.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
                      AND starts_with(a.name, v.receiver || '.')
                      AND regexp_matches(substr(a.name, length(v.receiver) + 2), '^\w+$')
    LEFT JOIN class_nodes c ON c.fact_id = a.fact_id AND c.type = 'call'
                           AND c.node_id = a.parent_id AND a.sibling_index = 0
"""

_CONSTRAINTS = """
    SELECT v.validator_id, count(*) AS constraints
    FROM pydantic_model_validator_defs v
    JOIN class_nodes b ON b.fact_id = v.file_id AND b.parent_id = v.method_node
                      AND b.type = 'block'
    JOIN class_nodes n ON n.fact_id = v.file_id AND n.node_id BETWEEN b.node_id
                          AND b.node_id + b.descendant_count
    LEFT JOIN class_nodes f ON f.fact_id = n.fact_id AND f.parent_id = n.node_id
                           AND f.sibling_index = 0 AND n.type = 'call'
    LEFT JOIN class_nodes x ON x.fact_id = n.fact_id AND x.parent_id = n.node_id
                           AND x.sibling_index = 0 AND n.type = 'comparison_operator'
    LEFT JOIN class_nodes y ON y.fact_id = n.fact_id AND y.parent_id = n.node_id
                           AND y.sibling_index = 2 AND n.type = 'comparison_operator'
    WHERE (n.type = 'call' AND f.type = 'attribute'
           AND f.name IN (v.value || '.strip', v.value || '.lower', v.value || '.upper'))
       OR (n.type = 'comparison_operator' AND n.children_count = 3
           AND ((y.type IN ('integer', 'float')
                 AND (x.type = 'identifier' AND x.name = v.value
                      OR x.type = 'call' AND x.name = 'len' AND EXISTS (
                           SELECT 1 FROM class_nodes q
                           JOIN class_nodes r ON r.fact_id = q.fact_id AND r.parent_id = q.node_id
                                             AND r.sibling_index = 1
                           WHERE q.fact_id = x.fact_id AND q.parent_id = x.node_id
                             AND q.type = 'argument_list' AND q.children_count = 3
                             AND r.type = 'identifier' AND r.name = v.value)))
                OR (x.type IN ('integer', 'float') AND y.type = 'identifier'
                    AND y.name = v.value)))
    GROUP BY v.validator_id
"""

_PRESENCE = r"""
    SELECT v.validator_id, c.node_id AS presence_node,
           substr(a.name, length(v.receiver) + 2) AS field
    FROM pydantic_model_validator_defs v
    JOIN class_nodes b ON b.fact_id = v.file_id AND b.parent_id = v.method_node
                      AND b.type = 'block'
    JOIN class_nodes c ON c.fact_id = v.file_id AND c.type = 'comparison_operator'
                      AND c.children_count = 3
                      AND c.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
    JOIN class_nodes a ON a.fact_id = c.fact_id AND a.parent_id = c.node_id AND a.sibling_index = 0
                      AND a.type = 'attribute' AND starts_with(a.name, v.receiver || '.')
                      AND regexp_matches(substr(a.name, length(v.receiver) + 2), '^\w+$')
    JOIN class_nodes o ON o.fact_id = c.fact_id AND o.parent_id = c.node_id
                      AND o.sibling_index = 1 AND o.type = 'is not'
    JOIN class_nodes n ON n.fact_id = c.fact_id AND n.parent_id = c.node_id
                      AND n.sibling_index = 2 AND n.type = 'none'
"""

_GUARDS = """
    WITH guards AS (
      SELECT v.validator_id, v.file_id, i.node_id AS guard, i.sibling_index, i.parent_id AS block,
             l.type AS left_type, l.node_id AS left_node, l.name AS left_name
      FROM pydantic_model_validator_defs v
      JOIN class_nodes m ON m.fact_id = v.file_id AND m.node_id = v.method_node
      JOIN class_nodes i ON i.fact_id = v.file_id AND i.type = 'if_statement'
                        AND i.node_id BETWEEN m.node_id AND m.node_id + m.descendant_count
      JOIN class_nodes cmp ON cmp.fact_id = i.fact_id AND cmp.parent_id = i.node_id
                          AND cmp.sibling_index = 1 AND cmp.type = 'comparison_operator'
                          AND cmp.children_count = 3
      JOIN class_nodes l ON l.fact_id = cmp.fact_id AND l.parent_id = cmp.node_id
                        AND l.sibling_index = 0
      JOIN class_nodes g ON g.fact_id = cmp.fact_id AND g.parent_id = cmp.node_id
                        AND g.sibling_index = 1 AND g.type = '>'
      JOIN class_nodes one ON one.fact_id = cmp.fact_id AND one.parent_id = cmp.node_id
                          AND one.sibling_index = 2 AND one.type = 'integer' AND one.name = '1'
      JOIN class_nodes body ON body.fact_id = i.fact_id AND body.parent_id = i.node_id
                           AND body.type = 'block' AND body.children_count = 1
      JOIN class_nodes r ON r.fact_id = i.fact_id AND r.parent_id = body.node_id
                        AND r.type = 'raise_statement'
      JOIN class_nodes e ON e.fact_id = i.fact_id AND e.parent_id = r.node_id AND e.type = 'call'
                        AND regexp_extract(e.name, '[^.]*$') IN ('ValueError',
                              'PydanticCustomError')
      WHERE NOT EXISTS (
        SELECT 1 FROM class_nodes p
        JOIN class_nodes q ON q.fact_id = p.fact_id AND q.node_id = p.parent_id
        WHERE p.fact_id = v.file_id AND p.node_id BETWEEN m.node_id AND
              m.node_id + m.descendant_count
          AND p.type = 'identifier' AND p.name = 'sum'
          AND (q.type = 'parameters' OR (q.type = 'assignment' AND p.sibling_index = 0)))
    )
    SELECT g.validator_id, g.file_id, g.guard, c.node_id AS sum_call
    FROM guards g
    JOIN class_nodes c ON c.fact_id = g.file_id AND c.node_id = g.left_node AND c.type = 'call'
                      AND c.name = 'sum'
    UNION ALL
    SELECT g.validator_id, g.file_id, g.guard, c.node_id
    FROM guards g
    JOIN class_nodes prev ON prev.fact_id = g.file_id AND prev.parent_id = g.block
                         AND prev.type <> 'comment' AND prev.sibling_index = (
                           SELECT max(p.sibling_index) FROM class_nodes p
                           WHERE p.fact_id = g.file_id AND p.parent_id = g.block
                             AND p.type <> 'comment' AND p.sibling_index < g.sibling_index)
    JOIN class_nodes s ON s.fact_id = g.file_id AND s.parent_id = prev.node_id
                      AND s.type = 'assignment' AND s.name = g.left_name
    JOIN class_nodes c ON c.fact_id = g.file_id AND c.parent_id = s.node_id AND c.type = 'call'
                      AND c.name = 'sum' AND c.sibling_index = s.children_count - 1
    WHERE g.left_type = 'identifier'
"""

_VARIANTS = """
    WITH items AS (
      SELECT s.validator_id, s.file_id, s.guard, i.node_id AS item,
             i.descendant_count AS item_size, i.type AS item_type
      FROM pydantic_model_guards s
      JOIN class_nodes a ON a.fact_id = s.file_id AND a.parent_id = s.sum_call
                        AND a.type = 'argument_list' AND a.children_count = 3
      JOIN class_nodes l ON l.fact_id = s.file_id AND l.parent_id = a.node_id
                        AND l.sibling_index = 1 AND l.type IN ('list', 'tuple')
      JOIN class_nodes i ON i.fact_id = s.file_id AND i.parent_id = l.node_id
                        AND i.type NOT IN ('[', ']', '(', ')', 'comment')
    ), operands AS (
      SELECT t.validator_id, t.guard, t.item, n.node_id AS operand, p.field
      FROM items t
      JOIN class_nodes n ON n.fact_id = t.file_id AND n.node_id BETWEEN t.item
                            AND t.item + t.item_size
      JOIN class_nodes q ON q.fact_id = n.fact_id AND q.node_id = n.parent_id
      LEFT JOIN pydantic_model_presence p ON p.validator_id = t.validator_id
                                         AND p.presence_node = n.node_id
      WHERE (n.node_id = t.item AND n.type <> 'boolean_operator')
         OR (q.type = 'boolean_operator' AND n.type NOT IN ('or', 'and', 'boolean_operator'))
    ), guarded AS (
      SELECT t.validator_id, t.guard, count(DISTINCT t.item) AS groups,
             count(o.operand) AS operand_count, count(o.field) AS presence_count,
             count(DISTINCT o.field) AS distinct_fields,
             bool_and(coalesce(f.is_optional, false)) FILTER (WHERE o.field IS NOT NULL)
               AS all_optional,
             bool_or(EXISTS (SELECT 1 FROM class_nodes n
                             WHERE n.fact_id = t.file_id AND n.type = 'and'
                               AND n.node_id BETWEEN t.item AND t.item + t.item_size)) AS has_and
      FROM items t
      JOIN pydantic_model_validator_defs v ON v.validator_id = t.validator_id
      LEFT JOIN operands o ON o.validator_id = t.validator_id AND o.item = t.item
      LEFT JOIN pydantic_model_models_fields f ON f.parent_id = v.record_id AND f.name = o.field
      GROUP BY t.validator_id, t.guard
    )
    SELECT validator_id, max(groups) AS variant_count
    FROM guarded
    WHERE groups >= 2 AND operand_count = presence_count AND distinct_fields = presence_count
      AND all_optional AND NOT has_and
    GROUP BY validator_id
"""

_INITS = """
    WITH defs AS (
      SELECT m.record_id, m.file_id, m.class_id, d.method_node, d.ordinal, n.is_async
      FROM pydantic_model_candidates m
      JOIN class_member_defs d ON d.class_id = m.class_id AND d.name = '__init__'
      JOIN class_nodes n ON n.fact_id = d.fact_id AND n.node_id = d.method_node
    ), first AS (
      SELECT record_id, file_id, count(*) FILTER (WHERE NOT is_async) AS synchronous_init_count,
             arg_min(method_node, ordinal) AS method_node
      FROM defs GROUP BY record_id, file_id
    )
    SELECT f.record_id, f.synchronous_init_count,
           greatest((SELECT count(*) FROM class_member_params p
                     WHERE p.fact_id = f.file_id AND p.method_node = f.method_node) - 1, 0)
             AS fixed_parameter_count,
           (SELECT count(*) FROM class_member_params p
            WHERE p.fact_id = f.file_id AND p.method_node = f.method_node
              AND p.param_type IN ('default_parameter', 'typed_default_parameter'))
             AS default_count,
           (SELECT count(*) FROM class_nodes n JOIN class_nodes m ON m.fact_id = n.fact_id
                                                                 AND m.node_id = f.method_node
            WHERE n.fact_id = f.file_id AND n.node_id BETWEEN m.node_id AND
                  m.node_id + m.descendant_count
              AND n.type IN ('raise_statement', 'assert_statement')) AS validation_count,
           (SELECT count(DISTINCT r.name)
            FROM class_nodes m
            JOIN class_nodes a ON a.fact_id = m.fact_id AND a.type = 'assignment'
                              AND a.children_count = 3
                              AND a.node_id BETWEEN m.node_id AND m.node_id + m.descendant_count
            JOIN class_nodes l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
                              AND l.sibling_index = 0 AND l.type = 'attribute'
                              AND regexp_matches(l.name, '^self\\.\\w+$')
            JOIN class_nodes r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
                              AND r.sibling_index = 2 AND r.type = 'identifier'
            WHERE m.fact_id = f.file_id AND m.node_id = f.method_node
              AND r.name IN (SELECT ltrim(p.name, '*') FROM class_member_params p
                             WHERE p.fact_id = f.file_id AND p.method_node = f.method_node))
             AS stored_parameter_count
    FROM first f
"""

_MODELS = f"""
    SELECT m.fact_id, m.fact_id AS parent_id, m.record_id, m.ordinal, m.name,
           m.is_pydantic_model, m.uses_flexible_model, m.flexible_base_span_path,
           m.flexible_base_span_start_line, m.flexible_base_span_start_column,
           m.flexible_base_span_end_line, m.flexible_base_span_end_column,
           m.is_undecorated_plain_class,
           coalesce(i.synchronous_init_count, 0) AS synchronous_init_count,
           coalesce(i.fixed_parameter_count, 0) AS fixed_parameter_count,
           coalesce(i.stored_parameter_count, 0) AS stored_parameter_count,
           coalesce(i.validation_count, 0) AS validation_count,
           coalesce(i.default_count, 0) AS default_count,
           NOT EXISTS (SELECT 1 FROM class_member_defs d
                       WHERE d.class_id = m.class_id
                         AND d.name NOT IN ({_IDENTITY_METHODS})) AS has_only_data_identity_methods
    FROM pydantic_model_candidates m
    LEFT JOIN pydantic_model_inits i ON i.record_id = m.record_id
"""

_VALIDATORS = """
    SELECT v.fact_id, v.record_id AS parent_id, v.validator_id AS record_id, v.ordinal, v.kind,
           (SELECT count(DISTINCT r.field) FROM pydantic_model_reads r
            WHERE r.validator_id = v.validator_id) AS fields_read_length,
           true AS fields_read_present,
           EXISTS (SELECT 1 FROM pydantic_model_reads r
                   WHERE r.validator_id = v.validator_id AND r.call_node IS NOT NULL)
             AS has_self_call,
           EXISTS (SELECT 1 FROM pydantic_model_reads r
                   WHERE r.validator_id = v.validator_id
                     AND r.field NOT IN (SELECT f.name FROM pydantic_model_models_fields f
                                         WHERE f.parent_id = v.record_id)) AS has_nonfield_access,
           coalesce(c.constraints, 0) AS declarative_constraint_count,
           coalesce(t.variant_count, 0) >= 2 AS proves_disjoint_optional_variants,
           coalesce(t.variant_count, 0) AS variant_count
    FROM pydantic_model_validator_defs v
    LEFT JOIN pydantic_model_constraints c ON c.validator_id = v.validator_id
    LEFT JOIN pydantic_model_variants t ON t.validator_id = v.validator_id
"""

_VALUES = """
    SELECT v.fact_id, 'models.validators.fields_read' AS relation, v.validator_id AS parent_id,
           row_number() OVER (PARTITION BY v.validator_id ORDER BY r.field) - 1 AS ordinal,
           r.field AS string_value
    FROM pydantic_model_validator_defs v
    JOIN (SELECT DISTINCT validator_id, field FROM pydantic_model_reads) r
      ON r.validator_id = v.validator_id
"""


class Pydantic(Family):
    family = "PydanticModelFact"
    relations = {
        "facts": _FACTS,
        "bases": _BASES,
        "candidates": _CANDIDATES,
        "annotated": _ANNOTATED,
        "shapes": _SHAPES,
        "models_fields": _FIELDS,
        "validator_defs": _VALIDATOR_DEFS,
        "reads": _READS,
        "constraints": _CONSTRAINTS,
        "presence": _PRESENCE,
        "guards": _GUARDS,
        "variants": _VARIANTS,
        "inits": _INITS,
        "models": _MODELS,
        "models_validators": _VALIDATORS,
        "values": _VALUES,
    }
