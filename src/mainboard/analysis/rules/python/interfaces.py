"""Rules over what an interface promises (mcmr's PY-INTE family)."""

from ...catalog import Rule


class ConcreteIsinstanceCapability(Rule, id="PY-INTE0001"):
    """Concrete runtime checks that stand in for a capability.

    A check is an `isinstance(subject, type)` call of two arguments, but for those in an `elif`
    condition. A check in the condition of an `if` is guarded by what the branch's block does with
    the subject: reading an attribute or calling a method of it, indexing it, testing membership
    in it, doing arithmetic on it, looping over it, or passing it to `len`, `iter`, `sorted` and
    the other reading builtins. A subject that is no plain name guards nothing."""

    reading: tuple[str, ...] = (
        "len",
        "iter",
        "reversed",
        "sorted",
        "enumerate",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "next",
    )

    query = """
        WITH sites AS (
          SELECT c.file_id, c.node_id AS call_node,
                 CASE WHEN s.type = 'identifier' THEN s.name ELSE '' END AS subject,
                 CASE WHEN t.type IN ('identifier', 'attribute') THEN t.name ELSE '' END
                   AS concrete_type,
                 g.block_node
          FROM core_node c
          JOIN core_node a ON a.file_id = c.file_id AND a.parent_id = c.node_id
            AND a.type = 'argument_list' AND a.children_count = 4
          JOIN core_node s ON s.file_id = c.file_id AND s.parent_id = a.node_id
            AND s.sibling_index = 1
          JOIN core_node t ON t.file_id = c.file_id AND t.parent_id = a.node_id
            AND t.sibling_index = 2
          LEFT JOIN LATERAL (
            SELECT b.node_id AS block_node
            FROM core_node i
            JOIN core_node k ON k.file_id = i.file_id AND k.parent_id = i.node_id
              AND k.sibling_index = 1
            JOIN core_node b ON b.file_id = i.file_id AND b.parent_id = i.node_id
              AND b.type = 'block'
            WHERE i.file_id = c.file_id AND i.type = 'if_statement'
              AND c.node_id BETWEEN k.node_id AND k.node_id + k.descendant_count
            ORDER BY i.node_id DESC LIMIT 1) g ON true
          WHERE c.type = 'call' AND c.name = 'isinstance'
            AND NOT EXISTS (
              SELECT 1 FROM core_node e
              JOIN core_node k ON k.file_id = e.file_id AND k.parent_id = e.node_id
                AND k.sibling_index = 1
              WHERE e.file_id = c.file_id AND e.type = 'elif_clause'
                AND c.node_id BETWEEN k.node_id AND k.node_id + k.descendant_count)
        ), guarded AS (
          SELECT DISTINCT s.file_id, s.call_node
          FROM sites s
          JOIN core_node b ON b.file_id = s.file_id AND b.node_id = s.block_node,
          LATERAL (
            SELECT n.node_id FROM core_node n
            WHERE n.file_id = s.file_id AND n.type = 'attribute'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND starts_with(n.name, s.subject || '.')
              AND regexp_matches(substr(n.name, length(s.subject) + 2), '^\\w+$')
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node v ON v.file_id = n.file_id AND v.parent_id = n.node_id
              AND v.sibling_index = 0
            WHERE n.file_id = s.file_id AND n.type = 'subscript'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND v.type IN ('identifier', 'attribute') AND v.name = s.subject
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node o ON o.file_id = n.file_id AND o.parent_id = n.node_id
              AND o.sibling_index = 1 AND o.type IN ('in', 'not in')
            JOIN core_node r ON r.file_id = n.file_id AND r.parent_id = n.node_id
              AND r.sibling_index = 2
            WHERE n.file_id = s.file_id AND n.type = 'comparison_operator'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND r.type IN ('identifier', 'attribute') AND r.name = s.subject
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node o ON o.file_id = n.file_id AND o.parent_id = n.node_id
              AND o.sibling_index IN (0, 2)
            WHERE n.file_id = s.file_id AND n.type = 'binary_operator'
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND o.type IN ('identifier', 'attribute') AND o.name = s.subject
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node l ON l.file_id = n.file_id AND l.parent_id = n.node_id
              AND l.type = 'argument_list'
            JOIN core_node x ON x.file_id = n.file_id AND x.parent_id = l.node_id
              AND x.type IN ('identifier', 'attribute') AND x.name = s.subject
            WHERE n.file_id = s.file_id AND n.type = 'call'
              AND list_contains($reading::VARCHAR[], n.name)
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
            UNION ALL
            SELECT n.node_id FROM core_node n
            JOIN core_node r ON r.file_id = n.file_id AND r.parent_id = n.node_id
              AND r.sibling_index = 3
            WHERE n.file_id = s.file_id AND n.type IN ('for_statement', 'for_in_clause')
              AND n.node_id BETWEEN b.node_id AND b.node_id + b.descendant_count
              AND r.type IN ('identifier', 'attribute') AND r.name = s.subject
          ) op
          WHERE s.subject <> ''
        )
        SELECT f.path, 1 AS start_line, 0 AS start_column,
               'runtimetypecheckfact:' || f.path AS fact_id, count(*) AS value
        FROM sites s JOIN core_file f USING (file_id)
        WHERE f.language = 'python'
          AND s.concrete_type IN ('list', 'tuple', 'dict', 'set', 'int', 'float', 'complex')
          AND EXISTS (SELECT 1 FROM guarded g
                      WHERE g.file_id = s.file_id AND g.call_node = s.call_node)
        GROUP BY f.path
    """
    message = "`{fact_id}` satisfies concrete isinstance capability"


class SingleImplementationAbstractBase(Rule, id="PY-INTE0002"):
    """Abstract bases that support only one repository implementation."""

    query = r"""
        SELECT c.*, s.value AS subclass FROM class_classes c
        LEFT JOIN (SELECT class_id, value FROM class_direct_subclasses
                   QUALIFY ordinal = min(ordinal) OVER (PARTITION BY class_id)) s USING (class_id)
        WHERE c.scope = 'module' AND NOT c.is_protocol AND c.descendant_count = 1 AND (
          c.class_id IN (SELECT class_id FROM class_direct_bases
                         WHERE regexp_replace(value, '^.*\.', '', 'g') IN ('ABC', 'ABCMeta'))
          OR c.class_id IN (
            SELECT m.class_id FROM class_method_decorators d JOIN class_methods m USING (method_id)
            WHERE regexp_replace(d.value, '^.*\.', '', 'g') IN (
              'abstractclassmethod', 'abstractmethod', 'abstractproperty',
              'abstractstaticmethod')))
    """
    message = "`{name}` is an abstract base with only `{subclass}` below it"
