"""The edges along which a type flows, to a fixpoint, from the seeds of `core.py`.

Edges: a name carries the type of its variable; an assignment gives a variable the type of its
value when every assignment to it agrees; an operator, a subscript and a conditional carry their
operands' array type; a call carries its callee's return type; a method or attribute of an
instance carries the member's type, found through the instance's class and the classes it
inherits from; a field takes the type of what a constructor or `self.field = value` stores in it.
"""

from ..base import CudaRelation

_ARRAY_NODES = (
    "'identifier', 'attribute', 'subscript', 'call', 'binary_operator', 'unary_operator',"
    " 'parenthesized_expression', 'conditional_expression', 'comparison_operator',"
    " 'boolean_operator', 'named_expression'"
)


class CudaTypeEdges(CudaRelation):
    """Where a type moves unchanged from one key to another. `only` restricts what moves:
    `arrays` lets a `device`, `tensor` or `array` through, nothing else, and a null `only` lets any type."""

    stored = True
    query = f"""
        SELECT 'e:' || c.fact_id || ':' || c.node_id AS src,
               'e:' || p.fact_id || ':' || p.node_id AS dst, CAST(NULL AS VARCHAR) AS only
        FROM cuda_nodes() p JOIN cuda_nodes() c ON c.fact_id = p.fact_id AND c.parent_id = p.node_id
        WHERE c.type IN ({_ARRAY_NODES}) AND (
              p.type IN ('parenthesized_expression', 'boolean_operator')
           OR (p.type = 'conditional_expression' AND c.sibling_index IN (0, 4))
           OR (p.type = 'named_expression' AND c.sibling_index = 2))
        UNION ALL
        SELECT 'e:' || c.fact_id || ':' || c.node_id, 'e:' || p.fact_id || ':' || p.node_id, 'arrays'
        FROM cuda_nodes() p JOIN cuda_nodes() c ON c.fact_id = p.fact_id AND c.parent_id = p.node_id
        WHERE c.type IN ({_ARRAY_NODES}) AND (
              (p.type = 'subscript' AND c.sibling_index = 0)
           OR p.type IN ('binary_operator', 'unary_operator')
           OR (p.type = 'comparison_operator' AND NOT EXISTS (
                 SELECT 1 FROM cuda_nodes() o WHERE o.fact_id = p.fact_id AND o.parent_id = p.node_id
                   AND (o.type IN ('is', 'is not', 'in', 'not in')
                        OR o.type = 'none' AND EXISTS (
                          SELECT 1 FROM cuda_nodes() q WHERE q.fact_id = p.fact_id
                            AND q.parent_id = p.node_id AND q.type IN ('==', '!='))))))
        UNION ALL
        SELECT variable, 'e:' || fact_id || ':' || node_id, NULL FROM cuda_name_uses()
        UNION ALL
        SELECT 'e:' || s.fact_id || ':' || s.source_id,
               'v:' || s.fact_id || ':' || s.scope_id || ':' || s.name, 'arrays'
        FROM cuda_stores() s WHERE s.kind = 'loop' AND s.source_id IS NOT NULL
        UNION ALL
        SELECT 'e:' || c.fact_id || ':' || c.node_id, 'r:' || r.fact_id || ':' || r.scope."function", NULL
        FROM cuda_nodes() r JOIN cuda_nodes() c ON c.fact_id = r.fact_id AND c.parent_id = r.node_id
          AND c.sibling_index = 1
        WHERE r.type = 'return_statement' AND r.scope."function" IS NOT NULL
          AND c.type IN ({_ARRAY_NODES})
        UNION ALL
        SELECT DISTINCT 'r:' || t.definition_fact || ':' || t.definition_id,
               'e:' || t.fact_id || ':' || t.node_id, NULL
        FROM cuda_call_targets() t WHERE t.kind = 'function'
        UNION ALL
        SELECT 'e:' || f.fact_id || ':' || f.source_id, m.field, NULL
        FROM cuda_field_stores() f
        JOIN cuda_definitions() c ON c.fact_id = f.fact_id AND c.node_id = f.class_id
        JOIN cuda_fields() m ON m.class_key = c.key AND m.name = f.name
        WHERE f.source_id IS NOT NULL
        UNION ALL
        SELECT DISTINCT 'e:' || a.fact_id || ':' || a.value_id, m.member, NULL
        FROM cuda_call_targets() c
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
          AND a.keyword IS NOT NULL
        JOIN cuda_members() m ON m.receiver = c.key AND m.name = a.keyword AND m.kind = 'field'
        WHERE c.kind = 'class'
        UNION ALL
        SELECT DISTINCT 'e:' || a.fact_id || ':' || a.value_id,
               'v:' || d.fact_id || ':' || d.node_id || ':' || p.name, NULL
        FROM cuda_call_targets() c
        JOIN cuda_definitions() d ON d.kind = 'function'
          AND d.key IN (c.key, c.key || '.__init__')
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
        JOIN cuda_parameters() p ON p.fact_id = d.fact_id AND p.function_id = d.node_id
          AND p.annotation = ''
          AND (p.name = a.keyword
               OR p.position = a.position + 1
                  + CASE WHEN d.class_id IS NOT NULL AND d.outer_id IS NULL
                          AND NOT EXISTS (SELECT 1 FROM cuda_decorators() x
                                          WHERE x.fact_id = d.fact_id
                                            AND x.definition_id = d.node_id
                                            AND x.written = 'staticmethod')
                         THEN 1 ELSE 0 END)
    """


class CudaUnpacks(CudaRelation):
    """The names a tuple assignment binds: the variable, the value assigned and the name's place."""

    stored = True
    query = """
        SELECT 'v:' || fact_id || ':' || scope_id || ':' || name AS variable,
               'e:' || fact_id || ':' || source_id AS source, position
        FROM cuda_stores() WHERE kind = 'unpack' AND source_id IS NOT NULL
    """


class CudaElements(CudaRelation):
    """Where one element of a container is taken out: the `source` expression and the `element`
    key it yields, a loop variable (`for stage in stages`) or an item (`stages[0]`, not a slice)."""

    stored = True
    query = """
        SELECT 'e:' || fact_id || ':' || source_id AS source,
               'v:' || fact_id || ':' || scope_id || ':' || name AS element
        FROM cuda_stores() WHERE kind = 'loop' AND source_id IS NOT NULL
        UNION ALL
        SELECT 'e:' || o.fact_id || ':' || o.node_id, 'e:' || p.fact_id || ':' || p.node_id
        FROM cuda_nodes() p
        JOIN cuda_nodes() o ON o.fact_id = p.fact_id AND o.parent_id = p.node_id
          AND o.sibling_index = 0
        JOIN cuda_nodes() i ON i.fact_id = p.fact_id AND i.parent_id = p.node_id
          AND i.sibling_index = 2
        WHERE p.type = 'subscript' AND i.type <> 'slice'
    """


class CudaVariableSources(CudaRelation):
    """The value each plain `name = value` stores in a variable. `literal` marks a `None`, which
    says nothing of the variable's type (`out = None` before `out = cupy.empty(n)`)."""

    stored = True
    query = """
        SELECT 'v:' || s.fact_id || ':' || s.scope_id || ':' || s.name AS variable,
               'e:' || s.fact_id || ':' || s.source_id AS source,
               v.type = 'none' AS literal
        FROM cuda_stores() s
        JOIN cuda_nodes() v ON v.fact_id = s.fact_id AND v.node_id = s.source_id
        WHERE s.kind = 'assignment' AND s.source_id IS NOT NULL
    """
