"""Rules over PyTorch call chains (mcmr's PY-TORC family)."""

from ....catalog import Rule

_TENSOR_METHODS = ", ".join(f"'torch.{name}'" for name in (
    "abs", "acos", "asin", "atan", "ceil", "cos", "cosh", "erf", "erfinv", "exp", "exp2", "expm1",
    "floor", "frac", "log", "log10", "log1p", "log2", "neg", "reciprocal", "relu", "round",
    "rsqrt", "sigmoid", "sign", "sin", "sinh", "sqrt", "square", "tan", "tanh", "trunc",
))  # fmt: skip


class FluentTensorCallChain(Rule, id="PY-TORC0001", applicability="safe"):
    """Nested `torch.f(torch.g(x))` calls that read better as the fluent chain `x.g().f()`."""

    minimum_operations: int = 2

    query = rf"""
            WITH edges AS (
              SELECT DISTINCT parent_id, parent_kind, child_expression_id, ordinal
              FROM call_expression_ancestry WHERE relation = 'argument'
            ), operands AS (
              SELECT a.parent_id, a.parent_kind, count(*) AS arguments,
                     max(CASE WHEN a.ordinal = 0 THEN e.text END) AS first_text,
                     max(CASE WHEN a.ordinal = 0 THEN e.literal_kind END) AS first_kind
              FROM edges a JOIN call_expressions e ON e.expression_id = a.child_expression_id
              GROUP BY ALL
            ), parents AS (
              SELECT call_id AS parent_id, 'call' AS parent_kind, qualified_name FROM call_calls
              WHERE NOT is_shadowed AND call_id NOT IN (SELECT call_id FROM call_keywords)
              UNION ALL
              SELECT expression_id, 'expression', qualified_name FROM call_expressions
            ), operations AS (
              SELECT parent_id, parent_kind, power::INT AS operand,
                     CASE WHEN power THEN 'exp2'
                          ELSE regexp_replace(qualified_name, '^torch\.', '') END AS method
              FROM (SELECT *, qualified_name IN ('torch.pow', 'torch.float_power')
                                AND arguments = 2 AND first_kind = 'number'
                                AND first_text IN ('2', '2.0') AS power
                    FROM parents JOIN operands USING (parent_id, parent_kind))
              WHERE power OR (qualified_name IN ({_TENSOR_METHODS}) AND arguments = 1)
            ), foldable AS (
              SELECT a.parent_id, a.parent_kind, a.child_expression_id, o.method
              FROM edges a JOIN operations o ON o.parent_id = a.parent_id
                AND o.parent_kind = a.parent_kind AND o.operand = a.ordinal
            ), descents AS (
              SELECT x.call_id, x.descendant_expression_id, count(*) AS operation_count,
                     string_agg('`' || f.method || '`', ', ' ORDER BY x.step DESC) AS method_names,
                     string_agg('.' || f.method || '()', '' ORDER BY x.step DESC)
                       AS ordinary_chain,
                     string_agg('.' || f.method || '_()', '' ORDER BY x.step DESC)
                       AS in_place_chain
              FROM call_expression_ancestry x
              LEFT JOIN foldable f USING (parent_id, parent_kind, child_expression_id)
              WHERE x.call_id IN (SELECT parent_id FROM operations WHERE parent_kind = 'call')
              GROUP BY x.call_id, x.descendant_expression_id
              HAVING count(f.method) = count(*)
              QUALIFY operation_count = max(operation_count) OVER (PARTITION BY x.call_id)
            ), chains AS (
              SELECT c.call_id, c.fact_id, c.assigned_target,
                     CASE WHEN c.node_text <> '' THEN c.node_text ELSE c.qualified_name END
                       AS shown,
                     c.node_path AS path, c.node_start_line AS start_line,
                     c.node_start_column AS start_column, c.node_end_line AS end_line,
                     c.node_end_column AS end_column, d.operation_count, d.method_names,
                     d.ordinary_chain, d.in_place_chain, e.text AS tensor_text
              FROM call_calls c JOIN descents d USING (call_id)
              JOIN call_expressions e ON e.expression_id = d.descendant_expression_id
              WHERE d.operation_count >= $minimum_operations
            )
            SELECT * FROM chains i WHERE NOT EXISTS (
              SELECT 1 FROM chains o
              WHERE o.fact_id = i.fact_id AND o.call_id <> i.call_id
                AND o.operation_count > i.operation_count
                AND (o.start_line < i.start_line
                     OR (o.start_line = i.start_line AND o.start_column <= i.start_column))
                AND (o.end_line > i.end_line
                     OR (o.end_line = i.end_line AND o.end_column >= i.end_column)))
        """
    message = "`{shown}` reverses the fluent chain on `{tensor_text}` through {method_names}"
    # In place (`x = x.log2_().round_()`) when the chain's result replaces its own tensor.
    fix = """
        SELECT *, tensor_text || CASE WHEN assigned_target <> '' AND assigned_target = tensor_text
                                      THEN in_place_chain ELSE ordinary_chain END AS text
        FROM finding
    """
