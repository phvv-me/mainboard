"""Helpers that pack two values into one word, and which operand each takes as the high half."""

from ..base import CudaRelation


class CudaPackers(CudaRelation):
    """Functions that return `(high << shift) | low` (or `+`, `^`) of two of their parameters,
    each possibly cast by a one-argument call: `high_first` says whether the high half is the
    earlier parameter."""

    stored = True
    query = """
        WITH combined AS (
          SELECT r.fact_id, r.scope."function" AS function_id, e.node_id AS node_id,
                 e.children_count AS children
          FROM cuda_nodes() r
          JOIN cuda_nodes() e ON e.fact_id = r.fact_id
            AND e.node_id BETWEEN r.node_id + 1 AND r.node_id + r.descendant_count
            AND e.type = 'binary_operator'
          WHERE r.type = 'return_statement' AND r.scope."function" IS NOT NULL
        ), operands AS (
          SELECT c.fact_id, c.function_id, c.node_id AS combined_id, o.sibling_index AS side,
                 CASE WHEN o.type = 'parenthesized_expression'
                      THEN (SELECT i.node_id FROM cuda_nodes() i
                            WHERE i.fact_id = o.fact_id AND i.parent_id = o.node_id
                              AND i.type NOT IN ('(', ')') LIMIT 1)
                      ELSE o.node_id END AS operand_id
          FROM combined c
          JOIN cuda_nodes() op ON op.fact_id = c.fact_id AND op.parent_id = c.node_id
            AND op.sibling_index = 1 AND op.type IN ('|', '+', '^')
          JOIN cuda_nodes() o ON o.fact_id = c.fact_id AND o.parent_id = c.node_id
            AND o.sibling_index IN (0, 2)
        ), named AS (
          SELECT p.fact_id, p.function_id, p.combined_id, p.side, n.type, n.node_id,
                 CASE WHEN n.type = 'identifier' THEN n.name
                      WHEN n.type = 'call' THEN (
                        SELECT v.name FROM cuda_calls() k
                        JOIN cuda_arguments() a ON a.fact_id = k.fact_id AND a.call_id = k.node_id
                          AND a.position = 0
                        JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.node_id = a.value_id
                        WHERE k.fact_id = n.fact_id AND k.node_id = n.node_id AND v.type = 'identifier')
                 END AS parameter
          FROM operands p JOIN cuda_nodes() n ON n.fact_id = p.fact_id AND n.node_id = p.operand_id
        ), shifted AS (
          SELECT n.fact_id, n.function_id, n.combined_id, n.side AS high_side,
                 (SELECT s.name FROM cuda_nodes() s
                  WHERE s.fact_id = n.fact_id AND s.parent_id = n.node_id AND s.sibling_index = 2
                    AND s.type = 'integer') AS shift,
                 (SELECT CASE WHEN v.type = 'identifier' THEN v.name
                              WHEN v.type = 'call' THEN (
                                SELECT w.name FROM cuda_calls() k
                                JOIN cuda_arguments() a ON a.fact_id = k.fact_id
                                  AND a.call_id = k.node_id AND a.position = 0
                                JOIN cuda_nodes() w ON w.fact_id = a.fact_id AND w.node_id = a.value_id
                                WHERE k.fact_id = v.fact_id AND k.node_id = v.node_id
                                  AND w.type = 'identifier') END
                  FROM cuda_nodes() v WHERE v.fact_id = n.fact_id AND v.parent_id = n.node_id
                    AND v.sibling_index = 0) AS high
          FROM named n
          JOIN cuda_nodes() op ON op.fact_id = n.fact_id AND op.parent_id = n.node_id
            AND op.sibling_index = 1 AND op.type = '<<'
          WHERE n.type = 'binary_operator'
        )
        SELECT f.fact_id, f.path, f.qualname, f.start_line, f.start_column, f.end_line,
               f.end_column, s.shift::INTEGER AS shift, s.high, l.parameter AS low,
               hp.position < lp.position AS high_first
        FROM shifted s
        JOIN named l ON l.fact_id = s.fact_id AND l.combined_id = s.combined_id
          AND l.side <> s.high_side AND l.parameter IS NOT NULL
        JOIN cuda_definitions() f ON f.fact_id = s.fact_id AND f.node_id = s.function_id
        JOIN cuda_parameters() hp ON hp.fact_id = s.fact_id AND hp.function_id = s.function_id
          AND hp.name = s.high
        JOIN cuda_parameters() lp ON lp.fact_id = s.fact_id AND lp.function_id = s.function_id
          AND lp.name = l.parameter
        WHERE s.shift IS NOT NULL AND s.high IS NOT NULL
    """
