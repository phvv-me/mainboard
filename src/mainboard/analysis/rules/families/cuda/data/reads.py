"""Where the host waits for device memory: every read of a device value into Python.

`implicit` reads are the ones that do not look like a transfer: `int(x[0])`, `float(x)`,
`bool(x)`, `range(x)`, `print(x)`, `if x:`, `while x > 0:`. `explicit` reads are spelled as one:
`x.get()`, `x.item()`, `x.tolist()`, `x.cpu()`, `cupy.asnumpy(x)`. `iterate` reads (`list(x)`,
`sum(x)`, `max(x)`) walk the array element by element, one transfer each.
"""

from ..base import CudaRelation


class CudaHostReads(CudaRelation):
    """Each read of a device value into Python, at the expression that reads it."""

    stored = True
    query = """
        WITH spoken AS (
          SELECT c.fact_id, c.node_id AS call_id, a.node_id,
                 row_number() OVER (PARTITION BY c.fact_id, c.node_id ORDER BY a.sibling_index) - 1
                   AS ordinal
          FROM cuda_calls() c JOIN cuda_nodes() a ON a.fact_id = c.fact_id
            AND a.parent_id = c.arguments_id
          WHERE a.type NOT IN ('(', ')', ',', 'comment', 'keyword_argument', 'list_splat',
                               'dictionary_splat')
        ), converting AS (
          SELECT c.fact_id, c.path, c.node_id, c.start_line, c.start_column, c.end_line,
                 c.end_column, c.scope_id, c.shown, t.fw, 'e:' || a.fact_id || ':' || a.node_id AS operand,
                 CASE WHEN c.qualified_name IN ('builtins.int', 'builtins.float', 'builtins.bool',
                        'builtins.complex', 'builtins.range', 'operator.index', 'operator.truth')
                           OR starts_with(c.qualified_name, 'math.')
                      THEN 'implicit'
                      WHEN c.qualified_name IN ('builtins.print', 'builtins.str', 'builtins.repr',
                                                'builtins.format')
                      THEN 'implicit'
                      ELSE 'iterate' END AS kind,
                 regexp_extract(c.qualified_name, '[^.]+$') AS api
          FROM cuda_calls() c
          JOIN spoken a ON a.fact_id = c.fact_id AND a.call_id = c.node_id AND a.ordinal <= 1
          JOIN cuda_types() t ON t.key = 'e:' || a.fact_id || ':' || a.node_id AND t.ty = 'device'
          WHERE c.qualified_name IN ('builtins.int', 'builtins.float', 'builtins.bool',
                  'builtins.complex', 'builtins.range', 'operator.index', 'operator.truth',
                  'builtins.print', 'builtins.str', 'builtins.repr', 'builtins.format',
                  'builtins.list', 'builtins.tuple', 'builtins.sorted', 'builtins.sum',
                  'builtins.max', 'builtins.min', 'builtins.any', 'builtins.all')
             OR starts_with(c.qualified_name, 'math.')
        ), methods AS (
          SELECT c.fact_id, c.path, c.node_id, c.start_line, c.start_column, c.end_line,
                 c.end_column, c.scope_id, c.shown, t.fw, m.receiver_key AS operand,
                 'explicit' AS kind, c.method AS api
          FROM cuda_method_calls() m
          JOIN cuda_calls() c ON 'e:' || c.fact_id || ':' || c.node_id = m.call_key
          JOIN cuda_types() t ON t.key = m.receiver_key AND t.ty = 'device'
          WHERE m.name IN ('get', 'item', 'tolist', 'numpy', 'tobytes', 'cpu', 'copy_to_host')
            AND (c.method <> 'get' OR c.bare OR t.fw IN ('cupy', 'numba', 'protocol', 'patos'))
          UNION ALL
          SELECT c.fact_id, c.path, c.node_id, c.start_line, c.start_column, c.end_line,
                 c.end_column, c.scope_id, c.shown, t.fw, 'e:' || a.fact_id || ':' || a.node_id,
                 'explicit', regexp_extract(c.qualified_name, '[^.]+$')
          FROM cuda_calls() c
          JOIN spoken a ON a.fact_id = c.fact_id AND a.call_id = c.node_id AND a.ordinal = 0
          JOIN cuda_types() t ON t.key = 'e:' || a.fact_id || ':' || a.node_id AND t.ty = 'device'
          WHERE c.qualified_name IN ('cupy.asnumpy', 'cupyx.scipy.sparse.csr_matrix.get')
        ), conditions AS (
          SELECT x.fact_id, x.path, x.node_id, x.start_line, x.start_column, x.end_line,
                 x.end_column, coalesce(x.scope."function", -1) AS scope_id,
                 CAST(NULL AS VARCHAR) AS shown, t.fw, 'e:' || x.fact_id || ':' || x.node_id AS operand,
                 CASE WHEN p.type = 'interpolation' THEN 'implicit' ELSE 'implicit' END AS kind,
                 CASE p.type WHEN 'if_statement' THEN 'if' WHEN 'elif_clause' THEN 'elif'
                             WHEN 'while_statement' THEN 'while' WHEN 'assert_statement' THEN 'assert'
                             WHEN 'conditional_expression' THEN 'if-else'
                             WHEN 'not_operator' THEN 'not' WHEN 'boolean_operator' THEN 'and/or'
                             ELSE 'format' END AS api
          FROM cuda_nodes() p JOIN cuda_nodes() x ON x.fact_id = p.fact_id AND x.parent_id = p.node_id
          JOIN cuda_types() t ON t.key = 'e:' || x.fact_id || ':' || x.node_id AND t.ty = 'device'
          WHERE (p.type IN ('if_statement', 'elif_clause', 'while_statement', 'assert_statement',
                            'not_operator') AND x.sibling_index = 1)
             OR (p.type = 'conditional_expression' AND x.sibling_index = 2)
             OR (p.type = 'boolean_operator' AND x.sibling_index = 0)
             OR (p.type = 'interpolation' AND x.type NOT IN ('{', '}', 'format_specifier', 'type_conversion', '='))
        )
        SELECT * FROM converting UNION ALL SELECT * FROM methods UNION ALL SELECT * FROM conditions
    """
