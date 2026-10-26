"""Rules over the control flow inside one callable (mcmr's ALL-FUNC flow rules)."""

from ....catalog import Relation, Rule


class FlowControls(Relation):
    """The control constructs each function owns (`conditional`, `loop`, `switch`, `catch`,
    `jump`, and `alternative` for an `elif` or `else`, whose own `if` is no construct), each with
    the number of enclosing constructs of its own function."""

    stored = True
    query = """
    WITH raw AS (
      SELECT n.file_id, n.node_id, n.descendant_count AS size, n.function_id AS def_id,
             CASE WHEN n.type = 'if_statement' AND p.type <> 'else_clause' THEN 'conditional'
                  WHEN n.type IN ('for_statement', 'while_statement', 'for_in_statement',
                                  'do_statement') THEN 'loop'
                  WHEN n.type IN ('match_statement', 'switch_statement') THEN 'switch'
                  WHEN n.type = 'try_statement' THEN 'catch'
                  WHEN n.type IN ('break_statement', 'continue_statement') THEN 'jump'
                  WHEN p.type = 'if_statement' THEN 'alternative' END AS kind
      FROM core_node n
      JOIN core_node p ON p.file_id = n.file_id AND p.node_id = n.parent_id
      WHERE n.type IN ('if_statement', 'for_statement', 'while_statement', 'for_in_statement',
                       'do_statement', 'match_statement', 'switch_statement', 'try_statement',
                       'break_statement', 'continue_statement', 'elif_clause', 'else_clause')
        AND n.function_id IS NOT NULL AND coalesce(n.class_id, -1) < n.function_id
    ), kept AS (
      SELECT * FROM raw WHERE kind IS NOT NULL
    )
    SELECT c.file_id, c.def_id, c.node_id, c.kind,
           count(m.node_id)::INT - (c.kind = 'alternative')::INT AS nesting_depth
    FROM kept c
    LEFT JOIN kept m ON m.file_id = c.file_id AND m.def_id = c.def_id
      AND m.kind IN ('conditional', 'loop', 'switch', 'catch')
      AND m.node_id < c.node_id AND c.node_id <= m.node_id + m.size
    GROUP BY c.file_id, c.def_id, c.node_id, c.kind
"""


class FlowMeasured(Relation):
    """The `flow_measured()` relation: each function with the number of control
    constructs it owns (`total`) and the nesting those that nest add up to (`nested`)."""

    query = """
    SELECT d.path, d.start_line, d.start_column,
           'function:' || d.path || ':' || d.start_line || ':' || d.start_column || ':' || d.name
             AS fact_id,
           coalesce(c.total, 0) AS total, coalesce(c.nested, 0) AS nested
    FROM core_def d LEFT JOIN (
      SELECT file_id, def_id, count(*) AS total, sum(nesting_depth) FILTER (WHERE nests) AS nested
      FROM (SELECT *, kind IN ('conditional', 'loop', 'switch', 'catch') AS nests
            FROM flow_controls())
      GROUP BY file_id, def_id) c USING (file_id, def_id)
    WHERE d.kind = 'function' AND d.type <> 'lambda'
"""


class CognitiveComplexity(Rule, id="ALL-FUNC0008"):
    """Control structures in one callable, each nesting level counted against it."""

    maximum: int = 8
    nesting_penalty: int = 1

    query = (
        "SELECT *, total + nested * $nesting_penalty AS value FROM flow_measured() "
        "WHERE value > $maximum"
    )
    message = "cognitive complexity is {value} for `{fact_id}`"
