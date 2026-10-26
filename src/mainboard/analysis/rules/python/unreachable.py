"""Rules over statements no path reaches (the one dead-code check vulture adds to ruff)."""

from ...catalog import Example, Relation, Rule


class UnreachableBlocks(Relation):
    """For each block that leaves early (`return`, `raise`, `continue`, `break`), the statements
    after the first such exit, as one row at the first of them.

    An exit followed only by comments has none, and neither has one followed by a `yield` of its
    own function: `return` then `yield` is how a generator that yields nothing is written. A scope
    with a syntax error reports nothing, since the parser's recovery invents statements.

    `removable` says whether deleting the lines keeps the program's meaning. The code must start
    on a line of its own and declare no `global` or `nonlocal`. A name it binds that the rest of
    the enclosing function (or module) mentions must also be bound there, since a name bound
    anywhere in a function is local to all of it: deleting its only binding would turn those
    mentions into reads of a global."""

    query = """
    WITH exits AS (
      SELECT fact_id, path, parent_id, min(sibling_index) AS position
      FROM ast_nodes
      WHERE type IN ('return_statement', 'raise_statement', 'continue_statement',
                     'break_statement')
      GROUP BY ALL
    ), exited AS (
      SELECT e.fact_id, e.path, e.position, n.parent_id, n.node_id AS exit_node,
             n.end_line AS exit_end, regexp_extract(n.type, '^[a-z]+') AS keyword
      FROM exits e
      JOIN ast_nodes n ON n.fact_id = e.fact_id AND n.parent_id = e.parent_id
       AND n.sibling_index = e.position
    ), dead AS (
      SELECT x.fact_id, x.path, x.exit_node, x.exit_end, x.keyword, count(*) AS statements,
             min(d.node_id) AS first_node, max(d.node_id + d.descendant_count) AS last_node,
             arg_min(d.start_line, d.sibling_index) AS start_line,
             arg_min(d.start_column, d.sibling_index) AS start_column,
             max(d.end_line) AS end_line
      FROM exited x
      JOIN ast_nodes d ON d.fact_id = x.fact_id AND d.parent_id = x.parent_id
       AND d.sibling_index > x.position
      WHERE d.type <> 'comment'
      GROUP BY x.fact_id, x.path, x.exit_node, x.exit_end, x.keyword
    ), bounded AS (
      SELECT d.*, coalesce(max(f.node_id), 0) AS scope_first,
             coalesce(arg_max(f.node_id + f.descendant_count, f.node_id), 9223372036854775807)
               AS scope_last
      FROM dead d
      LEFT JOIN ast_nodes f ON f.fact_id = d.fact_id AND f.type = 'function_definition'
       AND f.node_id < d.exit_node AND d.exit_node <= f.node_id + f.descendant_count
      GROUP BY ALL
    ), binders AS (
      SELECT i.fact_id, i.node_id, i.name
      FROM ast_nodes i
      JOIN ast_nodes p ON p.fact_id = i.fact_id AND p.node_id = i.parent_id
      WHERE i.fact_id IN (SELECT fact_id FROM dead) AND i.type = 'identifier' AND (
              p.type IN ('pattern_list', 'tuple_pattern', 'list_pattern', 'as_pattern_target',
                         'global_statement', 'nonlocal_statement', 'delete_statement',
                         'dotted_name', 'aliased_import', 'parameters', 'list_splat_pattern',
                         'dictionary_splat_pattern')
           OR (p.type IN ('assignment', 'augmented_assignment', 'named_expression',
                          'default_parameter', 'typed_parameter', 'typed_default_parameter')
               AND i.sibling_index = 0)
           OR (p.type = 'for_statement' AND i.sibling_index = 1)
           OR (p.type = 'tuple' AND EXISTS (
                 SELECT 1 FROM ast_nodes g WHERE g.fact_id = p.fact_id AND g.node_id = p.parent_id
                   AND g.type = 'as_pattern_target')))
      UNION ALL
      SELECT fact_id, node_id, name FROM ast_nodes
      WHERE fact_id IN (SELECT fact_id FROM dead)
        AND type IN ('function_definition', 'class_definition')
    ), unsafe AS (
      SELECT DISTINCT s.fact_id, s.exit_node
      FROM bounded s
      JOIN binders b ON b.fact_id = s.fact_id AND b.node_id BETWEEN s.first_node AND s.last_node
      WHERE EXISTS (
              SELECT 1 FROM ast_nodes o
              WHERE o.fact_id = s.fact_id AND o.type = 'identifier' AND o.name = b.name
                AND o.node_id BETWEEN s.scope_first AND s.scope_last
                AND o.node_id NOT BETWEEN s.first_node AND s.last_node)
        AND NOT EXISTS (
              SELECT 1 FROM binders k
              WHERE k.fact_id = s.fact_id AND k.name = b.name
                AND k.node_id BETWEEN s.scope_first AND s.scope_last
                AND k.node_id NOT BETWEEN s.first_node AND s.last_node
                AND NOT EXISTS (
                  SELECT 1 FROM ast_nodes f
                  WHERE f.fact_id = s.fact_id AND f.type IN ('function_definition', 'lambda')
                    AND f.node_id > s.scope_first AND f.node_id < k.node_id
                    AND k.node_id <= f.node_id + f.descendant_count))
    )
    SELECT s.path, s.start_line, s.start_column, s.exit_end + 1 AS from_line, s.end_line,
           s.statements, s.keyword,
           CASE WHEN s.statements = 1 THEN 'statement' ELSE 'statements' END AS noun,
           s.start_line > s.exit_end
             AND NOT EXISTS (
               SELECT 1 FROM ast_nodes g
               WHERE g.fact_id = s.fact_id AND g.node_id BETWEEN s.first_node AND s.last_node
                 AND g.type IN ('global_statement', 'nonlocal_statement'))
             AND NOT EXISTS (
               SELECT 1 FROM unsafe u WHERE u.fact_id = s.fact_id AND u.exit_node = s.exit_node)
             AS removable
    FROM bounded s
    WHERE NOT EXISTS (
      SELECT 1 FROM ast_nodes e
      WHERE e.fact_id = s.fact_id AND e.type = 'ERROR'
        AND e.node_id BETWEEN s.scope_first AND s.scope_last)
      AND NOT EXISTS (
      SELECT 1 FROM ast_nodes y
      WHERE y.fact_id = s.fact_id AND y.type = 'yield'
        AND y.node_id BETWEEN s.first_node AND s.last_node
        AND NOT EXISTS (
          SELECT 1 FROM ast_nodes f
          WHERE f.fact_id = s.fact_id AND f.type IN ('function_definition', 'lambda')
            AND f.node_id BETWEEN s.first_node AND y.node_id - 1
            AND y.node_id <= f.node_id + f.descendant_count))
"""


class UnreachableStatement(Rule, id="PY-DEAD0002", applicability="review"):
    """Statements after a `return`, `raise`, `continue` or `break` in the same block.

    The fix is for review: code parked after a `raise` on purpose (a stub, a sketch kept beside
    its refusal) is deleted only by an explicit `fix --unsafe`."""

    query = "SELECT * FROM unreachable_blocks()"
    message = "{statements} {noun} after `{keyword}` can never run"
    examples = (
        Example(
            bad=(
                """def run(value):
    if value:
        return 1
        print("never")
        value += 1
    return 0
"""
            ),
            good="def run(value):\n    if value:\n        return 1\n    return 0\n",
        ),
        Example(
            bad=(
                """def first(items):
    for item in items:
        if item:
            break
            # never reached
            item.close()
    return None
"""
            ),
            good=(
                """def first(items):
    for item in items:
        if item:
            break
    return None
"""
            ),
        ),
        Example(
            bad="def read():\n    return y\n    y = 1\n",
            good="",
        ),
    )
    # The lines after the exit up to the end of the last unreachable statement (comments and blank
    # lines between them included), unless deleting them could change what the function means
    # (`removable` in the relation).
    fix = """
        SELECT finding_key, path, from_line AS start_line, 0 AS start_column,
               end_line + 1 AS end_line, 0 AS end_column, '' AS text
        FROM finding WHERE removable
    """
