"""mcmr's KernelLaunchFact, derived from sitting_duck's syntax trees.

tree-sitter's C++ grammar has no CUDA launch: it reads `kernel<<<grid, block, bytes, stream>>>(a)`
as `(kernel << <grid, block, bytes, stream>) >> (a)`, the configuration a template argument list
behind an identifier it had to invent (zero width). That shape is the launch: the kernel is the
left operand of the `<<`, the configuration the list's items, and the launch spans the outer `>>`.
"""

from ...catalog import Family


class KernelLaunch(Family):
    family = "KernelLaunchFact"
    verified = True
    relations = {
        "facts": r"""
            WITH cuda AS (
              SELECT n.* FROM ast_nodes n JOIN ast_files f USING (fact_id)
              WHERE f.language = 'cuda'
            ), configurations AS (
              SELECT c.fact_id, c.path, c.node_id, c.parent_id
              FROM cuda c
              JOIN cuda invented ON invented.fact_id = c.fact_id AND invented.parent_id = c.node_id
                AND invented.type = 'identifier' AND invented.start_line = invented.end_line
                AND invented.start_column = invented.end_column
              WHERE c.type = 'template_function'
            ), launches AS (
              SELECT g.fact_id, g.path, g.node_id AS configuration,
                     syntax_slice(s.content, k.start_line, k.start_column, k.end_line,
                                  k.end_column) AS kernel,
                     o.start_line, o.start_column, o.end_line, o.end_column, o.node_id AS launch
              FROM configurations g
              JOIN cuda i ON i.fact_id = g.fact_id AND i.node_id = g.parent_id
                AND i.type = 'binary_expression'
              JOIN cuda k ON k.fact_id = g.fact_id AND k.parent_id = i.node_id
                AND k.sibling_index = 0
              JOIN cuda o ON o.fact_id = g.fact_id AND o.node_id = i.parent_id
                AND o.type = 'binary_expression'
              JOIN source_files s ON s.path = g.path
            ), items AS (
              SELECT l.launch, l.fact_id,
                     list(syntax_slice(t.content, a.start_line, a.start_column, a.end_line,
                                       a.end_column) ORDER BY a.sibling_index) AS arguments
              FROM launches l
              JOIN cuda list ON list.fact_id = l.fact_id AND list.parent_id = l.configuration
                AND list.type = 'template_argument_list'
              JOIN cuda a ON a.fact_id = l.fact_id AND a.parent_id = list.node_id
                AND a.type NOT IN ('<', '>', ',')
              JOIN source_files t ON t.path = l.path
              GROUP BY l.launch, l.fact_id
            ), streamed AS (
              SELECT fact_id, bool_or(starts_with(name, 'cudaStream')
                                      OR ends_with(name, 'stream_ref')) AS streamed
              FROM cuda
              WHERE type IN ('identifier', 'type_identifier', 'qualified_identifier')
              GROUP BY fact_id
            ), enclosing AS (
              SELECT l.launch, l.fact_id, arg_max(f.name, f.depth) AS name
              FROM launches l
              JOIN cuda f ON f.fact_id = l.fact_id AND f.type = 'function_definition'
                AND l.launch BETWEEN f.node_id AND f.node_id + f.descendant_count
              GROUP BY l.launch, l.fact_id
            )
            SELECT row_number() OVER (ORDER BY l.path, l.start_line, l.start_column) - 1
                     AS fact_order,
                   'launch:' || l.path || ':' || kernel AS fact_id, l.path,
                   l.start_line, l.start_column, l.end_line, l.end_column,
                   'cuda' AS language, coalesce(i.arguments[2], '') AS block,
                   coalesce(i.arguments[3], '') AS dynamic_shared_bytes,
                   coalesce(e.name, '') AS enclosing_function, 0 AS evidence_length,
                   true AS evidence_present, coalesce(i.arguments[1], '') AS grid, kernel,
                   coalesce(i.arguments[4], '') AS stream,
                   coalesce(s.streamed, false) AS unit_uses_streams
            FROM launches l
            LEFT JOIN items i ON i.launch = l.launch AND i.fact_id = l.fact_id
            LEFT JOIN enclosing e ON e.launch = l.launch AND e.fact_id = l.fact_id
            LEFT JOIN streamed s ON s.fact_id = l.fact_id
        """,
    }
