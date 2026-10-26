"""Rules over how local names read (mcmr's ALL-NAMI family)."""

from ....catalog import Rule


class UninformativeLocalName(Rule, id="ALL-NAMI0001"):
    """Local bindings in callables whose name is too short to say what they hold."""

    minimum_length: int = 3
    conventional_names: tuple[str, ...] = ("i", "j", "k", "n", "x", "y", "z", "_")

    query = """
        WITH bound AS (
          SELECT a.file_id, a.function_id, a.node_id, a.start_line, a.start_column,
                 string_split(t.name, '.')[-1] AS name
          FROM core_node a
          JOIN core_node t ON t.file_id = a.file_id AND t.parent_id = a.node_id
                          AND t.sibling_index = 0 AND t.type IN ('identifier', 'attribute')
          JOIN core_node p ON p.file_id = a.file_id AND p.node_id = a.parent_id
          WHERE a.type IN ('assignment', 'augmented_assignment') AND p.type <> 'assignment'
            AND a.owner_id = a.function_id
          UNION ALL
          SELECT d.file_id, d.function_id, t.node_id, t.start_line, t.start_column, t.name
          FROM core_node d
          JOIN core_node t ON t.file_id = d.file_id AND t.parent_id = d.node_id
            AND t.type = 'identifier'
          LEFT JOIN core_node k ON k.file_id = d.file_id AND k.parent_id = d.node_id
            AND k.sibling_index = t.sibling_index - 1
          WHERE d.owner_id = d.function_id
            AND (d.type = 'variable_declarator' AND t.sibling_index = 0
                 OR d.type = 'for_in_statement' AND k.type IN ('const', 'let', 'var'))
        )
        SELECT arg_min(f.path, b.node_id) AS path, arg_min(b.start_line, b.node_id) AS start_line,
               arg_min(b.start_column, b.node_id) AS start_column, b.name,
               any_value(d.qualname) AS qualname
        FROM bound b
        JOIN core_def d ON d.file_id = b.file_id AND d.def_id = b.function_id
        JOIN core_file f ON f.file_id = b.file_id
        WHERE d.parses AND b.name <> '' AND length(b.name) < $minimum_length
          AND NOT list_contains($conventional_names::VARCHAR[], b.name)
        GROUP BY b.file_id, b.function_id, b.name
    """
    message = (
        "`{qualname}` binds `{name}`, which is shorter than the "
        f"{minimum_length} characters a name needs to say what it holds"
    )
