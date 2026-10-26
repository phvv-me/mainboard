"""Rules over which failures a program names (mcmr's ALL-ERRO taxonomy rules)."""

from ....catalog import Rule


class VanillaErrorType(Rule, id="ALL-ERRO0003"):
    """Failures raised as the language's base error rather than as a named one."""

    base_errors: tuple[str, ...] = (
        "Exception",
        "BaseException",
        "Error",
        "Throwable",
        "RuntimeException",
        "std::exception",
    )

    query = """
        SELECT * FROM (
          SELECT f.path, r.start_line, r.start_column,
                 str_split(arg_min(e.name, e.node_id), '.')[-1] AS thrown
          FROM core_node r
          JOIN core_file f USING (file_id)
          JOIN core_node e ON e.file_id = r.file_id AND e.node_id > r.node_id
           AND e.node_id <= r.node_id + r.descendant_count AND e.type IN ('call', 'identifier')
          WHERE r.semantic_type = 'ERROR_THROW' AND r.children_count > 0 AND f.parses
          GROUP BY r.file_id, r.node_id, f.path, r.start_line, r.start_column)
        WHERE list_contains($base_errors::VARCHAR[], thrown)
    """
    message = "raise constructs the base error `{thrown}`"
