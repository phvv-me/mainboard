"""Rules over how external packages are used (mcmr's ALL-DEPE transformations family)."""

from ...catalog import Rule


class RepeatedExternalUnaryTransformation(Rule, id="ALL-DEPE0003"):
    """External callables applied to one argument at many sites, which a boundary should own."""

    minimum_repetitions: int = 3
    minimum_files: int = 2
    ignored_callables: tuple[str, ...] = ()
    first_party_modules: tuple[str, ...] = ()
    transformation_names: tuple[str, ...] = (
        "camelize",
        "canonicalize",
        "convert",
        "decode",
        "deserialize",
        "encode",
        "normalize",
        "parse",
        "sanitize",
        "serialize",
        "slugify",
        "underscore",
    )

    query = r"""
        WITH unary AS (
          SELECT call_id FROM call_expressions WHERE root_relation = 'argument' AND depth = 0
          GROUP BY call_id HAVING count(*) = 1
        ), eligible AS (
          SELECT c.qualified_name, c.path, f.fact_order, c.ordinal,
                 c.node_path, c.node_start_line, c.node_start_column,
                 '`' || c.node_path || ':' || c.node_start_line
                 || CASE WHEN c.node_end_line > c.node_start_line THEN '-' || c.node_end_line
                         ELSE '' END || '`' AS location
          FROM call_calls c JOIN call_facts f USING (fact_id)
          SEMI JOIN unary USING (call_id) ANTI JOIN call_keywords USING (call_id)
          WHERE c.is_external AND NOT c.is_standard_library AND NOT c.is_first_party
            AND NOT c.is_constructor AND NOT c.is_shadowed AND NOT c.has_ambiguous_alias
            AND NOT c.is_decorator_factory AND NOT c.has_starred_arguments
            AND list_contains($transformation_names::VARCHAR[],
                              regexp_replace(c.qualified_name, '^.*\.', '', 'g'))
            AND NOT list_contains($ignored_callables::VARCHAR[], c.qualified_name)
            AND len(list_filter($first_party_modules::VARCHAR[], lambda m: c.qualified_name = m
                                OR starts_with(c.qualified_name, m || '.'))) = 0
        )
        SELECT arg_min(node_path, (fact_order, ordinal)) AS path,
               arg_min(node_start_line, (fact_order, ordinal)) AS start_line,
               arg_min(node_start_column, (fact_order, ordinal)) AS start_column,
               qualified_name, count(*) AS group_count, count(DISTINCT path) AS file_count,
               string_agg(location, ', ' ORDER BY fact_order, ordinal) AS locations
        FROM eligible GROUP BY qualified_name
        HAVING count(*) >= $minimum_repetitions AND count(DISTINCT path) >= $minimum_files
    """
    message = (
        "external boundary candidate `{qualified_name}` repeats {group_count} times across "
        "{file_count} files at {locations}"
    )
