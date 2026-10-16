"""Rules over cached callables (mcmr's PY-CACH family)."""

from ...catalog import Rule


class InstanceIndependentCachedProperty(Rule, id="PY-CACH0001"):
    """Cached properties computing something independent of their instance."""

    query = """
            SELECT d.path, d.start_line, d.start_column,
                   'function:' || d.path || ':' || d.start_line || ':' || d.start_column || ':'
                     || d.name AS fact_id
            FROM core_def d
            LEFT JOIN core_param r ON r.file_id = d.file_id AND r.def_id = d.def_id
              AND r.ordinal = 0 AND NOT r.keyword_only AND r.name IN ('self', 'cls')
              AND NOT EXISTS (SELECT 1 FROM core_decor x
                              WHERE x.file_id = d.file_id AND x.def_id = d.def_id
                                AND regexp_extract(trim(split_part(x.name, '(', 1)), '[^.]*$')
                                  = 'staticmethod')
            WHERE d.type = 'function_definition' AND d.language = 'python'
              AND d.class_id > coalesce(d.function_id, -1)
              AND EXISTS (SELECT 1 FROM core_decor x
                          WHERE x.file_id = d.file_id AND x.def_id = d.def_id
                            AND regexp_extract(trim(split_part(x.name, '(', 1)), '[^.]*$')
                              = 'cached_property')
              AND d.statements - (d.docstring IS NOT NULL)::INTEGER = 1
              AND NOT coalesce(EXISTS (
                SELECT 1 FROM core_node i
                WHERE i.file_id = d.file_id AND i.type = 'identifier' AND i.name = r.name
                  AND i.node_id > d.body_id AND i.node_id <= d.def_id + d.size
                  AND NOT EXISTS (
                    SELECT 1 FROM core_node a
                    JOIN core_node l ON l.file_id = a.file_id AND l.parent_id = a.node_id
                      AND l.sibling_index = 0
                    WHERE a.file_id = i.file_id
                      AND a.type IN ('assignment', 'augmented_assignment')
                      AND i.node_id BETWEEN l.node_id AND l.node_id + l.descendant_count)),
                false)
        """
    message = "`{fact_id}` satisfies instance independent cached property"
