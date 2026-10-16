"""Rules over the width of a type's surface (mcmr's ALL-CLAS shape family)."""

from ...catalog import Rule


class PublicMethodCount(Rule, id="ALL-CLAS0003"):
    """The type of each module declaring the most public methods, above the ceiling."""

    maximum: int = 10

    query = """
        SELECT * FROM (
          SELECT c.path, c.start_line, c.start_column, c.name, m.amount, m.method_names,
                 max(m.amount) OVER (PARTITION BY c.fact_id) AS most
          FROM class_classes c JOIN (
            SELECT class_id, count(*) AS amount,
                   string_agg(name, '`, `' ORDER BY ordinal) AS method_names
            FROM class_methods
            WHERE kind NOT IN ('field', 'property') AND visibility = 'public'
              AND NOT is_protocol_name
            GROUP BY class_id) m USING (class_id))
        WHERE amount > $maximum AND amount = most
    """
    message = "`{name}` declares {amount} public methods named `{method_names}`"


class DeclaredFieldCount(Rule, id="ALL-CLAS0004"):
    """The type of each module declaring the most fields, above the ceiling."""

    maximum: int = 64

    query = r"""
        WITH owners AS (
          SELECT fact_id, regexp_replace(qualname, '(?:\.|::)[^.:]+$', '', 'g') AS owner,
                 count(*) AS width, string_agg(qualname, '`, `' ORDER BY qualname) AS fields,
                 arg_min(span_path, ordinal) AS path,
                 arg_min(span_start_line, ordinal) AS start_line,
                 arg_min(span_start_column, ordinal) AS start_column
          FROM symbol_reach_declarations WHERE kind = 'attribute' GROUP BY ALL
        )
        SELECT * FROM (
          SELECT coalesce(c.span_path, o.path) AS path,
                 coalesce(c.span_start_line, o.start_line) AS start_line,
                 coalesce(c.span_start_column, o.start_column) AS start_column,
                 o.owner, o.width, o.fields, max(o.width) OVER (PARTITION BY o.fact_id) AS most
          FROM owners o LEFT JOIN symbol_reach_declarations c
            ON c.kind = 'class' AND c.fact_id = o.fact_id AND c.qualname = o.owner)
        WHERE width > $maximum AND width = most
    """
    message = "`{owner}` declares {width} fields named `{fields}`"


class AncestorCount(Rule, id="ALL-CLAS0005"):
    """Classes with more than `maximum` types above them in their inheritance chain."""

    maximum: int = 5

    query = r"""
        SELECT f.* FROM override_facts f
        LEFT JOIN override_values b
          ON b.fact_id = f.fact_id AND b.relation = 'base_names' AND b.ordinal = 0
        WHERE f.depth = 1 AND f.ancestor_names_length > $maximum
          AND coalesce(b.string_value, '') = regexp_replace(f.base, '^.*\.', '', 'g')
    """
    message = "ancestor count is {ancestor_names_length} for `{fact_id}`"
