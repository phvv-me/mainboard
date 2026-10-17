"""Rules over how TypeScript modules step around their types (mcmr's TS-TYPE family)."""

from ...catalog import Rule


class NonErasableConstruct(Rule, id="TS-TYPE0001"):
    """Constructs that generate JavaScript instead of disappearing with the types."""

    maximum: int = 0

    query = """
        SELECT f.path, v.line AS start_line, 0 AS start_column,
               CASE WHEN v.name = '' THEN f.path ELSE v.name END AS name,
               replace(v.kind, '_', ' ') AS kind
        FROM module_surface_erasable_violations v
        JOIN module_surface_facts f USING (fact_order, fact_id)
        WHERE f.erasable_violations_length > $maximum
    """
    message = (
        "`{name}` is a {kind}, which generates JavaScript rather than disappearing with the types"
    )


class EscapeHatchDensity(Rule, id="TS-TYPE0002"):
    """Modules where type-system escape hatches exceed `maximum` percent of their lines."""

    maximum: float = 2

    query = """
        SELECT f.path, v.line AS start_line, 0 AS start_column,
          f.escape_hatches_length
            || CASE f.escape_hatches_length WHEN 1 THEN ' place' ELSE ' places' END AS places,
          CASE v.kind WHEN 'assertion' THEN 'type assertion'
                      WHEN 'non_null' THEN 'non-null assertion' WHEN 'any' THEN '`any`'
                      WHEN 'ignore_comment' THEN 'suppression comment' ELSE v.kind END AS kind
        FROM module_surface_escape_hatches v
        JOIN module_surface_facts f USING (fact_order, fact_id)
        WHERE CASE WHEN f.physical_line_count = 0 THEN 0.0
                   ELSE f.escape_hatches_length / f.physical_line_count * 100.0 END > $maximum
    """
    message = "`{path}` states a {kind} here, one of {places} it steps around its own types"
