"""Rules over what TypeScript modules re-export and import (mcmr's TS-MODU family)."""

from ...catalog import Rule


class StarReexportSurface(Rule, id="TS-MODU0001"):
    """Modules that re-export everything another module happens to export."""

    query = """
        SELECT f.path, f.start_line, f.start_column, v.string_value
        FROM module_surface_values v JOIN module_surface_facts f USING (fact_order, fact_id)
        WHERE v.relation = 'star_reexports'
    """
    message = (
        "`{path}` re-exports everything `{string_value}` happens to export, so a helper added "
        "there joins this module's contract unreviewed"
    )


class RelativeImportDepth(Rule, id="TS-MODU0002"):
    """Modules whose deepest relative import climbs out of more directories than `maximum`."""

    maximum: int = 1

    query = """
        SELECT *, '`' || path || '` imports through `' || deepest_relative_specifier
          || '`, which climbs ' || deepest_relative_import
          || CASE deepest_relative_import WHEN 1 THEN ' directory' ELSE ' directories' END
          || ' out of its own' AS message
        FROM module_surface_facts WHERE deepest_relative_import > $maximum
    """
