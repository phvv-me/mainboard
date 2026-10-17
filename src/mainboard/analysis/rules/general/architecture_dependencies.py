"""Rules over how modules depend on each other (mcmr's ALL-ARCH dependencies family)."""

from ...catalog import Example, Rule
from ...facts import Setting

_PACKAGE_MODULE = r"(__init__\.py|mod\.rs)$"
_TEST_PATH = r"(^|/)(tests?|specs?)(/|$)|(^|[._-])(test|spec)\.[^.]+$"


class DependencyOnALessStableModule(Rule, id="ALL-ARCH0003"):
    """Package imports pointing at a package less stable than the importer."""

    tolerance: float = 0.0

    query = r"""
        WITH packages AS (
          SELECT fact_order, fact_id, module AS name, path, start_line, start_column,
                 CASE WHEN regexp_matches(path, $package_module) THEN dotted
                      ELSE regexp_replace(dotted, '\.[^.]+$', '', 'g') END AS package
          FROM (SELECT *, replace(replace(module, '::', '.'), '/', '.') AS dotted
                FROM module_coupling_facts)
        ), edges AS (
          SELECT i.package AS module, d.package AS dependency_module,
                 i.path, i.start_line, i.start_column
          FROM module_coupling_dependencies e
          JOIN packages i ON i.fact_order = e.fact_order AND i.fact_id = e.fact_id
          JOIN packages d ON d.name = e.module
          WHERE NOT regexp_matches(i.path, $package_module)
            AND NOT (regexp_matches(i.path, $test_path) AND NOT regexp_matches(d.path, $test_path))
            AND i.package <> d.package
          QUALIFY row_number() OVER (PARTITION BY i.package, d.package
                                     ORDER BY e.fact_order, e.ordinal) = 1
        ), instability AS (
          SELECT module, afferent, efferent / (afferent + efferent) AS value FROM (
            SELECT module, count(*) FILTER (WHERE outgoing) AS efferent,
                   count(*) FILTER (WHERE NOT outgoing) AS afferent
            FROM (SELECT module, true AS outgoing FROM edges
                  UNION ALL SELECT dependency_module, false FROM edges)
            GROUP BY module)
        )
        SELECT e.path, e.start_line, e.start_column,
          '`' || e.module || '` sits at ' || printf('%.3g', s.value * 100.0)
          || ' percent instability and imports `' || e.dependency_module || '` at '
          || printf('%.3g', d.value * 100.0) || ' percent instability, so every change to the '
          || 'second one reaches the ' || s.afferent
          || CASE WHEN s.afferent = 1 THEN ' package' ELSE ' packages' END
          || ' that depend on the first' AS message
        FROM edges e JOIN instability s ON s.module = e.module
        JOIN instability d ON d.module = e.dependency_module
        WHERE d.value > s.value + $tolerance
    """

    def parameters(self) -> dict[str, Setting]:
        package_module = _PACKAGE_MODULE
        test_path = _TEST_PATH
        return super().parameters() | {"package_module": package_module, "test_path": test_path}


class AbstractionNothingDependsOn(Rule, id="ALL-ARCH0005", applicability="review"):
    """Modules written as a contract that no other module imports. Package markers and entry
    points are never imported by name and are not dead (`__init__`, `__main__`, `conftest`, `mod`,
    `main`, `lib`, `index`), so they are not reported."""

    minimum_abstractness: float = 0.5

    query = r"""
        SELECT * FROM module_coupling_facts
        WHERE afferent_count = 0 AND CASE WHEN declaration_count > 0
          THEN abstract_declaration_count / declaration_count ELSE 0.0 END >= $minimum_abstractness
          AND NOT regexp_matches(
            path, '(^|/)(__init__|__main__|conftest|mod|main|lib|index)\.[^/.]+$')
    """
    message = "`{fact_id}` satisfies abstraction nothing depends on"
    # The file goes. Package markers and entry points are never imported by name yet are not
    # dead, so they stay (`__init__`, `__main__`, `conftest`, `mod`, `main`, `lib`, `index`).
    fix = r"""
        SELECT finding_key, path, 'remove' AS operation FROM finding
        WHERE NOT regexp_matches(
          path, '(^|/)(__init__|__main__|conftest|mod|main|lib|index)\.[^/.]+$')
    """
    examples = (
        Example(
            bad=(
                "from typing import Protocol\n\n\nclass Reader(Protocol):\n"
                "    def read(self) -> str: ...\n"
            ),
            good="",
        ),
    )


class ImportCycles(Rule, id="ALL-ARCH0002"):
    """Groups of modules that import each other, directly or through a chain. A package
    initializer's `from . import sub` names its own submodule, not itself."""

    query = r"""
        WITH cycles AS (
          SELECT * FROM dependency_component_import_edges
          WHERE source_component = target_component
            AND NOT (source = target AND regexp_matches(path, '(^|/)__init__\.pyi?$'))
        ), members AS (
          SELECT fact_id, source_component, count(*) AS size,
                 string_agg(member, '`, `' ORDER BY member) AS names
          FROM (SELECT fact_id, source_component, source AS member FROM cycles
                UNION SELECT fact_id, source_component, target FROM cycles)
          GROUP BY fact_id, source_component
        ), first AS (
          SELECT fact_id, source_component, count(*) AS arrows, arg_min(path, ordinal) AS path,
                 arg_min(line, ordinal) AS line, arg_min(source, ordinal) AS source,
                 arg_min(target, ordinal) AS target
          FROM cycles GROUP BY fact_id, source_component
        )
        SELECT path, line AS start_line, 0 AS start_column,
          size || CASE WHEN size = 1 THEN ' module' ELSE ' modules' END
          || ' import each other in one cycle, which are `' || names || '`, and `' || source
          || '` importing `' || target || '` is one of the ' || arrows
          || CASE WHEN arrows = 1 THEN ' arrow' ELSE ' arrows' END || ' closing it' AS message
        FROM first JOIN members USING (fact_id, source_component)
    """
