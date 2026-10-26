"""Rules over where enums live (mcmr's PY-ENUM organization family)."""

from ....catalog import Rule


class SharedEnumsModuleCandidate(Rule, id="PY-ENUM0003"):
    """Reused enum classes that deserve one shared `enums.py`."""

    minimum_definitions: int = 3
    minimum_imported_definitions: int = 2
    minimum_cross_module_imports: int = 3
    preferred_modules: tuple[str, ...] = ("enums.py", "enums")

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, sum(s.enum_count) AS value
            FROM enum_scopes s JOIN enum_facts f USING (fact_id)
            WHERE s.enum_count >= $minimum_definitions
              AND s.reused_enum_count >= $minimum_imported_definitions
              AND s.cross_module_import_count >= $minimum_cross_module_imports
              AND NOT EXISTS (SELECT 1 FROM unnest($preferred_modules::VARCHAR[]) AS p(module)
                              WHERE ends_with(s.destination, p.module))
            GROUP BY ALL HAVING sum(s.enum_count) > 0
        """
    message = "shared enums module candidate is {value} for `{fact_id}`"


class SharedEnumFileShape(Rule, id="PY-ENUM0004"):
    """Shared `enums` implementation files that do not hold exactly one enum class."""

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM enum_files e JOIN enum_facts f USING (fact_id)
            WHERE NOT e.is_package_initializer
              AND (e.top_level_class_count <> 1 OR e.enum_class_count <> 1)
            GROUP BY ALL
        """
    message = "shared enum file shape is {value} for `{fact_id}`"
