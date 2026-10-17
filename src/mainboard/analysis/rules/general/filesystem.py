"""Rules over the directory tree (mcmr's ALL-FILE family)."""

from ...catalog import Rule


class EmptyDirectories(Rule, id="ALL-FILE0001", applicability="safe"):
    """Directories left empty after the repository's Git ignores."""

    query = "SELECT * FROM directory_facts WHERE entry_count = 0"
    message = "`{fact_id}` satisfies empty directories"
    fix = "SELECT path, 'remove' AS operation FROM finding"


class PackageDepth(Rule, id="ALL-FILE0002"):
    """Directory levels between a directory and the source root above it."""

    maximum: int = 5
    framework_roots: tuple[str, ...] = ()

    query = """
        SELECT *, source_depth AS value FROM directory_facts
        WHERE source_depth > $maximum AND NOT EXISTS (
          SELECT 1 FROM unnest($framework_roots::VARCHAR[]) AS r(root)
          WHERE path = root OR starts_with(path, rtrim(root, '/') || '/'))
    """
    message = "package depth is {value} for `{fact_id}`"


class DirectoryModuleCount(Rule, id="ALL-FILE0003"):
    """Source files held directly by one directory, definition catalogs excepted."""

    maximum: int = 3
    allow_definition_catalogs: bool = True

    query = """
        SELECT * FROM (
          SELECT *, CASE WHEN $allow_definition_catalogs AND is_definition_catalog THEN 0
                         ELSE direct_module_count END AS value
          FROM directory_facts)
        WHERE value > $maximum
    """
    message = "directory module count is {value} for `{fact_id}`"


class DirectoryPathway(Rule, id="ALL-FILE0004"):
    """Directories that only lead to one child directory."""

    query = """
        SELECT * FROM directory_facts
        WHERE direct_directory_count = 1 AND direct_file_count = 0 AND source_depth > 1
          AND NOT (only_child_directory IN ('deterministic', 'contextual', 'external')
                   AND contains('/' || path || '/', '/rules/'))
    """
    message = "`{fact_id}` satisfies directory pathway"
