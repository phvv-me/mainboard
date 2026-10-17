"""Rules over source modules (mcmr's ALL-MODU family)."""

from ...catalog import Rule


class ModuleLineCount(Rule, id="ALL-MODU0001"):
    """Modules longer than the physical-line ceiling."""

    maximum: int = 500

    query = """
        SELECT path, start_line, start_column,
          '`' || path || '` runs ' || physical_line_count
            || CASE WHEN physical_line_count = 1 THEN ' line' ELSE ' lines' END
            || ' holding ' || class_count
            || CASE WHEN class_count = 1 THEN ' class' ELSE ' classes' END
            || ' and ' || function_count
            || CASE WHEN function_count = 1 THEN ' function' ELSE ' functions' END AS message
        FROM module_facts WHERE physical_line_count > $maximum
    """


class ModuleMemberCount(Rule, id="ALL-MODU0002"):
    """Top-level classes and functions of a non-test module, as a focus proxy."""

    maximum: int = 12

    query = """
        SELECT *, CASE WHEN is_test THEN 0 ELSE (class_count + function_count)::DOUBLE END AS value
        FROM module_facts WHERE value > $maximum
    """
    message = "module member count is {value} for `{fact_id}`"


class ModuleInception(Rule, id="ALL-MODU0003"):
    """A module named after the package directory that already contains it."""

    query = r"""
        SELECT * FROM (SELECT *, regexp_replace(path, '^.*/', '', 'g') AS name FROM module_facts)
        WHERE contains(path, '/')
          AND CASE WHEN regexp_matches(name, '^\.+[^.]*$') THEN name
                   ELSE regexp_replace(name, '\.[^.]*$', '', 'g') END
              = regexp_replace(regexp_replace(path, '/[^/]*$', '', 'g'), '^.*/', '', 'g')
    """
    message = "`{fact_id}` satisfies module inception"


class NonAsciiSourcePath(Rule, id="ALL-MODU0004", applicability="review"):
    """Path components outside ASCII on the way to one source file."""

    query = r"""
        SELECT *, len(list_filter(str_split(path, '/'),
                                  lambda part: regexp_matches(part, '[^\x00-\x7f]'))) AS value
        FROM module_facts WHERE value > 0
    """
    message = "non ascii source path is {value} for `{fact_id}`"
    # The first non-ASCII component (the file or a directory above it) is renamed with its
    # accents folded (`café` to `cafe`), when that leaves plain ASCII and no import in the
    # repository spells a non-ASCII module or name, since nothing is then left to rewrite. Only
    # Python files; a directory moves once however many of its files are findings.
    fix = r"""
        SELECT finding_key, old AS path, 'rename' AS operation, new AS target
        FROM (
          SELECT finding_key, path, array_to_string(parts[1:first], '/') AS old,
                 array_to_string(list_transform(parts[1:first], lambda p: strip_accents(p)), '/')
                   AS new
          FROM (
            SELECT finding_key, path, string_split(path, '/') AS parts,
                   list_position(list_transform(string_split(path, '/'),
                                                lambda p: regexp_matches(p, '[^\x00-\x7f]')),
                                 true) AS first
            FROM finding)
        )
        WHERE (ends_with(path, '.py') OR ends_with(path, '.pyi'))
          AND NOT regexp_matches(new, '[^\x00-\x7f]')
          AND NOT EXISTS (SELECT 1 FROM import_binding_facts b
                          WHERE regexp_matches(b.module || b.imported_name || b.name,
                                               '[^\x00-\x7f]'))
    """


class ModuleClassCount(Rule, id="ALL-MODU0005"):
    """Classes declared at module scope, outside type stubs."""

    maximum: int = 1

    query = """
        SELECT *, CASE WHEN ends_with(path, '.pyi') THEN 0 ELSE class_count END AS value
        FROM module_facts WHERE value > $maximum
    """
    message = "module class count is {value} for `{fact_id}`"


class ModuleStatementCount(Rule, id="ALL-MODU0006"):
    """Statements throughout one module, nested ones included."""

    maximum: int = 200

    query = "SELECT *, statement_count AS value FROM module_facts WHERE value > $maximum"
    message = "module statement count is {value} for `{fact_id}`"
