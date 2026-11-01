"""Rules over source modules (mcmr's ALL-MODU family)."""

from ....catalog import Example, Part, Rule
from ....constants import VOCABULARY_PATH

_PROJECT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}


class ModuleLineCount(Part):
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


class ModuleMemberCount(Part):
    """Top-level classes and functions of a non-test module, as a focus proxy. A shared
    `constants.py`, `enums.py` or `exceptions.py` has no focus to protect: it holds the
    definitions its folder shares."""

    maximum: int = 12

    query = f"""
        SELECT *, CASE WHEN is_test OR regexp_matches(path, '{VOCABULARY_PATH}') THEN 0
                       ELSE (class_count + function_count)::DOUBLE END AS value
        FROM module_facts WHERE value > $maximum
    """
    message = "module member count is {value} for `{fact_id}`"


class ModuleInception(Rule, id="ALL-MODU0003", applicability="review"):
    """A module named after the package directory that already contains it.

    The fix renames a Python module to `core.py`, its stub with it, every import of it following,
    and leaves the module as it is when `core.py` exists or a file that is no Python names it."""

    query = r"""
        SELECT * FROM (SELECT *, regexp_replace(path, '^.*/', '', 'g') AS name FROM module_facts)
        WHERE contains(path, '/')
          AND CASE WHEN regexp_matches(name, '^\.+[^.]*$') THEN name
                   ELSE regexp_replace(name, '\.[^.]*$', '', 'g') END
              = regexp_replace(regexp_replace(path, '/[^/]*$', '', 'g'), '^.*/', '', 'g')
    """
    message = "`{fact_id}` is named after the directory that holds it"
    change = "renamed `{path}` to `core.py`, every import of it following"
    fix = r"""
        SELECT f.finding_key, 'move-modules' AS operation,
               [{'path': f.path, 'target': regexp_replace(f.path, '[^/]+\.py$', 'core.py')}]
                 || coalesce((SELECT list({'path': s.path,
                                           'target': regexp_replace(s.path, '[^/]+\.pyi$', 'core.pyi')})
                              FROM core_file s WHERE s.path = f.path || 'i'), []) AS moves,
               r.files,
               coalesce((SELECT list(h.path) FROM core_file h
                         WHERE h.language = 'python'
                           AND regexp_replace(h.path, '/[^/]+$', '')
                               = regexp_replace(f.path, '/[^/]+$', '')), []) AS held,
               coalesce((SELECT list(s.path) FROM source_files s
                         WHERE NOT regexp_matches(s.path, '\.pyi?$')
                           AND (contains(s.content, c.module) OR contains(s.content, f.path))),
                        []) AS mentions
        FROM finding f
        JOIN core_file c ON c.path = f.path AND c.language = 'python',
             module_readers([c.module]) r
        WHERE ends_with(f.path, '.py')
    """
    examples = (
        Example(
            bad="from .rules import X\n",
            good="from .core import X\n",
            path="src/pkg/rules/__init__.py",
            context={**_PROJECT, "src/pkg/rules/rules.py": "X = 1\n"},
        ),
        Example(
            bad="import pkg.rules.rules\n\nY = pkg.rules.rules.X\n",
            good="import pkg.rules.core\n\nY = pkg.rules.core.X\n",
            path="src/pkg/consumer.py",
            context={
                **_PROJECT,
                "src/pkg/rules/__init__.py": "",
                "src/pkg/rules/rules.py": "X = 1\n",
            },
        ),
        Example(
            bad="from .rules import X\n",
            good="from .core import X\n",
            path="src/pkg/rules/__init__.py",
            context={
                **_PROJECT,
                "src/pkg/rules/rules.py": "X = 1\n",
                "src/pkg/rules/rules.pyi": "X: int\n",
            },
        ),
        Example(
            bad="from .rules import X\n",
            good="from .rules import X\n",
            path="src/pkg/rules/__init__.py",
            context={
                **_PROJECT,
                "src/pkg/rules/core.py": "Y = 2\n",
                "src/pkg/rules/rules.py": "X = 1\n",
            },
        ),
        Example(
            bad="from .rules import X\n",
            good="from .rules import X\n",
            path="src/pkg/rules/__init__.py",
            context={
                **_PROJECT,
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n# pkg.rules.rules\n',
                "src/pkg/rules/rules.py": "X = 1\n",
            },
        ),
    )


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


class ModuleStatementCount(Part):
    """Statements throughout one module, nested ones included."""

    maximum: int = 200

    query = "SELECT *, statement_count AS value FROM module_facts WHERE value > $maximum"
    message = "module statement count is {value} for `{fact_id}`"
