"""Rules over module constants (mcmr's PY-CONS family)."""

from ....catalog import Example, Relation, Rule


class PlacedSymbols(Relation):
    """The `placed_symbols()` relation."""

    query = """
    SELECT s.* REPLACE (
      CASE WHEN s.reference_id IS NULL THEN f.path
           ELSE s.reference_declaration_span_path END AS path,
      CASE WHEN s.reference_id IS NULL THEN f.start_line
           ELSE s.reference_declaration_span_start_line END AS start_line,
      CASE WHEN s.reference_id IS NULL THEN f.start_column
           ELSE s.reference_declaration_span_start_column END AS start_column)
    FROM symbol_symbols s JOIN symbol_facts f USING (fact_order, fact_id)
"""


class PublicModuleConstant(Rule, id="PY-CONS0001", applicability="review"):
    """Project module constants exposed without a leading underscore outside a `constants.py` or a
    `constants/` package, the modules whose constants are public for the others to share."""

    preferred_module: str = "constants"

    query = """
            SELECT * FROM placed_symbols()
            WHERE scope = 'module' AND is_constant_assignment
              AND upper(name) = name AND lower(name) <> name AND NOT starts_with(name, '_')
              AND NOT regexp_matches(path, '(^|/)' || $preferred_module || '(\\.py$|/)')
        """
    message = (
        "`{name}` is a public module constant: keep it private to its file, or move it to a "
        "constants.py if other modules share it"
    )
    # The constant becomes `_NAME` at its declaration and every reference its module holds, when
    # the kernel saw all of them and nothing outside the module can reach the name: no import
    # of that name, no attribute access with it, no star import of the module, no `__all__`
    # entry, no `_NAME` already in the module. Names are matched project-wide (an upper bound on
    # importers), and a constant that other modules use is PY-CONS0002's to move, not renamed.
    # Review: strings and dynamic lookups (`getattr`, templates) cannot be seen.
    fix = r"""
        WITH renamable AS (
          SELECT * FROM finding f
          WHERE f.reference_id IS NOT NULL AND f.reference_are_references_complete
            AND NOT EXISTS (SELECT 1 FROM import_binding_facts i
                            WHERE i.imported_name = f.name OR i.name = f.name
                               OR (i.is_wildcard AND ends_with(
                                     regexp_replace(
                                       regexp_replace(f.path, '(/__init__)?\.py$', ''),
                                       '/', '.', 'g'),
                                     ltrim(i.module, '.'))))
            AND NOT EXISTS (SELECT 1 FROM core_ref a
                            WHERE a.parent_type = 'attribute' AND a.place = 1
                              AND a.name = f.name)
            AND NOT EXISTS (SELECT 1 FROM module_all_declarations d
                            WHERE d.span_path = f.path
                              AND (contains(d.text, '"' || f.name || '"')
                                   OR contains(d.text, '''' || f.name || '''')))
            AND NOT EXISTS (SELECT 1 FROM symbol_symbols t
                            WHERE t.reference_declaration_span_path = f.path
                              AND t.scope = 'module'
                              AND (t.name = '_' || f.name
                                   OR (t.name = f.name AND t.record_id <> f.record_id)))
        )
        SELECT finding_key, reference_declaration_span_path AS path,
               reference_declaration_span_start_line AS start_line,
               reference_declaration_span_start_column AS start_column,
               reference_declaration_span_end_line AS end_line,
               reference_declaration_span_end_column AS end_column, '_' || name AS text
        FROM renamable
        UNION ALL BY NAME
        SELECT r.finding_key, x.span_path AS path, x.span_start_line AS start_line,
               x.span_start_column AS start_column, x.span_end_line AS end_line,
               x.span_end_column AS end_column, '_' || r.name AS text
        FROM renamable r JOIN symbol_symbols_reference_references x ON x.parent_id = r.record_id
    """
    examples = (
        Example(
            bad="LIMIT = 3\n\n\ndef clamp(value):\n    return min(value, LIMIT)\n",
            good="_LIMIT = 3\n\n\ndef clamp(value):\n    return min(value, _LIMIT)\n",
        ),
    )


class CrossModuleProjectConstantImport(Rule, id="PY-CONS0002", applicability="review"):
    """Project constants used outside their defining module, imported by name or reached through
    the module (`schema.ALIAS`), unless that module is a `constants.py` or sits in a `constants/`
    package, where every constant other modules share belongs. A capitalized name the project defines as a function or a
    class (`PK`, `FK`) is a callable in the style of a column DSL, not a constant."""

    preferred_module: str = "constants"

    query = """
            WITH constants AS (
              SELECT DISTINCT f.module, s.name
              FROM placed_symbols() s JOIN core_file f ON f.path = s.path
              WHERE s.scope = 'module' AND s.is_constant_assignment
                AND regexp_matches(ltrim(s.name, '_'), '^[A-Z][A-Z0-9_]*$')
            ), imported AS (
              SELECT i.path, i.start_line, i.start_column,
                     if(i.imported_name <> '', i.imported_name, i.name) AS name,
                     coalesce(i.source_module, i.module) AS module
              FROM core_import i
              WHERE i.is_project_owned
                AND regexp_matches(ltrim(if(i.imported_name <> '', i.imported_name, i.name), '_'),
                                   '^[A-Z][A-Z0-9_]*$')
                AND NOT EXISTS (SELECT 1 FROM core_def d
                                WHERE d.language = 'python' AND d.owner_id IS NULL
                                  AND d.name = if(i.imported_name <> '', i.imported_name, i.name))
            ), reached AS (
              SELECT f.path, n.start_line, n.start_column, c.name, c.module
              FROM core_import i
              JOIN core_file f ON f.file_id = i.file_id
              JOIN constants c ON c.module = i.target AND c.module <> f.module
              JOIN core_node n ON n.file_id = i.file_id AND n.type = 'attribute'
                AND n.name = i.name || '.' || c.name
              WHERE i.is_project_owned
            )
            SELECT * FROM (SELECT * FROM imported UNION ALL SELECT * FROM reached)
            WHERE NOT list_contains(string_split(module, '.'), $preferred_module)
        """
    message = (
        "`{name}` is a constant of `{module}` used in another module; a shared constant "
        "belongs in the constants.py of the lowest package its users share"
    )
    # One move per defining module and destination (`move-constants`), so each importing
    # statement is rewritten once for every constant leaving that module. The destination is the
    # `constants.py` of the lowest package holding the module and every user of the constant in
    # its own top package (beside the module when nothing else shares it; inside a package's
    # `__init__.py`); a user in another top package (a test suite) imports it by its full name.
    # Each definition goes with the comments above it; every `from ... import NAME` is respelled
    # to the destination, relatively when it was relative, and every `module.NAME` reads
    # `constants.NAME`; the old module imports back what it still reads. Left alone: a constant
    # some use of which the facts did not count, a destination beside a `constants/` package, a
    # module something star-imports, and what the operation refuses (a value that is no literal,
    # a name the destination holds otherwise, an importer binding `constants` already).
    fix = r"""
        WITH wanted AS (
          SELECT module, name, min(finding_key) AS finding_key, count(*) AS uses
          FROM finding GROUP BY module, name
        ), files AS (
          SELECT path, module,
                 if(ends_with(path, '__init__.py'), string_split(module, '.'),
                    list_slice(string_split(module, '.'), 1, -2)) AS package
          FROM core_file WHERE regexp_matches(path, '\.py$')
        ), defined AS (
          SELECT w.*, f.path, f.package,
                 string_split(rtrim(regexp_replace(f.path, '[^/]*$', ''), '/'), '/') AS folders
          FROM wanted w JOIN files f USING (module)
        ), imported AS (
          SELECT x.* FROM imported_names() x SEMI JOIN defined USING (module, name)
        ), reaching AS (
          SELECT x.* FROM reached_names() x SEMI JOIN defined USING (module, name)
        ), shared AS (
          SELECT d.module, d.name,
                 coalesce(list_position(list_transform(
                   range(1, least(len(d.package), len(x.importer_package)) + 1),
                   lambda k: d.package[k] = x.importer_package[k]), false) - 1,
                   least(len(d.package), len(x.importer_package))) AS common
          FROM defined d
          JOIN (SELECT module, name, importer_package FROM imported
                UNION ALL SELECT module, name, importer_package FROM reaching) x
            USING (module, name)
        ), accounted AS (
          SELECT d.*, least(len(d.package), coalesce(
                   (SELECT min(x.common) FROM shared x
                    WHERE x.module = d.module AND x.name = d.name AND x.common > 0),
                   len(d.package))) AS common
          FROM defined d
          WHERE d.uses = (SELECT count(*) FROM imported x WHERE x.module = d.module
                                                           AND x.name = d.name)
                       + (SELECT count(*) FROM reaching x WHERE x.module = d.module
                                                           AND x.name = d.name)
        ), placed AS (
          SELECT a.*,
                 array_to_string(list_slice(a.package, 1, a.common), '.') AS above,
                 array_to_string(
                   list_slice(a.folders, 1, len(a.folders) - len(a.package) + a.common), '/')
                   AS directory
          FROM accounted a
        ), targets AS (
          SELECT p.*, if(p.above = '', 'constants', p.above || '.constants') AS destination_module,
                 if(p.directory = '', '', p.directory || '/') AS folder
          FROM placed p
        ), importer_lists AS (
          SELECT t.module, t.destination_module,
                 list(x ORDER BY x.path, x.start_line, x.start_column) AS importers
          FROM imported x JOIN targets t USING (module, name)
          GROUP BY t.module, t.destination_module
        ), reach_lists AS (
          SELECT t.module, t.destination_module,
                 list(x ORDER BY x.path, x.statement_start_line, x.start_line, x.start_column)
                   AS reached
          FROM reaching x JOIN targets t USING (module, name)
          GROUP BY t.module, t.destination_module
        )
        SELECT min(t.finding_key) AS finding_key, 'move-constants' AS operation, t.path,
               t.module, s.content, t.folder || 'constants.py' AS destination,
               t.destination_module, c.content AS destination_content,
               list(t.name ORDER BY t.name) AS names, i.importers, r.reached
        FROM targets t
        JOIN source_files s ON s.path = t.path
        LEFT JOIN importer_lists i USING (module, destination_module)
        LEFT JOIN reach_lists r USING (module, destination_module)
        LEFT JOIN source_files c ON c.path = t.folder || 'constants.py'
        WHERE NOT EXISTS (SELECT 1 FROM core_file x
                          WHERE starts_with(x.path, t.folder || 'constants/'))
          AND NOT EXISTS (SELECT 1 FROM core_import w
                          WHERE coalesce(w.source_module, w.module) = t.module AND w.name = '*')
        GROUP BY t.module, t.path, s.content, t.folder, t.destination_module, c.content,
                 i.importers, r.reached
    """
    examples = (
        Example(
            bad="from .limits import LIMIT\n\n\ndef clamp(v):\n    return min(v, LIMIT)\n",
            good="from .constants import LIMIT\n\n\ndef clamp(v):\n    return min(v, LIMIT)\n",
            path="pkg/use.py",
            context={
                "pkg/__init__.py": "",
                "pkg/limits.py": "# How far a value may go.\nLIMIT = 3\n\n\ndef top():\n    return LIMIT\n",
            },
        ),
        Example(
            bad="from . import limits\n\n\ndef clamp(v):\n    return min(v, limits.LIMIT)\n",
            good="from . import constants\n\n\ndef clamp(v):\n    return min(v, constants.LIMIT)\n",
            path="pkg/use.py",
            context={
                "pkg/__init__.py": "",
                "pkg/limits.py": "LIMIT = 3\n",
                "pkg/constants.py": "LIMIT = 3\n",
            },
        ),
        Example(
            bad="from ..a.limits import LIMIT\n\n\ndef clamp(v):\n    return min(v, LIMIT)\n",
            good="from ..constants import LIMIT\n\n\ndef clamp(v):\n    return min(v, LIMIT)\n",
            path="pkg/b/use.py",
            context={
                "pkg/__init__.py": "",
                "pkg/a/__init__.py": "",
                "pkg/b/__init__.py": "",
                "pkg/a/limits.py": "LIMIT = 3\n\n\ndef top():\n    return LIMIT\n",
            },
        ),
    )


class DependencySafeConstantOrder(Rule, id="PY-CONS0003", applicability="review"):
    """Constants declared later than their earliest dependency-safe top-level position."""

    query = """
            SELECT m.path, m.start_line, m.start_column, m.fact_id, count(*) AS value
            FROM module_constant_placements c JOIN module_facts m USING (fact_id)
            WHERE c.intervening_statement_count > 0 AND NOT m.is_test
            GROUP BY ALL
        """
    message = "dependency safe constant order is {value} for `{fact_id}`"
    # Each late constant moves up (`raise-constants`) to just below the last import or the last
    # statement binding a name its value reads, its comments along. Review: a value that reads a
    # name only through a string (`getattr`, a template) moves past whatever binds it.
    fix = r"""
        SELECT f.finding_key, 'raise-constants' AS operation, m.path, s.content,
               list(c.name ORDER BY c.ordinal) AS names
        FROM finding f
        JOIN module_facts m USING (fact_id)
        JOIN module_constant_placements c USING (fact_id)
        JOIN source_files s ON s.path = m.path
        WHERE c.intervening_statement_count > 0
        GROUP BY f.finding_key, m.path, s.content
    """
    examples = (
        Example(
            bad=(
                """import os


def home():
    return os.environ.get(HOME, "")


# The variable naming the home folder.
HOME = "HOME"
"""
            ),
            good=(
                """import os

# The variable naming the home folder.
HOME = "HOME"


def home():
    return os.environ.get(HOME, "")
"""
            ),
        ),
    )
