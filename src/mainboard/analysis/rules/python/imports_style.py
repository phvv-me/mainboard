"""Rules over how imports are spelled (mcmr's PY-IMPO style family)."""

from ...catalog import Example, Rule


class InternalRelativeImport(Rule, id="PY-IMPO0001", applicability="safe"):
    """Absolute imports of a module owned by the importing project package."""

    query = """
        WITH nodes AS (
          SELECT node_id, role, path, start_line, start_column, end_line, end_column
          FROM import_binding_nodes WHERE role IN ('module', 'declaration') AND ordinal = 0
        )
        SELECT i.*, m.path AS module_path, m.start_line AS module_start_line,
          m.start_column AS module_start_column, m.end_line AS module_end_line,
          m.end_column AS module_end_column,
          d.path AS declaration_path, d.start_line AS declaration_start_line,
          d.start_column AS declaration_start_column, d.end_line AS declaration_end_line,
          d.end_column AS declaration_end_column
        FROM import_binding_facts i
        LEFT JOIN nodes m ON m.node_id = i.module_node_id AND m.role = 'module'
        LEFT JOIN nodes d ON d.node_id = i.declaration_id AND d.role = 'declaration'
        WHERE i.is_project_owned AND NOT i.is_relative AND NOT i.is_wildcard
        QUALIFY row_number() OVER (PARTITION BY i.fact_id, i.declaration_id
                                   ORDER BY i.fact_order) = 1
    """
    message = "`{fact_id}` satisfies internal relative import"
    examples = (
        Example(
            bad="from pkg.helpers import tool\n",
            good="from .helpers import tool\n",
            path="src/pkg/use.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
                "src/pkg/helpers.py": "def tool():\n    return 1\n",
            },
        ),
    )
    # The module through the shortest relative path from the importer's package, once per
    # statement (`from a.b import c, d` states `a.b` once); `import a.b.c as x` becomes
    # `from .. import c as x` when it states one binding. Only inside a real package: a file
    # whose directory has no `__init__.py`, or that puts its root on `sys.path`, is a script
    # run by path, where a relative import fails.
    fix = """
        WITH spelled AS (
          SELECT f.*, str_split(f.module, '.') AS target,
                 str_split(f.importer_module, '.')[1:len(str_split(f.importer_module, '.'))
                   - CASE WHEN ends_with(f.path, '__init__.py') THEN 0 ELSE 1 END] AS package
          FROM finding f
          WHERE f.importer_module <> ''
            AND regexp_replace(f.path, '/[^/]*$', '') || '/__init__.py' IN (
              SELECT path FROM module_facts WHERE is_package_initializer)
            AND NOT EXISTS (SELECT 1 FROM call_calls c WHERE c.node_path = f.path
                            AND c.qualified_name IN ('sys.path.insert', 'sys.path.append'))
        ), related AS (
          SELECT *, coalesce(list_min(list_filter(range(1, len(package) + 1),
                   lambda i: target[i] IS DISTINCT FROM package[i])) - 1, len(package)) AS shared
          FROM spelled
        ), relative AS (
          SELECT *, repeat('.', len(package) - shared + 1) AS dots
          FROM related
        )
        SELECT finding_key,
          CASE WHEN imported_name <> module THEN module_path ELSE declaration_path END AS path,
          CASE WHEN imported_name <> module THEN module_start_line
               ELSE declaration_start_line END AS start_line,
          CASE WHEN imported_name <> module THEN module_start_column
               ELSE declaration_start_column END AS start_column,
          CASE WHEN imported_name <> module THEN module_end_line
               ELSE declaration_end_line END AS end_line,
          CASE WHEN imported_name <> module THEN module_end_column
               ELSE declaration_end_column END AS end_column,
          CASE
            WHEN imported_name <> module AND module_node_id <> ''
                 AND fact_order = min(fact_order) OVER (PARTITION BY module_node_id)
              THEN dots || array_to_string(target[shared + 1:], '.')
            WHEN imported_name = module AND name <> target[1] AND declaration_id <> ''
                 AND is_sole_binding AND len(target) > shared
              THEN 'from ' || dots || array_to_string(target[shared + 1:len(target) - 1], '.')
                   || ' import ' || target[len(target)] || ' as ' || name
          END AS text
        FROM relative
    """


class ProjectPrivateImport(Rule, id="PY-IMPO0002"):
    """Imports crossing a project-owned nonpublic boundary."""

    query = """
        SELECT path, start_line, start_column, 'import:' || path || ':' || name AS fact_id
        FROM core_import
        WHERE is_project_owned AND (has_private_module_component
          OR (starts_with(imported_name, '_') AND NOT starts_with(imported_name, '__')
              AND regexp_matches(substr(imported_name, 2), '\\p{Ll}')))
    """
    message = "`{fact_id}` satisfies project private import"


class RelativeImportBeyondPackage(Rule, id="PY-IMPO0004"):
    """Relative imports climbing past the top-level package they start in."""

    query = """
        WITH stated AS (
          SELECT *, len(str_split(importer_module, '.'))
                      - CASE WHEN ends_with(path, '__init__.py') THEN 0 ELSE 1 END AS owned
          FROM core_import WHERE relative_level > 0
        )
        SELECT path, start_line, start_column, 'import:' || path || ':' || name AS fact_id
        FROM stated WHERE owned > 0 AND relative_level > owned
    """
    message = "`{fact_id}` satisfies relative import beyond package"
