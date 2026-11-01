"""Rules over how imports are spelled (mcmr's PY-IMPO style family)."""

from ....catalog import Example, Rule

_PROJECT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}


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
    message = (
        "`{fact_id}` imports a module owned by its own project package through an absolute name"
    )
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
    # whose directory has no package initializer, or that puts its root on `sys.path`, is a script
    # run by path, where a relative import fails.
    fix = """
        WITH spelled AS (
          SELECT f.*, str_split(f.module, '.') AS target,
                 str_split(f.importer_module, '.')[1:len(str_split(f.importer_module, '.'))
                   - CASE WHEN is_initializer_path(f.path) THEN 0 ELSE 1 END] AS package
          FROM finding f
          WHERE f.importer_module <> '' AND contains(f.path, '/')
            AND regexp_replace(f.path, '/[^/]*$', '') IN (
              SELECT regexp_replace(path, '/[^/]*$', '') FROM module_facts
              WHERE is_package_initializer)
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


class ProjectPrivateImport(Rule, id="PY-IMPO0002", applicability="review"):
    """Imports crossing a project-owned nonpublic boundary.

    The fix makes what the project imports public. A name `_hidden` that a module defines is
    renamed `hidden`, in its module and in every file reading it, and a module file `_tools.py`
    is moved to `tools.py`, its stub with it, every import of it following. It leaves the import
    as it is when the new name or file is taken, when a file that is no Python spells the old one,
    when a star import or a string could read it, and for a nonpublic package, a name of a
    module with a stub beside it, or a name the module does not define."""

    query = r"""
        WITH imported AS (
          SELECT *, starts_with(imported_name, '_') AND NOT starts_with(imported_name, '__')
                    AND regexp_matches(substr(imported_name, 2), '\p{Ll}') AS private_name
          FROM core_import WHERE is_project_owned
        )
        SELECT path, start_line, start_column, 'import:' || path || ':' || name AS fact_id,
               statement_type, imported_name, private_name, has_private_module_component,
               coalesce(source_module, module) AS imported_module, target
        FROM imported WHERE has_private_module_component OR private_name
    """
    message = "`{fact_id}` imports a nonpublic module or name that the project owns"
    # One operation per thing made public, whatever the number of imports of it. A private name
    # `_n` that the imported module defines is renamed `n` by `rewrite-references`; a module
    # file with a private last component, named by the statement or imported from its package
    # as a name, is moved by `move-modules` to the file without the underscore. Only modules with
    # one source file are renamed in, so that no stub is left holding the old name. A rename also
    # hands over (`scanned`) every Python file importing the module or spelling `module.name`, for
    # the operation to refuse when one of them spells the name in a string.
    fix = r"""
        WITH sources AS (
          SELECT c.module, c.path, c.is_package,
                 EXISTS (SELECT 1 FROM core_file s WHERE s.path = c.path || 'i') AS stubbed
          FROM core_file c WHERE c.language = 'python' AND ends_with(c.path, '.py')
          QUALIFY count(*) OVER (PARTITION BY c.module) = 1
        ), renamed AS (
          SELECT d.path, d.module, f.imported_name AS name, min(f.finding_key) AS finding_key
          FROM finding f JOIN sources d ON d.module = f.imported_module
          WHERE f.private_name AND NOT f.has_private_module_component
            AND f.statement_type = 'import_from_statement' AND NOT d.stubbed
            AND NOT EXISTS (SELECT 1 FROM sources t WHERE t.module = f.target)
            AND NOT EXISTS (SELECT 1 FROM source_files s
                            WHERE NOT regexp_matches(s.path, '\.pyi?$')
                              AND (contains(s.content, d.module || '.' || f.imported_name)
                                   OR contains(s.content, d.module || ':' || f.imported_name)))
          GROUP BY ALL
        ), scanned AS (
          SELECT r.finding_key,
                 list({'path': s.path, 'content': s.content} ORDER BY s.path) AS files
          FROM renamed r, module_readers([r.module]) x, source_files s
          WHERE regexp_matches(s.path, '\.pyi?$') AND s.path <> r.path
            AND (list_contains(list_transform(x.files, lambda f: f.path), s.path)
                 OR contains(s.content, r.module || '.' || r.name)
                 OR contains(s.content, r.module || ':' || r.name))
          GROUP BY r.finding_key
        ), moved AS (
          SELECT m.path, m.module, min(f.finding_key) AS finding_key
          FROM finding f
          JOIN sources m ON m.module = CASE WHEN f.has_private_module_component
                                            THEN f.imported_module ELSE f.target END
          WHERE NOT m.is_package AND regexp_matches(m.module, '(^|\.)_[^_.][^.]*$')
            AND (f.has_private_module_component
                 OR f.private_name AND f.statement_type = 'import_from_statement')
          GROUP BY ALL
        )
        SELECT r.finding_key, 'rewrite-references' AS operation, 'module' AS scope, r.path,
               s.content, r.module, r.name, substr(r.name, 2) AS "to", n.importers, n.reached,
               n.dotted, n.starred, c.files AS scanned
        FROM renamed r JOIN source_files s ON s.path = r.path
             LEFT JOIN scanned c ON c.finding_key = r.finding_key,
             name_readers(r.module, [r.name]) n
        UNION ALL BY NAME
        SELECT m.finding_key, 'move-modules' AS operation,
               [{'path': m.path, 'target': regexp_replace(m.path, '(^|/)_([^/]+)$', '\1\2')}]
                 || coalesce((SELECT list({'path': s.path,
                                           'target': regexp_replace(s.path, '(^|/)_([^/]+)$', '\1\2')})
                              FROM core_file s WHERE s.path = m.path || 'i'), []) AS moves,
               x.files,
               coalesce((SELECT list(h.path) FROM core_file h
                         WHERE h.language = 'python'
                           AND regexp_replace(h.path, '[^/]+$', '')
                               = regexp_replace(m.path, '[^/]+$', '')), []) AS held,
               coalesce((SELECT list(s.path) FROM source_files s
                         WHERE NOT regexp_matches(s.path, '\.pyi?$')
                           AND (contains(s.content, m.module) OR contains(s.content, m.path))),
                        []) AS mentions
        FROM moved m, module_readers([m.module]) x
    """
    examples = (
        Example(
            bad="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            good="from .a import hidden\n\nVALUE = hidden + 1\n",
            path="src/pkg/user.py",
            context={**_PROJECT, "src/pkg/a.py": "_hidden = 1\n"},
        ),
        Example(
            bad="from ._tools import tool\n\nVALUE = tool()\n",
            good="from .tools import tool\n\nVALUE = tool()\n",
            path="src/pkg/user.py",
            context={**_PROJECT, "src/pkg/_tools.py": "def tool():\n    return 1\n"},
        ),
        Example(
            bad="from .a import _hidden\n\nhidden = _hidden + 1\n",
            good="from .a import _hidden\n\nhidden = _hidden + 1\n",
            path="src/pkg/user.py",
            context={**_PROJECT, "src/pkg/a.py": "_hidden = 1\n"},
        ),
        Example(
            bad="from ._tools import tool\n\nVALUE = tool()\n",
            good="from ._tools import tool\n\nVALUE = tool()\n",
            path="src/pkg/user.py",
            context={
                **_PROJECT,
                "src/pkg/_tools.py": "def tool():\n    return 1\n",
                "src/pkg/tools.py": "OTHER = 1\n",
            },
        ),
        Example(
            bad="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            good="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            path="src/pkg/user.py",
            context={
                **_PROJECT,
                "pyproject.toml": (
                    """[project]
name = "pkg"
version = "0"

[project.scripts]
pkg = "pkg.a:_hidden"
"""
                ),
                "src/pkg/a.py": "_hidden = 1\n",
            },
        ),
        Example(
            bad="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            good="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            path="src/pkg/user.py",
            context={
                **_PROJECT,
                "src/pkg/a.py": "_hidden = 1\n",
                "src/pkg/a.pyi": "_hidden: int\n",
            },
        ),
        Example(
            bad="from ._inner import Z\n\nVALUE = Z\n",
            good="from ._inner import Z\n\nVALUE = Z\n",
            path="src/pkg/user.py",
            context={**_PROJECT, "src/pkg/_inner/__init__.py": "Z = 2\n"},
        ),
        Example(
            bad="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            good="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            path="src/pkg/user.py",
            context={
                **_PROJECT,
                "src/pkg/a.py": "_hidden = 1\n",
                "src/pkg/other.py": 'from . import a\n\nVALUE = getattr(a, "_hidden")\n',
            },
        ),
        Example(
            bad="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            good="from .a import _hidden\n\nVALUE = _hidden + 1\n",
            path="src/pkg/user.py",
            context={
                **_PROJECT,
                "src/pkg/a.py": "_hidden = 1\n",
                "tests/test_b.py": 'from unittest import mock\n\nmock.patch("pkg.a._hidden", 2)\n',
            },
        ),
    )


class RelativeImportBeyondPackage(Rule, id="PY-IMPO0004"):
    """Relative imports climbing past the top-level package they start in."""

    query = """
        WITH stated AS (
          SELECT *, len(str_split(importer_module, '.'))
                      - CASE WHEN is_initializer_path(path) THEN 0 ELSE 1 END AS owned
          FROM core_import WHERE relative_level > 0
        )
        SELECT path, start_line, start_column, 'import:' || path || ':' || name AS fact_id
        FROM stated WHERE owned > 0 AND relative_level > owned
    """
    message = (
        "`{fact_id}` is a relative import that climbs past the top-level package it starts in"
    )
    examples = (
        Example(
            bad="from ... import base\n",
            good="from .. import base\n",
            path="pkg/sub/__init__.pyi",
            context={"pkg/__init__.py": "", "pkg/base.py": "VALUE = 1\n"},
        ),
    )
