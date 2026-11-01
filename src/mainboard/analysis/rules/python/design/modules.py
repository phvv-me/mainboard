"""Rules over barrels, package initializers and explicit exports (mcmr's PY-MODU family)."""

from ....catalog import Example, Rule


class NonInitReexportModule(Rule, id="PY-MODU0001"):
    """Barrel modules that define no behavior beyond re-exporting imports."""

    query = """
        SELECT * FROM module_facts
        WHERE has_only_imports_and_all AND NOT is_package_initializer
          AND NOT ends_with(path, 'conftest.py')
    """
    message = "`{fact_id}` only re-exports imports and is not a package initializer"
    # An `__init__.pyi` is a package initializer like `__init__.py`: re-exporting is its job.
    examples = (
        Example(
            bad='from .tool import tool\n\n__all__ = ["tool"]\n',
            good="from .tool import tool\n\n\ndef run() -> None:\n    tool()\n",
            context={
                "stubs/pkg/__init__.pyi": "from .tool import tool as tool\n",
                "stubs/pkg/sub/__init__.pyi": "# The stubs of `sub` sit beside this file.\n",
            },
        ),
    )


class EmptyPackageInitializer(Rule, id="PY-MODU0002", applicability="review"):
    """Package initializers that state no executable package surface."""

    # A stub initializer is left out: nothing in a stub executes, and removing one turns its
    # package into a namespace package for the type checkers.
    query = """
        SELECT * FROM module_facts
        WHERE is_package_initializer AND NOT is_test AND executable_statement_count = 0
          AND NOT ends_with(path, '.pyi')
    """
    message = "`{fact_id}` is a package initializer with no executable statement"
    examples = (
        Example(
            bad="\n",
            good=None,
            path="src/pkg/blank/__init__.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": '"""The package."""\n',
                "src/pkg/blank/mod.py": "x = 1\n",
            },
        ),
    )
    # The blank initializer goes and its package becomes a namespace package. One holding a
    # docstring or comments says something, so it stays.
    fix = """
        SELECT path, 'remove' AS operation FROM finding
        WHERE EXISTS (SELECT 1 FROM source_files s
                      WHERE s.path = finding.path AND trimmed(s.content) = '')
    """


_HELPER = """def helper():
    return 1


__all__ = ["helper"]
"""
_HELPER_READING = "LIMIT = 3\n\n\ndef helper():\n    return LIMIT\n"
_HELPER_USER = "from pkg import helper\n\n\ndef run() -> int:\n    return helper()\n"
_PRIVATES = "def _one():\n    return 1\n\n\ndef _two():\n    return 2\n"
_PRIVATES_USER = """from pkg import _one, _two


def run() -> int:
    return _one() + _two()
"""
_HIDDEN = "def _hidden():\n    return 1\n"
_HIDDEN_USER = "from pkg import _hidden\n\n\ndef run() -> int:\n    return _hidden()\n"


class InitializerDeclaration(Rule, id="PY-MODU0003", applicability="review"):
    """Ordinary function and class declarations in package initializers."""

    # A stub initializer is left out: its declarations are the surface the stub describes, not
    # behavior. Each declaration is a finding of its own, so that moving one is progress the fixer
    # sees.
    query = """
        SELECT f.path, d.start_line, d.start_column, m.name,
               '`' || m.name || '` is declared in a package initializer' AS message
        FROM module_facts f
        JOIN module_members m ON m.fact_id = f.fact_id
        JOIN core_def d ON d.path = f.path AND d.name = m.name AND d.owner_id IS NULL
        WHERE f.is_package_initializer AND NOT ends_with(f.path, '.pyi')
          AND (m.kind = 'class' OR (m.kind = 'function' AND m.name NOT IN ('__getattr__', '__dir__')))
    """
    # Each declaration moves into a module of its own beside the initializer
    # (`move-definitions`). A public one is imported back, so that the package keeps its surface
    # and its importers stay as they are (a module of the same name as the function would be
    # shadowed by it, so `package.name` must not be rerouted through the module); a private one is
    # not, and every import of it is respelled. A module that exists is left untouched, and so is
    # what the operation refuses (a declaration reading a name of the initializer that stays, an
    # importer binding the package as `import a as b`).
    fix = r"""
        WITH moved AS (
          SELECT finding_key, path, name, module_path(regexp_replace(path, '/?[^/]*$', ''), name)
                   AS destination, starts_with(name, '_') AS is_private
          FROM finding
        ), free AS (
          SELECT * FROM moved m
          WHERE NOT EXISTS (SELECT 1 FROM core_file c WHERE c.path = m.destination)
        )
        SELECT m.finding_key, 'move-definitions' AS operation,
               r.* REPLACE (list_filter(r.importers, lambda i: m.is_private) AS importers,
                            list_filter(r.reached, lambda i: m.is_private) AS reached)
        FROM free m, move_row(m.path, [m.name], m.destination) r
        UNION ALL BY NAME
        SELECT finding_key, path,
               'from .' || regexp_extract(destination, '([^/]*)\.py$', 1) || ' import ' || name
                 AS "import"
        FROM free WHERE NOT is_private
    """
    examples = (
        Example(
            bad="def launch():\n    return 1\n",
            good="from .launch import launch\n",
            path="src/launcher/__init__.py",
            context={"stubs/launcher/__init__.pyi": "def launch() -> int: ...\n"},
        ),
        Example(
            bad=_HELPER,
            good='from .helper import helper\n\n__all__ = ["helper"]\n',
            path="pkg/__init__.py",
            context={"pkg/use.py": _HELPER_USER},
        ),
        Example(
            bad=_HIDDEN_USER,
            good=_HIDDEN_USER.replace("from pkg import", "from pkg._hidden import"),
            path="pkg/use.py",
            context={"pkg/__init__.py": _HIDDEN},
        ),
        Example(
            bad=_PRIVATES_USER,
            good=_PRIVATES_USER.replace(
                "from pkg import _one, _two",
                "from pkg._one import _one\nfrom pkg._two import _two",
            ),
            path="pkg/use.py",
            context={"pkg/__init__.py": _PRIVATES},
        ),
        Example(
            bad=_HELPER_USER,
            good=_HELPER_USER,
            path="pkg/use.py",
            context={"pkg/__init__.py": _HELPER, "pkg/helper.py": "VALUE = 1\n"},
        ),
        Example(
            bad=_HELPER_USER,
            good=_HELPER_USER,
            path="pkg/use.py",
            context={"pkg/__init__.py": _HELPER_READING},
        ),
    )


class ExplicitAllOnlyInInitializer(Rule, id="PY-MODU0004", applicability="review"):
    """Explicit package export lists outside package initializers."""

    query = """
        SELECT f.*, d.declarations FROM module_facts f
        LEFT JOIN (
          SELECT fact_id, list({'path': span_path, 'start_line': span_start_line,
                                'start_column': span_start_column, 'end_line': span_end_line,
                                'end_column': span_end_column} ORDER BY ordinal) AS declarations
          FROM module_all_declarations GROUP BY fact_id) d ON d.fact_id = f.fact_id
        WHERE f.declares_all AND NOT f.is_package_initializer
    """
    message = "`{fact_id}` declares an explicit export list outside a package initializer"
    examples = (
        Example(
            bad='__all__ = ["tool"]\n\n\ndef tool():\n    return 1\n',
            good="def tool():\n    return 1\n",
        ),
    )
    # Every `__all__` statement of the module goes, together or not at all. A module whose
    # imports re-export through it is a barrel: deleting the list would change what it exports.
    fix = """
        WITH declared AS (
          SELECT finding_key, path AS module_path, unnest(declarations, recursive := true)
          FROM finding
        )
        SELECT d.finding_key, d.path,
               imports_tidy_start(d.start_line, first_gap.previous_line, last_gap.final)
                 AS start_line,
               0 AS start_column,
               imports_tidy_end(d.start_line, d.end_line, first_gap.previous_line,
                                last_gap.next_line) AS end_line,
               0 AS end_column,
               CASE WHEN bool_and(
                      trimmed(imports_before(opening.text, d.start_column::BIGINT)) = ''
                      AND regexp_matches(imports_from(closing.text, d.end_column::BIGINT),
                                         '^\\s*(#.*)?$')
                      AND NOT coalesce(imports_empties_block(
                        opening.previous_text, opening.text, closing.following_text), false))
                      OVER (PARTITION BY d.finding_key)
                     AND NOT EXISTS (SELECT 1 FROM import_binding_facts i
                                     WHERE i.path = d.module_path AND i.is_reexported)
                    THEN '' END AS text
        FROM declared d
        LEFT JOIN imports_line_gaps() first_gap
          ON first_gap.path = d.path AND first_gap.line = d.start_line
        LEFT JOIN imports_line_gaps() last_gap
          ON last_gap.path = d.path AND last_gap.line = d.end_line
        LEFT JOIN imports_code_lines() opening
          ON opening.path = d.path AND opening.line = d.start_line
        LEFT JOIN imports_code_lines() closing
          ON closing.path = d.path AND closing.line = d.end_line
    """


class UnusedExplicitExport(Rule, id="PY-MODU0005", applicability="review"):
    """Explicit package exports no repository consumer uses. The initializers of the package a
    `pyproject.toml` publishes (its `[project]` name) are left out: their exports are for the
    projects that install it, which the repository does not hold."""

    query = r"""
        SELECT e.*, (SELECT min(n.id) FROM export_public_export_nodes n
                     WHERE n.fact_id = e.fact_id) AS member_id
        FROM export_facts e
        WHERE public_export_consumer_count = 0
          AND NOT EXISTS (
            SELECT 1 FROM source_files t
            WHERE ends_with(t.path, 'pyproject.toml') AND is_initializer_path(e.path)
              AND starts_with(e.path, regexp_replace(t.path, 'pyproject\.toml$', ''))
              AND regexp_matches(t.content, '(?ms)^\[project\]\s*$.*?^name\s*=\s*"[^"]+"')
              AND replace(lower(regexp_extract(t.content,
                    '(?ms)^\[project\]\s*$.*?^name\s*=\s*"([^"]+)"', 1)), '-', '_')
                  = split_part(e.public_export_module, '.', 1))
    """
    message = (
        "`{public_export_name}` explicitly exports `{public_export_target}`, but no repository "
        "consumer uses that public route"
    )
    examples = (
        Example(
            bad='from .tool import tool\n\n__all__ = ["tool"]\n',
            good="",
        ),
    )
    # The item leaves `__all__` (the whole statement when it was the last), and with it the
    # import that only re-exported the name, since that import would be PY-IMPO0003's next
    # finding: together or not at all. A `conftest.py` is skipped (pytest finds its hooks and
    # fixtures by name, not by import), as is an import that states its re-export as `x as x`.
    fix = """
        WITH reexports AS (
          SELECT f.finding_key, i.binding_id AS member_id
          FROM finding f
          JOIN import_binding_facts i ON i.path = f.path AND i.name = f.public_export_name
          JOIN import_binding_nodes b
            ON b.fact_id = i.fact_id AND b.node_id = i.binding_id AND b.role = 'binding'
          WHERE i.reference_count = 0 AND NOT (i.has_qualifying_use OR i.has_documented_side_effect
            OR i.is_wildcard OR i.module = '__future__')
            AND b.text <> i.name || ' as ' || i.name
        ), candidates AS (
          SELECT finding_key, member_id FROM finding
          WHERE member_id IS NOT NULL AND NOT ends_with(path, 'conftest.py')
          UNION ALL
          SELECT finding_key, member_id FROM reexports
          WHERE finding_key IN (SELECT finding_key FROM finding
                                WHERE NOT ends_with(path, 'conftest.py'))
        ), tried AS (
          SELECT * FROM imports_member_removals((SELECT list(member_id) FROM candidates))
        ), chosen AS (
          SELECT finding_key, member_id FROM candidates
          WHERE finding_key IN (
            SELECT c.finding_key FROM candidates c LEFT JOIN tried t USING (member_id)
            GROUP BY c.finding_key HAVING bool_and(coalesce(t.feasible, false)))
        )
        SELECT c.finding_key, r.path, r.start_line, r.start_column, r.end_line, r.end_column,
               '' AS text
        FROM chosen c
        JOIN imports_member_removals((SELECT list(member_id) FROM chosen)) r USING (member_id)
        WHERE r.start_line IS NOT NULL
    """
