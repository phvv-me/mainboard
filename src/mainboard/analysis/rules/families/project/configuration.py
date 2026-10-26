"""mcmr's ProjectConfigurationFact, derived from the root `pyproject.toml` and the Python trees.

One fact states what the project configures about itself. Its Python target comes from the
manifest, parsed by `tomllib` in `ProjectManifest`: the minimum minor `requires-python` allows,
every tool that states a target and the minor each states, and the minors of ruff's per-file
targets. A manifest without a `[project]` table (tool-only) inherits the `project` table of the
nearest `pyproject.toml` above it, as mcmr does, and its checkers that state no target take the
inherited minimum. Its assignments are the literal collections of strings that Python modules
assign to a name, at module level or directly in a class or function body (not under `if`, `for`
or `with`), whose names can state a discovery policy; a collection whose elements are all plain
strings is kept with its values.
"""

import re
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from itertools import takewhile
from pathlib import Path

from pathspec import GitIgnoreSpec
from patos import FrozenModel

from .....core.project import Project
from ....catalog import Family, Function
from ....engine.facts import Facts, project_files
from ....project.config import ProjectConfig

_CHECKERS = ("ruff", "pyrefly", "ty", "mypy", "pyright", "basedpyright")
_TARGET_KEYS = ("target-version", "python_version", "python-version", "pythonVersion")


class ProjectManifest(FrozenModel):
    """What the root `pyproject.toml` declares about the Python version, with the `project` table
    it inherits when it has none. Every table holds specifier text for `project_minor` to read."""

    root: Path

    @staticmethod
    def _file_targets(tools: Mapping) -> list[tuple[str, str]]:
        """Ruff's per-file targets that are strings, by pattern."""
        ruff = tools.get("ruff")
        targets = ruff.get("per-file-target-version") if isinstance(ruff, dict) else None
        if not isinstance(targets, dict):
            return []
        return sorted((key, text) for key, text in targets.items() if isinstance(text, str))

    @staticmethod
    def _place(
        manifests: Sequence[tuple[str, Sequence[str]]], *, rule: str, pattern: str
    ) -> tuple[str, int]:
        """Where `rule`'s exclude of `pattern` is written, as a manifest and a line.

        That is the line quoting it inside the rule's own table, else the first line of any
        manifest quoting it, else the first line of `pyproject.toml`.
        """
        quoted = f'"{pattern}"'
        header = re.compile(rf'\[tool\.mcmr\.rules\.(?:"{re.escape(rule)}"|{re.escape(rule)})\]')
        owned = (
            (name, number)
            for name, lines in manifests
            for start, opening in enumerate(lines, 1)
            if header.fullmatch(opening.strip())
            for number, line in enumerate(
                takewhile(lambda line: not line.lstrip().startswith("["), lines[start:]),
                start + 1,
            )
            if quoted in line
        )
        anywhere = (
            (name, number)
            for name, lines in manifests
            for number, line in enumerate(lines, 1)
            if quoted in line
        )
        return next(owned, None) or next(anywhere, ("pyproject.toml", 1))

    @staticmethod
    def _read(path: Path) -> dict:
        with path.open("rb") as manifest:
            return tomllib.load(manifest)

    @staticmethod
    def _requires(project: dict | None) -> str:
        """The `requires-python` text, empty when absent or not a string."""
        requires = project.get("requires-python") if isinstance(project, dict) else None
        return requires if isinstance(requires, str) else ""

    @staticmethod
    def _target(settings: object) -> tuple[object, ...]:
        """The one-tuple holding the value a tool states its target under, whatever its type.

        It is empty when the tool states none.
        """
        if not isinstance(settings, dict):
            return ()
        for key in _TARGET_KEYS:
            if key in settings:
                return (settings[key],)
        environment = settings.get("environment")
        if isinstance(environment, Mapping) and "python-version" in environment:
            return (environment["python-version"],)
        return ()

    @staticmethod
    def _tools(tools: Mapping, inherited: str | None) -> list[tuple[str, str | None]]:
        """Each configured tool and the text stating its target.

        That is its own string, nothing when it states a non-string, and the `inherited` project
        declaration when it states none.
        """
        rows = []
        for name, settings in sorted(tools.items()):
            match ProjectManifest._target(settings):
                case ():
                    if name in _CHECKERS:
                        rows.append((name, inherited))
                case (str() as declaration,):
                    rows.append((name, declaration))
                case _:
                    rows.append((name, None))
        return rows

    def _project(self, manifest: Mapping) -> dict | None:
        """The `project` table in force: the manifest's own, else the nearest ancestor's."""
        if "project" in manifest:
            return manifest["project"]
        for ancestor in self.root.resolve().parents:
            with suppress(FileNotFoundError):
                if "project" in (above := self._read(ancestor / "pyproject.toml")):
                    return above["project"]
        return None

    def _unmatched(self) -> list[tuple[str, int, str, str]]:
        """Each rule exclude that matches no file of the project, where it is declared.

        The pattern is matched as a rule's own exclude is (gitignore), against every file the
        project holds, vendored ones included. Its place is the line quoting it in the rule's own
        table (`_place`).
        """
        files = project_files(self.root)
        manifests = [
            (path.relative_to(self.root).as_posix(), path.read_text(encoding="utf-8").splitlines())
            for path in (self.root / "pyproject.toml", Project().manifest(self.root))
            if path.is_file() and path.is_relative_to(self.root)
        ]
        return [
            (*ProjectManifest._place(manifests, rule=rule, pattern=pattern), rule, pattern)
            for rule, config in sorted(ProjectConfig.of(self.root).rules.items())
            for pattern in config.exclude
            if next(iter(GitIgnoreSpec.from_lines([pattern]).match_files(files)), None) is None
        ]

    def register(self, facts: Facts) -> None:
        """Create the four manifest tables in `facts`, once; they are empty without a manifest.

        They are `manifest_requirements` (declaration), `manifest_tools` (tool, declaration: every
        tool that states a target or is a checker), `manifest_file_targets` (pattern,
        declaration) and `manifest_rule_excludes` (path, line, rule, pattern: the rule excludes
        matching no file).
        """
        if facts.db.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = 'manifest_requirements'"
        ).fetchone():
            return
        for name, columns in (
            ("manifest_requirements", "declaration VARCHAR"),
            ("manifest_tools", "tool VARCHAR, declaration VARCHAR"),
            ("manifest_file_targets", "pattern VARCHAR, declaration VARCHAR"),
            (
                "manifest_rule_excludes",
                "path VARCHAR, line INTEGER, rule VARCHAR, pattern VARCHAR",
            ),
        ):
            facts.db.execute(f"CREATE TABLE {name} ({columns})")
        if unmatched := self._unmatched():
            facts.db.executemany(
                "INSERT INTO manifest_rule_excludes VALUES (?, ?, ?, ?)", unmatched
            )
        try:
            manifest = self._read(self.root / "pyproject.toml")
        except FileNotFoundError:
            return
        requires = self._requires(self._project(manifest))
        tools = manifest.get("tool")
        tools = tools if isinstance(tools, dict) else {}
        inherited = None if "project" in manifest else requires
        facts.db.execute("INSERT INTO manifest_requirements VALUES (?)", [requires])
        for table, rows in (
            ("manifest_tools", self._tools(tools, inherited)),
            ("manifest_file_targets", self._file_targets(tools)),
        ):
            if rows:
                facts.db.executemany(f"INSERT INTO {table} VALUES (?, ?)", rows)


class ProjectLiteral(Function):
    """The value of a quoted string literal, Python or TOML: prefix and quotes dropped, simple
    escapes decoded unless the literal is raw."""

    text: str

    expression = r"""
        CASE WHEN contains(lower(regexp_extract(text, '^([A-Za-z]*)', 1)), 'r')
             THEN project_unquoted(text)
             ELSE replace(replace(replace(replace(replace(replace(replace(
                    project_unquoted(text), chr(92) || chr(92), chr(1)),
                    chr(92) || 'n', chr(10)), chr(92) || 't', chr(9)),
                    chr(92) || '''', ''''), chr(92) || '"', '"'),
                    chr(92) || chr(10), ''), chr(1), chr(92)) END"""


class ProjectUnquoted(Function):
    """The text between the quotes of a string literal, whatever its prefix and quote style."""

    text: str

    expression = r"""
        CASE WHEN regexp_matches(text, '^[A-Za-z]*("{3}|''{3})')
             THEN regexp_extract(text, '(?s)^[A-Za-z]*(?:"{3}|''{3})(.*)(?:"{3}|''{3})$', 1)
             ELSE regexp_extract(text, '(?s)^[A-Za-z]*.(.*).$', 1) END"""


class ProjectIsPlainString(Function):
    """Whether a string literal is neither an f-string nor bytes, so a Python string value."""

    text: str

    expression = "NOT regexp_matches(text, '^[A-Za-z]*[fFbB]')"


class ProjectDigits(Function):
    """The digits and dots of the first `>=` clause of a version specifier (the whole of it when
    it has none)."""

    declaration: str

    expression = r"""regexp_replace(coalesce(list_filter(string_split(declaration, ','),
                                                         lambda part: contains(part, '>='))[1],
                                                declaration), '[^0-9.]', '', 'g')"""


class ProjectMinorText(Function):
    """What follows the major in a specifier's digits: after the first dot, or after a leading 3
    when there is none (`py312` reads 12)."""

    declaration: str

    expression = """CASE WHEN contains(project_digits(declaration), '.')
                         THEN substr(project_digits(declaration),
                                     strpos(project_digits(declaration), '.') + 1)
                         WHEN starts_with(project_digits(declaration), '3')
                         THEN substr(project_digits(declaration), 2) END"""


class ProjectMinor(Function):
    """The Python minor version a specifier accepts, NULL when it cannot be read as one or does
    not fit 32 bits, as in mcmr."""

    declaration: str

    expression = """CASE WHEN regexp_matches(project_minor_text(declaration), '^[0-9]+$')
                         THEN CAST(TRY_CAST(project_minor_text(declaration) AS UINTEGER) AS BIGINT)
                    END"""


class ProjectConfiguration(Family):
    family = "ProjectConfigurationFact"
    relations = {
        "facts": r"""
            SELECT 0 AS fact_order, 'configuration:pyproject' AS fact_id,
                   'pyproject.toml' AS path, 1 AS start_line, 0 AS start_column, 1 AS end_line,
                   0 AS end_column, 'python' AS language,
                   project_minor(r.declaration) AS python_target_project_minimum_minor
            FROM manifest_requirements r
        """,
        "candidates": r"""
            WITH python AS (
              SELECT fact_id, path FROM ast_files
              WHERE language = 'python' AND ends_with(path, '.py')
            ), statements AS (
              SELECT p.path, s.fact_id, s.node_id,
                     coalesce(o.type = 'class_definition' AND ends_with(o.name, 'Configuration'),
                              false) AS typed
              FROM python p
              JOIN ast_nodes s ON s.fact_id = p.fact_id AND s.type = 'expression_statement'
              LEFT JOIN ast_nodes b ON b.fact_id = s.fact_id AND b.node_id = s.parent_id
                AND b.type = 'block'
              LEFT JOIN ast_nodes o ON o.fact_id = b.fact_id AND o.node_id = b.parent_id
                AND o.type IN ('class_definition', 'function_definition')
              WHERE s.parent_id = 0 OR o.node_id IS NOT NULL
            )
            SELECT s.fact_id AS ast, s.path, a.node_id AS assignment, v.node_id AS collection,
                   n.name, s.typed,
                   CASE v.type WHEN 'list' THEN 'list' WHEN 'set' THEN 'set' ELSE 'tuple' END
                     AS collection_kind
            FROM statements s
            JOIN ast_nodes a ON a.fact_id = s.fact_id AND a.parent_id = s.node_id
              AND a.type = 'assignment'
            JOIN ast_nodes n ON n.fact_id = a.fact_id AND n.parent_id = a.node_id
              AND n.sibling_index = 0 AND n.type = 'identifier'
            JOIN ast_nodes e ON e.fact_id = a.fact_id AND e.parent_id = a.node_id AND e.type = '='
            JOIN ast_nodes v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
              AND v.sibling_index = e.sibling_index + 1
              AND v.type IN ('list', 'tuple', 'set', 'expression_list')
        """,
        "elements": r"""
            SELECT c.ast, c.assignment, e.sibling_index AS position,
                   count(s.node_id) AS pieces,
                   count(s.node_id) FILTER (WHERE project_is_plain_string(s.name)) AS plain,
                   string_agg(project_literal(s.name), '' ORDER BY s.sibling_index) AS value
            FROM project_configuration_candidates c
            JOIN ast_nodes e ON e.fact_id = c.ast AND e.parent_id = c.collection
              AND e.type NOT IN ('[', ']', '(', ')', '{', '}', ',', 'comment')
            LEFT JOIN ast_nodes s ON s.fact_id = c.ast AND s.type = 'string'
              AND (s.node_id = e.node_id
                   OR (s.parent_id = e.node_id AND e.type = 'concatenated_string'))
            GROUP BY c.ast, c.assignment, e.node_id, e.sibling_index
        """,
        "assignments": r"""
            SELECT f.fact_id, f.fact_id AS parent_id,
                   f.fact_id || '/assignments:' || c.ordinal AS record_id, c.ordinal,
                   c.name, c.collection_kind, c.typed AS is_typed_configuration_field,
                   c.values_length, c.ast, c.assignment
            FROM project_configuration_facts f
            JOIN (SELECT *, row_number() OVER (ORDER BY path, assignment) - 1 AS ordinal
                  FROM (SELECT c.*, count(e.position) AS values_length
                        FROM project_configuration_candidates c
                        LEFT JOIN project_configuration_elements e
                          ON e.ast = c.ast AND e.assignment = c.assignment
                        GROUP BY ALL
                        HAVING count(e.position) FILTER (WHERE e.plain < e.pieces
                                                         OR e.pieces = 0) = 0)) c ON true
        """,
        "values": r"""
            WITH configured AS (
              SELECT f.fact_id, t.tool, project_minor(t.declaration) AS minor
              FROM project_configuration_facts f, manifest_tools t
            ), per_file AS (
              SELECT f.fact_id, p.pattern AS key, project_minor(p.declaration) AS minor
              FROM project_configuration_facts f, manifest_file_targets p
            )
            SELECT fact_id, 'python_target.configured_tools' AS relation, fact_id AS parent_id,
                   row_number() OVER (PARTITION BY fact_id ORDER BY tool) - 1 AS ordinal,
                   NULL::VARCHAR AS map_key, tool AS string_value, NULL::BIGINT AS integer_value
            FROM configured
            UNION ALL
            SELECT fact_id, 'python_target.tool_target_minors', fact_id,
                   row_number() OVER (PARTITION BY fact_id ORDER BY tool) - 1, tool, NULL, minor
            FROM configured WHERE minor IS NOT NULL
            UNION ALL
            SELECT fact_id, 'python_target.per_file_target_minors', fact_id,
                   row_number() OVER (PARTITION BY fact_id ORDER BY key) - 1, NULL, NULL, minor
            FROM per_file WHERE minor IS NOT NULL
            UNION ALL
            SELECT a.fact_id, 'assignments.values', a.record_id, e.position, NULL, e.value, NULL
            FROM project_configuration_assignments a
            JOIN (SELECT ast, assignment, value,
                         row_number() OVER (PARTITION BY ast, assignment ORDER BY position) - 1
                           AS position
                  FROM project_configuration_elements) e
              ON e.ast = a.ast AND e.assignment = a.assignment
        """,
    }

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Load the manifest the relations read."""
        ProjectManifest(root=root).register(facts)
