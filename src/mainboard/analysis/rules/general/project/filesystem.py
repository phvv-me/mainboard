"""Rules over the directory tree (mcmr's ALL-FILE family)."""

import os

from ....catalog import Example, Relation, Rule
from ....edits import Edit
from ....findings import Hit
from ....operations import Operation

_PROJECT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}

_PARTS = {
    **_PROJECT,
    "src/pkg/parts/__init__.py": "",
    "src/pkg/parts/a_one.py": "class One:\n    pass\n",
    "src/pkg/parts/a_two.py": "from .a_one import One\n\nTWO = One\n",
    "src/pkg/parts/b.py": "B = 1\n",
    "src/pkg/parts/c.py": "C = 1\n",
    "src/pkg/parts/d.py": "D = 1\n",
    "src/pkg/parts/e.py": "E = 1\n",
}


class Ceiling(Rule):
    """A count held at or under `maximum`: a subclass's query yields what passes it."""

    maximum: int


class CatalogDirectories(Relation):
    """The `catalog_directories()` relation: the directories whose modules no ceiling charges.

    A definition catalog's modules define one class or function each. A plugin folder's define
    only classes that an ancestor's `__init_subclass__` registers on definition (rules,
    relations, backends), so one more plugin is one more module there.
    """

    stored = True
    query = """
        WITH RECURSIVE hooked AS (
          SELECT class_id FROM class_methods
          WHERE name IN ('__init_subclass__', '__pydantic_init_subclass__')
          UNION
          SELECT e.child FROM override_edges e JOIN hooked h ON e.parent = h.class_id
        ), classes AS (
          SELECT c.path, count(*) AS defined, count(h.class_id) AS registered
          FROM class_classes c LEFT JOIN hooked h USING (class_id)
          WHERE c.scope = 'module' GROUP BY c.path
        ), plugins AS (
          SELECT coalesce(nullif(project_parent(m.path), ''), '.') AS directory
          FROM module_facts m LEFT JOIN classes c USING (path)
          WHERE NOT m.is_package_initializer
          GROUP BY directory
          HAVING bool_and(coalesce(c.defined > 0 AND c.defined = c.registered, false)
                          AND m.function_count = 0)
        )
        SELECT directory FROM plugins
        UNION
        SELECT path AS directory FROM directory_facts WHERE is_definition_catalog
    """


class PytestRoots(Relation):
    """The `pytest_roots()` relation: the folder each `pyproject.toml` has pytest collect from.

    That is the first `testpaths` entry of `[tool.pytest.ini_options]` (`tests` without one), as
    the `path` from the project root.
    """

    query = r"""
        WITH manifests AS (
          SELECT fact_id, regexp_replace(path, 'pyproject\.toml$', '') AS base
          FROM ast_files
          WHERE language = 'toml' AND regexp_matches(path, '(^|/)pyproject\.toml$')
        ), entries AS (
          SELECT m.base, regexp_replace(regexp_replace(trim(s.name, '"'''), '^\./', ''), '/+$', '')
                           AS root
          FROM manifests m
          JOIN ast_nodes t ON t.fact_id = m.fact_id AND t.type = 'table'
            AND t.name = 'tool.pytest.ini_options'
          JOIN ast_nodes p ON p.fact_id = m.fact_id AND p.parent_id = t.node_id
            AND p.type = 'pair' AND p.name = 'testpaths'
          JOIN ast_nodes a ON a.fact_id = m.fact_id AND a.parent_id = p.node_id
            AND a.type = 'array'
          JOIN ast_nodes s ON s.fact_id = m.fact_id AND s.parent_id = a.node_id
            AND s.type = 'string'
          QUALIFY row_number() OVER (PARTITION BY m.base ORDER BY s.sibling_index) = 1
        )
        SELECT m.base || coalesce(e.root, 'tests') AS path
        FROM manifests m LEFT JOIN entries e USING (base)
    """


class EmptyDirectories(Rule, id="ALL-FILE0001", applicability="safe"):
    """Directories left empty after the repository's Git ignores."""

    query = "SELECT * FROM directory_facts WHERE entry_count = 0"
    message = (
        "`{fact_id}` is a directory that holds no entry once the repository's Git ignores apply"
    )
    fix = "SELECT path, 'remove' AS operation FROM finding"


class PackageDepth(Rule, id="ALL-FILE0002"):
    """Directory levels between a directory and the source root above it, the same ceiling
    PY-IMPO0005 sets on the modules an import names."""

    maximum: int = 6
    framework_roots: tuple[str, ...] = ()

    query = """
        SELECT *, source_depth AS value FROM directory_facts
        WHERE source_depth > $maximum AND NOT EXISTS (
          SELECT 1 FROM unnest($framework_roots::VARCHAR[]) AS r(root)
          WHERE path = root OR starts_with(path, rtrim(root, '/') || '/'))
    """
    message = "package depth is {value} for `{fact_id}`"


class DirectoryModuleCount(Ceiling, id="ALL-FILE0003"):
    """Source files held directly by one directory, definition catalogs and plugin folders
    excepted. A shared `constants.py`, `enums.py` or `exceptions.py` holds definitions and is not
    counted. Modules sharing a name prefix there move into a folder of it (ALL-FILE0005); the
    rest need a person to say what groups them."""

    maximum: int = 5
    allow_definition_catalogs: bool = True

    query = """
        SELECT * FROM (
          SELECT d.*, CASE WHEN $allow_definition_catalogs AND c.directory IS NOT NULL THEN 0
                           ELSE d.direct_module_count END AS value
          FROM directory_facts d LEFT JOIN catalog_directories() c ON c.directory = d.path)
        WHERE value > $maximum
    """
    message = "`{path}` holds {value} modules directly, past the ceiling of {maximum}"


class PrefixedModuleGroup(Ceiling, id="ALL-FILE0005", applicability="review"):
    """Python modules of a directory past ALL-FILE0003's ceiling that share a name prefix
    (`cuda_base`, `cuda_flow`), which a folder of that prefix (`cuda/base.py`, `cuda/flow.py`)
    would hold. A prefix most modules of the directory open with groups nothing: the next word
    does, and each moved module keeps its own (`test_apple_fonts` becomes `apple/test_fonts`,
    `reverse_fixtures` beside it `reverse/fixtures`). `test_` is always such a prefix, however
    few modules carry it, since pytest collects a test module by it: three of six once grouped as
    `test/clouds.py` and the like, which nothing collected any more. The module the prefix
    names (`classes.py` beside `classes_graph.py`) moves with its group as the folder's `core.py`
    (`test_core.py` for `test_apple.py`), and a folder of that name already there takes the group,
    unless a name would collide.

    The fix moves the group and respells every import of it, in the moved modules and in their
    importers alike."""

    maximum: int = 5
    allow_definition_catalogs: bool = True

    query = r"""
        WITH over AS (
          SELECT d.path AS directory FROM directory_facts d
          LEFT JOIN catalog_directories() c ON c.directory = d.path
          WHERE CASE WHEN $allow_definition_catalogs AND c.directory IS NOT NULL THEN 0
                     ELSE d.direct_module_count END > $maximum
        ), modules AS (
          SELECT o.directory, f.path, f.module, regexp_extract(f.path, '([^/]+)\.py$', 1) AS stem
          FROM over o JOIN core_file f ON f.language = 'python' AND NOT f.is_package
            AND regexp_replace(f.path, '/[^/]+$', '') = o.directory
        ), worded AS (
          SELECT *, CASE WHEN starts_with(stem, 'test_') THEN 'test_'
                         WHEN 2 * count(*) OVER (PARTITION BY directory, split_part(stem, '_', 1))
                                > count(*) OVER (PARTITION BY directory)
                          AND contains(stem, '_')
                         THEN split_part(stem, '_', 1) || '_' ELSE '' END AS shared
          FROM modules
        ), keyed AS (
          SELECT *, split_part(substr(stem, len(shared) + 1), '_', 1) AS prefix,
                 contains(substr(stem, len(shared) + 1), '_') AS prefixed
          FROM worded
        ), groups AS (
          SELECT k.directory, k.prefix,
                 min(k.path) AS path,
                 list({'stem': k.stem, 'shared': k.shared} ORDER BY k.stem) AS members,
                 array_to_string(list(k.stem ORDER BY k.stem), ', ') AS listed,
                 list(k.module) AS modules,
                 (SELECT list(regexp_extract(f.path, '([^/]+)\.py$', 1))
                  FROM core_file f
                  WHERE regexp_replace(f.path, '/[^/]+$', '') = k.directory || '/' || k.prefix)
                   AS existing
          FROM keyed k
          GROUP BY k.directory, k.prefix
          HAVING count(*) FILTER (WHERE k.prefixed) > 1
        )
        SELECT g.path, 1 AS start_line, 0 AS start_column, g.directory, g.prefix,
               g.members, g.listed, coalesce(g.existing, []) AS existing, r.files
        FROM groups g, module_readers(g.modules) r
    """
    message = "`{directory}` is past its ceiling and `{listed}` share `{prefix}`, a folder's name"
    examples = (
        Example(
            bad="import pkg.parts.a_one\n\nX = pkg.parts.a_one.One\n",
            good="import pkg.parts.a.one\n\nX = pkg.parts.a.one.One\n",
            path="src/pkg/app.py",
            context=_PARTS,
        ),
        Example(
            bad="from .parts.a_one import One as One\nfrom .parts import a_two\n",
            good="from .parts.a.one import One as One\nfrom .parts.a import two as a_two\n",
            path="src/pkg/app.py",
            context=_PARTS,
        ),
        Example(
            bad="from .parts import a\nfrom .parts.a_one import One\n",
            good="from .parts.a import core as a\nfrom .parts.a.one import One\n",
            path="src/pkg/app.py",
            context={
                **_PROJECT,
                "src/pkg/parts/__init__.py": "",
                "src/pkg/parts/a.py": "from .a_one import One\n\nROOT = One\n",
                "src/pkg/parts/a_one.py": "class One:\n    pass\n",
                "src/pkg/parts/a_two.py": "from . import a\n\nTWO = a.ROOT\n",
                "src/pkg/parts/b.py": "B = 1\n",
                "src/pkg/parts/c.py": "C = 1\n",
                "src/pkg/parts/d.py": "D = 1\n",
            },
        ),
    )

    @staticmethod
    def _folder_plan(hit: Hit) -> tuple[str, list[dict[str, str]]]:
        """The folder the group moves into, and each member's file moving into it as a `move`.

        The folder takes every word the members open with, one word left to each of them.
        """
        directory = str(hit.directory)
        members = {str(member["stem"]): str(member["shared"]) for member in hit.records("members")}
        worded = {stem: stem.removeprefix(shared).split("_") for stem, shared in members.items()}
        common = os.path.commonprefix(list(worded.values()))
        common = common[: min(len(words) for words in worded.values()) - 1] or common[:1]
        prefix = "_".join(common)
        moves = [
            {
                "path": f"{directory}/{stem}.py",
                "target": f"{directory}/{prefix}/{members[stem]}"
                + ("_".join(words[len(common) :]) or "core")
                + ".py",
            }
            for stem, words in worded.items()
        ]
        return prefix, moves

    def repair(self, hit: Hit) -> list[Edit] | None:
        """Move the group into its folder, then respell every import the move touches.

        The operation is named, as a SQL fix names it, so the rule imports none.
        """
        prefix, moves = self._folder_plan(hit)
        held = [f"{hit.directory}/{prefix}/{name}.py" for name in hit.sequence("existing")]
        row = {
            "operation": "move-modules",
            "moves": moves,
            "files": hit.records("files"),
            "held": held,
        }
        return Operation.rendered(row)


class DirectoryPathway(Rule, id="ALL-FILE0004", applicability="review"):
    """Directories that only lead to one child directory.

    The fix dissolves the directory: the child takes its place, every file moved along and every
    import of the modules among them following (`engines/compile/ecosystems/` becomes
    `engines/ecosystems/`). A directory right under the tests root is the package a test folder
    covers, so it keeps its name and takes the child's files instead. It leaves the directory as
    it is when a file would land on another, the directory's package initializer holds code, a
    module imports the directory itself (nothing is left to import) or a file that is no Python
    names a moved module or path."""

    query = """
        SELECT d.*, CASE WHEN project_name(d.path) <> d.only_child_directory
                          AND NOT EXISTS (SELECT 1 FROM pytest_roots() r
                                          WHERE r.path = project_parent(d.path))
                         THEN project_parent(d.path) || '/' || d.only_child_directory
                         ELSE d.path END AS destination
        FROM directory_facts d
        WHERE d.direct_directory_count = 1 AND d.direct_file_count = 0 AND d.source_depth > 1
          AND NOT (d.only_child_directory IN ('deterministic', 'contextual', 'external')
                   AND contains('/' || d.path || '/', '/rules/'))
    """
    message = "`{fact_id}` holds nothing but one child directory"
    change = "moved the files of `{path}/{only_child_directory}` to `{destination}`"
    fix = r"""
        WITH planned AS (
          SELECT f.finding_key, f.path AS pathway, f.path || '/' || f.only_child_directory AS source,
                 f.destination, f.destination <> f.path AS dissolving
          FROM finding f
        ), moved AS (
          SELECT p.finding_key, e.entry AS path,
                 p.destination || substr(e.entry, length(p.source) + 1) AS target
          FROM planned p
          JOIN project_entries e ON NOT e.is_directory AND starts_with(e.entry, p.source || '/')
        ), modules AS (
          SELECT m.finding_key, list(c.module) AS modules,
                 any_value(substr(c.module, 1, length(c.module) - 1 - length(replace(
                   regexp_replace(substr(m.path, length(p.pathway) + 2), '(/__init__)?\.pyi?$', ''),
                   '/', '.')))) AS pathway_module
          FROM moved m
          JOIN planned p USING (finding_key)
          JOIN core_file c ON c.path = m.path AND c.language = 'python'
          GROUP BY m.finding_key
        ), named AS (
          SELECT p.finding_key, s.path
          FROM planned p, source_files s
          WHERE NOT regexp_matches(s.path, '\.pyi?$')
            AND (contains(s.content, CASE WHEN p.dissolving THEN p.pathway ELSE p.source END)
                 OR EXISTS (SELECT 1 FROM moved m JOIN core_file c ON c.path = m.path
                            WHERE m.finding_key = p.finding_key AND contains(s.content, c.module)))
          UNION ALL
          SELECT d.finding_key, i.path
          FROM modules d JOIN planned p USING (finding_key), core_import i
          WHERE p.dissolving AND i.is_project_owned
            AND (i.target = d.pathway_module OR i.source_module = d.pathway_module
                 OR (i.statement_type = 'import_statement' AND i.module = d.pathway_module))
        )
        SELECT p.finding_key, 'move-modules' AS operation,
               (SELECT list({'path': m.path, 'target': m.target} ORDER BY m.path)
                FROM moved m WHERE m.finding_key = p.finding_key) AS moves,
               r.files,
               (SELECT list(e.entry) FROM project_entries e
                WHERE NOT e.is_directory AND starts_with(e.entry, p.destination || '/')) AS held,
               CASE WHEN p.dissolving THEN [p.pathway] ELSE [] END AS removed,
               coalesce((SELECT list(DISTINCT n.path) FROM named n
                         WHERE n.finding_key = p.finding_key), []) AS mentions
        FROM planned p JOIN modules d USING (finding_key), module_readers(d.modules) r
        WHERE NOT p.dissolving OR NOT EXISTS (
          SELECT 1 FROM source_files s
          WHERE is_initializer_path(s.path) AND project_parent(s.path) = p.pathway
            AND trim(s.content) <> '')
    """
    examples = (
        Example(
            bad="from .outer.inner.a import A\n",
            good="from .inner.a import A\n",
            path="src/pkg/app.py",
            context={**_PROJECT, "src/pkg/outer/inner/a.py": "A = 1\n"},
        ),
        Example(
            bad="import pkg.outer.inner\n\nB = pkg.outer.inner.a.A\n",
            good="import pkg.inner\n\nB = pkg.inner.a.A\n",
            path="src/pkg/app.py",
            context={
                **_PROJECT,
                "src/pkg/outer/__init__.py": "",
                "src/pkg/outer/inner/__init__.py": "from . import a\n",
                "src/pkg/outer/inner/a.py": "A = 1\n",
            },
        ),
        Example(
            bad="from .outer.inner.a import A\n",
            good="from .outer.a import A\n",
            path="tests/app.py",
            context={**_PROJECT, "tests/outer/inner/a.py": "A = 1\n"},
        ),
        Example(
            bad="from .outer.inner.a import A\n",
            good="from .outer.inner.a import A\n",
            path="src/pkg/app.py",
            context={
                **_PROJECT,
                "src/pkg/outer/__init__.py": "OUTER = 1\n",
                "src/pkg/outer/inner/a.py": "A = 1\n",
            },
        ),
        Example(
            bad="from .outer.inner.a import A\n",
            good="from .outer.inner.a import A\n",
            path="src/pkg/app.py",
            context={
                **_PROJECT,
                "src/pkg/outer/__init__.pyi": "OUTER: int\n",
                "src/pkg/outer/inner/a.py": "A = 1\n",
            },
        ),
        Example(
            bad="from .outer import inner\n",
            good="from .outer import inner\n",
            path="src/pkg/app.py",
            context={
                **_PROJECT,
                "src/pkg/outer/__init__.py": "",
                "src/pkg/outer/inner/__init__.py": "",
            },
        ),
        Example(
            bad="from .outer.inner.a import A\n",
            good="from .outer.inner.a import A\n",
            path="src/pkg/app.py",
            context={
                **_PROJECT,
                "src/pkg/inner/a.py": "A = 2\n",
                "src/pkg/outer/inner/a.py": "A = 1\n",
            },
        ),
        Example(
            bad="from .outer.inner.a import A\n",
            good="from .outer.inner.a import A\n",
            path="src/pkg/app.py",
            context={
                **_PROJECT,
                "pyproject.toml": _PROJECT["pyproject.toml"] + "# src/pkg/outer\n",
                "src/pkg/outer/inner/a.py": "A = 1\n",
            },
        ),
    )
