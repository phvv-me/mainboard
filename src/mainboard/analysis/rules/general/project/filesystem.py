"""Rules over the directory tree (mcmr's ALL-FILE family)."""

import os

from .... import edits
from ....catalog import Example, Rule
from ....edits import Edit
from ....findings import Hit
from .relocation import respelled


class EmptyDirectories(Rule, id="ALL-FILE0001", applicability="safe"):
    """Directories left empty after the repository's Git ignores."""

    query = "SELECT * FROM directory_facts WHERE entry_count = 0"
    message = "`{fact_id}` satisfies empty directories"
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


class DirectoryModuleCount(Rule, id="ALL-FILE0003"):
    """Source files held directly by one directory, definition catalogs excepted. Modules sharing
    a name prefix there move into a folder of it (ALL-FILE0005); the rest need a person to say
    what groups them."""

    maximum: int = 5
    allow_definition_catalogs: bool = True

    query = """
        SELECT * FROM (
          SELECT *, CASE WHEN $allow_definition_catalogs AND is_definition_catalog THEN 0
                         ELSE direct_module_count END AS value
          FROM directory_facts)
        WHERE value > $maximum
    """
    message = "`{path}` holds {value} modules directly, past the ceiling of {maximum}"


class PrefixedModuleGroup(Rule, id="ALL-FILE0005", applicability="review"):
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
          SELECT path AS directory FROM directory_facts
          WHERE CASE WHEN $allow_definition_catalogs AND is_definition_catalog THEN 0
                     ELSE direct_module_count END > $maximum
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
                 regexp_replace(any_value(k.module), '\.[^.]+$', '') AS package,
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
        ), affected AS (
          SELECT g.directory, g.prefix, f.path, f.module, f.is_package, s.content
          FROM groups g
          JOIN core_file f ON f.language = 'python'
          JOIN source_files s ON s.path = f.path
          WHERE list_contains(g.modules, f.module)
             OR EXISTS (SELECT 1 FROM core_import i
                        WHERE i.file_id = f.file_id
                          AND (list_contains(g.modules, i.source_module)
                               OR list_contains(g.modules, i.target)))
        )
        SELECT g.path, 1 AS start_line, 0 AS start_column, g.directory, g.prefix, g.package,
               g.members, g.listed, coalesce(g.existing, []) AS existing,
               list({'path': a.path, 'module': a.module, 'package': a.is_package,
                     'content': a.content}) AS files
        FROM groups g
        JOIN affected a ON a.directory = g.directory AND a.prefix = g.prefix
        GROUP BY ALL
    """
    message = "`{directory}` is past its ceiling and `{listed}` share `{prefix}`, a folder's name"
    examples = (
        Example(
            bad="from .parts.a_one import One as One\nfrom .parts import a_two\n",
            good="from .parts.a.one import One as One\nfrom .parts.a import two as a_two\n",
            path="src/pkg/app.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
                "src/pkg/parts/__init__.py": "",
                "src/pkg/parts/a_one.py": "class One:\n    pass\n",
                "src/pkg/parts/a_two.py": "from .a_one import One\n\nTWO = One\n",
                "src/pkg/parts/b.py": "B = 1\n",
                "src/pkg/parts/c.py": "C = 1\n",
                "src/pkg/parts/d.py": "D = 1\n",
                "src/pkg/parts/e.py": "E = 1\n",
            },
        ),
        Example(
            bad="from .parts import a\nfrom .parts.a_one import One\n",
            good="from .parts.a import core as a\nfrom .parts.a.one import One\n",
            path="src/pkg/app.py",
            context={
                "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
                "src/pkg/__init__.py": "",
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
    def _folder_plan(hit: Hit) -> tuple[str, dict[str, str]]:
        """The folder the group moves into, and each member's old module mapped to its new one.

        The folder takes every word the members open with, one word left to each of them.
        """
        package = str(hit.package)
        members = {str(member["stem"]): str(member["shared"]) for member in hit.records("members")}
        worded = {stem: stem.removeprefix(shared).split("_") for stem, shared in members.items()}
        common = os.path.commonprefix(list(worded.values()))
        common = common[: min(len(words) for words in worded.values()) - 1] or common[:1]
        prefix = "_".join(common)
        moves = {
            f"{package}.{stem}": f"{package}.{prefix}.{members[stem]}"
            + ("_".join(words[len(common) :]) or "core")
            for stem, words in worded.items()
        }
        return prefix, moves

    def repair(self, hit: Hit) -> list[Edit] | None:
        """Move the group into its folder, then respell every import the move touches."""
        directory = str(hit.directory)
        prefix, moves = self._folder_plan(hit)
        names = [new.rsplit(".", 1)[1] for new in moves.values()]
        existing = {str(name) for name in hit.sequence("existing")}
        if len(set(names)) < len(names) or existing & set(names):
            return None
        made = respelled(hit, moves)
        if made is None:
            return None
        return made + [
            edits.rename(
                f"{directory}/{old.rsplit('.', 1)[1]}.py",
                target=f"{directory}/{prefix}/{new.rsplit('.', 1)[1]}.py",
            )
            for old, new in moves.items()
        ]


class DirectoryPathway(Rule, id="ALL-FILE0004"):
    """Directories that only lead to one child directory."""

    query = """
        SELECT * FROM directory_facts
        WHERE direct_directory_count = 1 AND direct_file_count = 0 AND source_depth > 1
          AND NOT (only_child_directory IN ('deterministic', 'contextual', 'external')
                   AND contains('/' || path || '/', '/rules/'))
    """
    message = "`{fact_id}` satisfies directory pathway"
