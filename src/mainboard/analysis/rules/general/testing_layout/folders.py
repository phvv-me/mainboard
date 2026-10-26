"""ALL-TEST0004: test files outside the place their ecosystem's layout gives them."""

from .... import edits
from ....catalog import Example, Rule
from ....edits import Edit
from ....engine.facts import Setting
from ....findings import Hit
from ..project.relocation import respelled
from .constants import ANCHOR

_PYPROJECT = """\
[project]
name = "pkg"
version = "0"

[tool.pytest.ini_options]
testpaths = ["tests"]
"""


def _table(targets: str) -> str:
    """The project's own `targets` setting for the rule, as a `[tool.mcmr.rules]` table."""
    return f'\n[tool.mcmr.rules."ALL-TEST0004"]\ntargets = {targets}\n'


_TEST = "def test_a_folder_holds_one_test_that_states_a_behavior():\n    pass\n"
_AMBIGUOUS = {
    "src/pkg/verbs/__init__.py": "",
    "src/pkg/verbs/lake.py": "VERBS = 1\n",
    "src/pkg/verbs/machines.py": "MACHINES = 1\n",
    "tests/constants.py": "X = 1\n",
}
_PACKAGE = {
    "pyproject.toml": _PYPROJECT,
    "src/pkg/__init__.py": "",
    "src/pkg/cli.py": "RUN = 1\n",
    "src/pkg/state/__init__.py": "",
    "src/pkg/state/lake.py": "LAKE = 1\n",
    "tests/gates/test_harness.py": "def test_the_harness_checks_itself():\n    pass\n",
    "tests/scenarios/test_flow.py": "def test_a_flow_spans_several_packages():\n    pass\n",
}
_IMPORTER = """\
from ..lake.fakes import Fake


def test_the_cli_runs_with_a_fake_lake():
    assert Fake
"""
_IMPORTER_FIXED = _IMPORTER.replace("..lake", "..state.lake")
_LAKE_TEST = """\
from ..constants import X


def test_the_constant_is_read_from_the_tests_root():
    assert X
"""
_STRAY = """\
from .constants import X


def test_a_stray_module_reads_the_shared_constant():
    assert X
"""


class TestFolderMirrorsSource(Rule, id="ALL-TEST0004", applicability="review"):
    """Test files outside the place their ecosystem's layout gives them: the tests of a source
    package sit where a stranger finds them by its path, and a layout that drifts hides which
    tests cover what. Each language has one layout, a setting a project overrides (empty turns it
    off): Python `mirror`, TypeScript `colocated`, Swift `target`, Rust `crate`.

    `mirror` reads the tests root from pytest's first `testpaths` entry (`tests` without one) and
    the source root from the distribution's package folder (`src/<name>`). The first folder of a
    test under the tests root must be a top-level package or module of the source root, or one of
    `allowed`; a test module directly in the tests root is a finding too. The fix moves the
    folder to the path of the one package or module anywhere under the source root that bears its
    name (a module directly in the tests root moves to `home`) and respells every import of the
    moved modules, in the moved files and their importers; with no unique match a person must
    choose the package, and records the choice once in `targets` (folder name to the path under
    the tests root it moves to, merging into a folder already there unless a file would collide;
    an empty path leaves that folder reported and unfixed on purpose). `colocated` wants a `*.test.ts` beside its `*.ts` (or under `__tests__/`
    mirroring the source folder), `target` a `Tests/<Target>Tests/` for each `Sources/<Target>/`
    of a Swift package, `crate` integration tests in the crate's `tests/` and a `tests.rs`
    declared `#[cfg(test)]`. Those three only flag, since moving a file there is a project's
    choice."""

    python: str = "mirror"
    typescript: str = "colocated"
    swift: str = "target"
    rust: str = "crate"
    allowed: tuple[str, ...] = ("scenarios", "gates")
    home: str = "scenarios"
    targets: dict[str, str] = {}

    query = r"""
        WITH files AS (
          SELECT entry FROM project_entries WHERE NOT is_directory
        ), dirs AS (
          SELECT path FROM directory_facts
        ), named AS (
          SELECT unnest($moved_names::VARCHAR[]) AS name, unnest($moved_paths::VARCHAR[]) AS path
        ), manifests AS (
          SELECT fact_id, regexp_replace(path, 'pyproject\.toml$', '') AS base
          FROM ast_files
          WHERE language = 'toml' AND regexp_matches(path, '(^|/)pyproject\.toml$')
        ), pairs AS (
          SELECT m.base, m.fact_id, t.name AS tbl, p.name AS key, p.node_id AS pair_id
          FROM manifests m
          JOIN ast_nodes t ON t.fact_id = m.fact_id AND t.type = 'table'
          JOIN ast_nodes p ON p.fact_id = m.fact_id AND p.parent_id = t.node_id AND p.type = 'pair'
        ), distribution AS (
          SELECT p.base, lower(regexp_replace(trim(v.name, '"'''), '[-_.]+', '_', 'g')) AS name
          FROM pairs p
          JOIN ast_nodes v
            ON v.fact_id = p.fact_id AND v.parent_id = p.pair_id AND v.sibling_index = 2
          WHERE p.tbl = 'project' AND p.key = 'name'
        ), testpaths AS (
          SELECT p.base,
                 regexp_replace(regexp_replace(trim(s.name, '"'''), '^\./', ''), '/+$', '') AS root
          FROM pairs p
          JOIN ast_nodes a
            ON a.fact_id = p.fact_id AND a.parent_id = p.pair_id AND a.type = 'array'
          JOIN ast_nodes s
            ON s.fact_id = p.fact_id AND s.parent_id = a.node_id AND s.type = 'string'
          WHERE p.tbl = 'tool.pytest.ini_options' AND p.key = 'testpaths'
          QUALIFY row_number() OVER (PARTITION BY p.base ORDER BY s.sibling_index) = 1
        ), projects AS (
          SELECT d.base, d.base || coalesce(t.root, 'tests') AS tests,
                 CASE WHEN EXISTS (SELECT 1 FROM core_file f WHERE f.language = 'python'
                                   AND starts_with(f.path, d.base || 'src/' || d.name || '/'))
                      THEN d.base || 'src/' || d.name
                      WHEN EXISTS (SELECT 1 FROM core_file f WHERE f.language = 'python'
                                   AND starts_with(f.path, d.base || d.name || '/'))
                      THEN d.base || d.name END AS source
          FROM distribution d LEFT JOIN testpaths t USING (base)
        ), members AS (
          SELECT DISTINCT base,
                 unnest(list_transform(generate_series(1, len(parts)),
                                       lambda i: array_to_string(parts[:i], '/'))) AS rel
          FROM (SELECT p.base,
                       string_split(regexp_replace(regexp_replace(substr(f.path, len(p.source) + 2),
                                                                  '(^|/)__init__\.pyi?$', ''),
                                                   '\.pyi?$', ''), '/') AS parts
                FROM projects p
                JOIN core_file f ON f.language = 'python' AND starts_with(f.path, p.source || '/'))
          WHERE parts[1] <> ''
        ), tops AS (
          SELECT base, rel AS name FROM members WHERE NOT contains(rel, '/')
        ), matches AS (
          SELECT base, regexp_extract(rel, '[^/]+$') AS leaf, list(rel ORDER BY rel) AS rels
          FROM members WHERE contains(rel, '/') GROUP BY ALL
        ), placed AS (
          SELECT p.*, f.path, f.module, f.is_package, regexp_extract(f.path, '[^/]+$') AS file,
                 split_part(substr(f.path, len(p.tests) + 2), '/', 1) AS first,
                 contains(substr(f.path, len(p.tests) + 2), '/') AS nested,
                 regexp_matches(f.path, '(^|/)(test_[^/]*|[^/]*_test)\.pyi?$') AS is_test_module
          FROM projects p
          JOIN core_file f ON f.language = 'python' AND starts_with(f.path, p.tests || '/')
          WHERE p.source IS NOT NULL AND $python = 'mirror'
        ), subjects AS (
          SELECT 'folder' AS kind, p.source, p.tests, p.first AS name, min(p.path) AS path,
                 p.tests || '/' || p.first AS moved, arg_min(p.module, p.path) AS module,
                 arg_min(p.is_package, p.path) AS is_package, m.rels, n.name IS NOT NULL AS named,
                 n.path AS chosen,
                 (SELECT count(*) > 0 FROM tops t
                  WHERE t.base = p.base AND t.name = split_part(n.path, '/', 1)) AS chosen_ok
          FROM placed p
          LEFT JOIN matches m ON m.base = p.base AND m.leaf = p.first
          LEFT JOIN named n ON n.name = p.first
          WHERE p.nested
            AND NOT EXISTS (SELECT 1 FROM tops t WHERE t.base = p.base AND t.name = p.first)
            AND NOT list_contains($allowed::VARCHAR[], p.first)
          GROUP BY p.base, p.source, p.tests, p.first, m.rels, n.name, n.path
          HAVING bool_or(p.is_test_module)
          UNION ALL
          SELECT 'module', p.source, p.tests, p.file, p.path, p.path, p.module, p.is_package,
                 NULL, FALSE, NULL, FALSE
          FROM placed p
          WHERE NOT p.nested AND p.is_test_module
        ), planned AS (
          SELECT s.*,
                 CASE WHEN s.kind = 'module' THEN s.tests || '/' || $home || '/' || s.name
                      WHEN s.named
                      THEN CASE WHEN s.chosen <> '' AND s.chosen_ok
                                THEN s.tests || '/' || s.chosen END
                      WHEN len(s.rels) = 1 THEN s.tests || '/' || s.rels[1] END AS wanted,
                 CASE WHEN s.kind = 'module' THEN [s.moved]
                      ELSE (SELECT list(f.entry ORDER BY f.entry) FROM files f
                            WHERE starts_with(f.entry, s.moved || '/')) END AS carried
          FROM subjects s
        ), staged AS (
          SELECT p.*,
                 list_transform(p.carried, lambda c: p.wanted || substr(c, len(p.moved) + 1))
                   AS arriving
          FROM planned p
        ), targeted AS (
          SELECT p.*,
                 CASE WHEN p.named
                      THEN CASE WHEN NOT EXISTS (SELECT 1 FROM files x
                                                 WHERE list_contains(p.arriving, x.entry))
                                THEN p.wanted END
                      WHEN NOT EXISTS (SELECT 1 FROM project_entries e
                                       WHERE e.entry = p.wanted
                                          OR starts_with(e.entry, p.wanted || '/'))
                      THEN p.wanted END AS target,
                 string_split(p.module, '.') AS parts
          FROM staged p
        ), modular AS (
          SELECT t.*,
                 CASE WHEN t.kind = 'module' THEN len(t.parts)
                      ELSE len(t.parts) - len(string_split(substr(t.path, len(t.moved) + 2), '/'))
                           + CAST(t.is_package AS INTEGER) END AS depth
          FROM targeted t
        ), renamed AS (
          SELECT m.*, array_to_string(m.parts[:m.depth], '.') AS folder_module,
                 array_to_string(string_split(regexp_replace(m.target, '\.pyi?$', ''), '/')[
                   len(string_split(regexp_replace(m.moved, '\.pyi?$', ''), '/')) - m.depth + 1:],
                   '.') AS target_module
          FROM modular m
        ), affected AS (
          SELECT r.moved, f.path, f.module, f.is_package, s.content,
                 (f.path = r.moved OR starts_with(f.path, r.moved || '/')) AS moving
          FROM renamed r
          JOIN core_file f ON f.language = 'python'
          JOIN source_files s ON s.path = f.path
          WHERE r.target IS NOT NULL
            AND (f.path = r.moved OR starts_with(f.path, r.moved || '/')
                 OR EXISTS (SELECT 1 FROM core_import i
                            WHERE i.file_id = f.file_id
                              AND (i.source_module = r.folder_module
                                   OR starts_with(i.source_module, r.folder_module || '.')
                                   OR i.target = r.folder_module
                                   OR starts_with(i.target, r.folder_module || '.'))))
        ), python AS (
          SELECT r.path, 1 AS start_line, 0 AS start_column, r.kind, r.moved AS source, r.target,
                 r.folder_module, r.target_module,
                 CASE
                   WHEN r.kind = 'module'
                   THEN format('`{}` is a test module directly in `{}`; it belongs in the folder '
                               || 'of the package it covers, or in `{}/{}` when it spans several',
                               r.moved, r.tests, r.tests, $home)
                   WHEN r.named AND r.chosen = ''
                   THEN format('`{}` is no package of `{}`; `targets` leaves it unresolved on '
                               || 'purpose', r.moved, r.source)
                   WHEN r.named AND NOT r.chosen_ok
                   THEN format('`{}` is no package of `{}`; `targets` sends it to `{}`, whose '
                               || 'first folder is no top-level package or module of it',
                               r.moved, r.source, r.chosen)
                   WHEN r.named AND r.target IS NOT NULL
                   THEN format('`{}` is no package of `{}`; `targets` moves it to `{}`',
                               r.moved, r.source, r.target)
                   WHEN r.named
                   THEN format('`{}` is no package of `{}`; `targets` sends it to `{}`, where a '
                               || 'file already has the name of one of its own',
                               r.moved, r.source, r.wanted)
                   WHEN r.target IS NOT NULL
                   THEN format('`{}` is no package of `{}`; `{}` is the only one named `{}`, so '
                               || 'these tests belong in `{}`',
                               r.moved, r.source, r.rels[1], r.name, r.target)
                   WHEN len(r.rels) = 1
                   THEN format('`{}` is no package of `{}`; `{}` is the only one named `{}` but '
                               || '`{}` is taken', r.moved, r.source, r.rels[1], r.name, r.wanted)
                   WHEN len(r.rels) > 1
                   THEN format('`{}` is no package of `{}`; choose the one these tests cover '
                               || 'among `{}`',
                               r.moved, r.source, array_to_string(r.rels, '`, `'))
                   ELSE format('`{}` is no package of `{}` and nothing there is named `{}`; '
                               || 'choose the package these tests cover',
                               r.moved, r.source, r.name)
                 END AS message,
                 r.carried,
                 list({'path': a.path, 'module': a.module, 'package': a.is_package,
                       'content': a.content, 'moving': a.moving}) FILTER (WHERE a.path IS NOT NULL)
                   AS files
          FROM renamed r LEFT JOIN affected a ON a.moved = r.moved
          GROUP BY ALL
        ), typescript AS (
          SELECT t.entry AS path,
                 format('`{}` has no `{}` beside it, nor under a `__tests__` folder mirroring '
                        || 'its source folder', t.entry, t.stem) AS message
          FROM (SELECT entry,
                       regexp_extract(entry, '^((?:.*/)?)[^/]*$', 1) AS folder,
                       regexp_replace(regexp_extract(entry, '[^/]+$'),
                                      '(\.(test|spec))?\.[cm]?[jt]sx?$', '') AS stem,
                       regexp_extract(entry, '^((?:.*?/)?)__tests__/', 1) AS before,
                       regexp_extract(entry, '__tests__/((?:.*/)?)[^/]*$', 1) AS after,
                       contains('/' || entry, '/__tests__/') AS guarded
                FROM files
                WHERE $typescript = 'colocated' AND regexp_matches(entry, '\.[cm]?[jt]sx?$')
                  AND NOT regexp_matches(entry, '\.d\.[cm]?ts$')
                  AND (regexp_matches(entry, '\.(test|spec)\.[cm]?[jt]sx?$')
                       OR contains('/' || entry, '/__tests__/'))) t
          WHERE NOT EXISTS (
            SELECT 1 FROM files s
            WHERE regexp_matches(s.entry, '\.[cm]?[jt]sx?$')
              AND NOT regexp_matches(s.entry, '\.(test|spec)\.')
              AND regexp_replace(s.entry, '\.[cm]?[jt]sx?$', '') IN (
                    CASE WHEN t.guarded THEN t.before || t.after || t.stem
                         ELSE t.folder || t.stem END,
                    CASE WHEN t.guarded THEN t.before || 'src/' || t.after || t.stem END))
        ), swift AS (
          SELECT t.entry AS path,
                 format('`{}` is outside `Tests/<Target>Tests/` for a `Sources/<Target>/` of its '
                        || 'package', t.entry) AS message
          FROM (SELECT entry, regexp_extract(entry, '^((?:.*?/)?)Tests/', 1) AS package,
                       regexp_extract(entry, '^(?:.*?/)?Tests/([^/]+)/', 1) AS target
                FROM files
                WHERE $swift = 'target' AND regexp_matches(entry, '(^|/)Tests/.*\.swift$')) t
          WHERE EXISTS (SELECT 1 FROM files p WHERE p.entry = t.package || 'Package.swift')
            AND NOT (len(t.target) > 5 AND ends_with(t.target, 'Tests')
                     AND EXISTS (SELECT 1 FROM dirs d
                                 WHERE d.path = t.package || 'Sources/'
                                                || regexp_replace(t.target, 'Tests$', '')))
        ), crates AS (
          SELECT regexp_replace(entry, 'Cargo\.toml$', '') AS dir
          FROM files WHERE regexp_matches(entry, '(^|/)Cargo\.toml$')
        ), rust AS (
          SELECT r.entry AS path,
                 CASE WHEN r.file = 'tests.rs'
                      THEN format('`{}` is a unit-test module its parent never declares '
                                  || '`#[cfg(test)]`', r.entry)
                      ELSE format('`{}` is an integration test outside the crate''s `tests/` '
                                  || 'folder', r.entry) END AS message
          FROM (SELECT f.entry, regexp_extract(f.entry, '[^/]+$') AS file,
                       regexp_extract(f.entry, '^((?:.*/)?)[^/]*$', 1) AS folder,
                       (SELECT max(c.dir) FROM crates c WHERE starts_with(f.entry, c.dir)) AS crate
                FROM files f WHERE $rust = 'crate' AND ends_with(f.entry, '.rs')) r
          WHERE r.crate IS NOT NULL AND NOT starts_with(r.entry, r.crate || 'tests/')
            AND (ends_with(r.file, '_test.rs')
                 OR (r.file = 'tests.rs' AND NOT EXISTS (
                      SELECT 1 FROM source_files s
                      WHERE list_contains([regexp_replace(r.folder, '/$', '') || '.rs',
                                           r.folder || 'mod.rs', r.folder || 'lib.rs',
                                           r.folder || 'main.rs'], s.path)
                        AND regexp_matches(s.content,
                              '#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*'
                              || '(?:pub(?:\([^)]*\))?\s+)?mod\s+tests\s*;'))))
        )
        SELECT path, start_line, start_column, message, kind, source, target, folder_module,
               target_module, carried, files
        FROM python
        UNION ALL
        SELECT path, 1, 0, message, 'file', path, NULL, NULL, NULL, CAST(NULL AS VARCHAR[]),
               CAST(NULL AS STRUCT(path VARCHAR, module VARCHAR, package BOOLEAN, content VARCHAR,
                                   moving BOOLEAN)[])
        FROM (SELECT * FROM typescript UNION ALL SELECT * FROM swift
              UNION ALL SELECT * FROM rust)
    """
    examples = (
        Example(
            bad=_IMPORTER,
            good=_IMPORTER_FIXED,
            path="tests/cli/test_run.py",
            context={
                **_PACKAGE,
                "tests/constants.py": "X = 1\n",
                "tests/lake/fakes.py": "class Fake:\n    pass\n",
                "tests/lake/test_core.py": _LAKE_TEST,
            },
        ),
        Example(
            bad=_STRAY,
            good=None,
            path="tests/test_stray.py",
            context={**_PACKAGE, "tests/constants.py": "X = 1\n"},
        ),
        Example(
            bad=_IMPORTER,
            good=_IMPORTER_FIXED,
            path="tests/cli/test_run.py",
            context={
                **_PACKAGE,
                **_AMBIGUOUS,
                "pyproject.toml": _PYPROJECT + _table('{ lake = "state/lake" }'),
                "tests/lake/fakes.py": "class Fake:\n    pass\n",
                "tests/lake/test_core.py": _LAKE_TEST,
                "tests/state/lake/test_existing.py": _TEST,
            },
        ),
        Example(
            bad=_TEST,
            path="tests/machines/test_binary_tools.py",
            context={
                **_PACKAGE,
                **_AMBIGUOUS,
                "pyproject.toml": _PYPROJECT + _table('{ machines = "" }'),
            },
        ),
        Example(
            bad=_TEST,
            path="tests/mixed/test_one.py",
            context={
                **_PACKAGE,
                "pyproject.toml": _PYPROJECT + _table('{ mixed = "nowhere/mixed" }'),
            },
        ),
        Example(
            bad="def test_a_folder_named_for_two_modules_is_left_to_a_person():\n    pass\n",
            path="tests/environments/test_one.py",
            context={
                **_PACKAGE,
                "src/pkg/fleet/__init__.py": "",
                "src/pkg/fleet/environments.py": "ENVIRONMENTS = 1\n",
                "src/pkg/state/environments.py": "ENVIRONMENTS = 2\n",
            },
        ),
        Example(
            bad='it("works", () => {});\n',
            path="web/b.test.ts",
            context={**ANCHOR, "web/a.ts": "export const a = 1;\n", "web/a.test.ts": "it();\n"},
        ),
        Example(
            bad="import XCTest\n",
            path="Tests/Stray/StrayTests.swift",
            context={
                **ANCHOR,
                "Package.swift": "// swift-tools-version: 6.0\n",
                "Sources/Foo/Foo.swift": "public struct Foo {}\n",
                "Tests/FooTests/FooTests.swift": "import XCTest\n",
            },
        ),
        Example(
            bad="#[test]\nfn it_adds() {}\n",
            path="src/util_test.rs",
            context={
                **ANCHOR,
                "Cargo.toml": '[package]\nname = "rs"\nversion = "0.1.0"\n',
                "src/lib.rs": "pub fn add() {}\n",
                "tests/it.rs": "#[test]\nfn it_adds() {}\n",
            },
        ),
    )

    @staticmethod
    def _moves(hit: Hit) -> dict[str, str]:
        """Each module the move carries and each folder module on the way, to its new name."""
        old, new = str(hit.folder_module), str(hit.target_module)
        carried: set[str] = set()
        for file in hit.records("files"):
            parts = str(file["module"]).split(".")
            if file["moving"]:
                carried.update(
                    ".".join(parts[:depth]) for depth in range(old.count(".") + 1, len(parts) + 1)
                )
        return {module: new + module.removeprefix(old) for module in carried}

    def parameters(self) -> dict[str, Setting]:
        """The settings, with `targets` as the two lists the query reads: folder names and paths."""
        return super().parameters() | {
            "moved_names": list(self.targets),
            "moved_paths": list(self.targets.values()),
        }

    def repair(self, hit: Hit) -> list[Edit] | None:
        """Move every file of the folder (or the module), then respell the imports of what moved.

        The engine proves a file at a time, so a directory is moved as its files.
        """
        if hit.target is None:
            return None
        made = respelled(hit, self._moves(hit))
        if made is None:
            return None
        source, target = str(hit.source), str(hit.target)
        return [
            *made,
            *(
                edits.rename(str(path), target=target + str(path).removeprefix(source))
                for path in hit.sequence("carried")
            ),
        ]
