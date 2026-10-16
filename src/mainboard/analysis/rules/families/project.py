"""mcmr's DirectoryFact, derived from the project's tree (`project_entries`: every file and
directory the project's `.gitignore` files keep).

A directory's source depth is measured from its source root: the nearest ancestor that is a `src`
directory or the parent of a top-level Python package (a directory holding `__init__.py` below no
other). mcmr also treats a same-named directory in another branch as a namespace portion of such a
package, but its `Packages::of` mutates a set while iterating it, so the depth of those directories
depends on iteration order and differs between runs; the derived depth never counts them.
"""

import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs git with fixed argv since=2026-10-15
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from patos import FrozenModel

from ...catalog import Family, Function
from ...facts import Facts, lint_excludes

NOT_A_REPOSITORY = b"not a git repository"
GIT = ["git", "-c", "core.excludesFile=/dev/null"]


class ProjectTree(FrozenModel):
    """The files and directories under `root` that the project's `.gitignore` files keep and the
    workspace's `[lint]` excludes leave in: symbolic links and `.git` are left out and the root
    itself is no entry.

    Git decides every verdict, with the user's global excludes file left out so every machine
    judges the same tree. Ignores are the patterns alone, as in mcmr: `git check-ignore
    --no-index` does not consult the index, so a tracked file in an ignored directory is ignored
    and a path inside a submodule is judged by the same patterns (without it git stops on the
    first such path with "Pathspec is in submodule", which used to drop the ignores of every later
    entry at that tree level). Outside any repository git reads the `.gitignore` files of the tree
    through an empty bare repository whose work tree is `root`, matching case as mcmr does there
    (`git init` on macOS would set `core.ignorecase`); a reimplementation of its pattern language
    (pathspec) disagrees on `X/**`, which git does not let match the directory X itself.
    """

    root: Path

    def register(self, facts: Facts) -> None:
        """Create `project_entries` (entry, is_directory) in `facts`, once."""
        if facts.db.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = 'project_entries'"
        ).fetchone():
            return
        facts.db.execute("CREATE TABLE project_entries (entry VARCHAR, is_directory BOOLEAN)")
        if entries := self.entries():
            facts.db.executemany("INSERT INTO project_entries VALUES (?, ?)", entries)

    def entries(self) -> list[tuple[str, bool]]:
        """Every kept entry, a level at a time so git judges only what no ignored directory
        above already hides."""
        found: list[tuple[str, bool]] = []
        excluded = lint_excludes(self.root)
        level = [""]
        with self._git() as git:
            while level:
                children = [
                    (prefix + child.name, child.is_dir(follow_symlinks=False))
                    for prefix in level
                    for child in sorted(os.scandir(self.root / prefix), key=lambda c: c.name)
                    if child.name != ".git" and not child.is_symlink()
                ]
                ignored = self._ignored(git, [entry for entry, _ in children])
                kept = [
                    (entry, is_dir)
                    for entry, is_dir in children
                    if entry not in ignored and not excluded(f"{entry}/" if is_dir else entry)
                ]
                found += kept
                level = [entry + "/" for entry, is_dir in kept if is_dir]
        return found

    @contextmanager
    def _git(self) -> Generator[list[str]]:
        """The git command that judges `root`: the repository holding it, else a throwaway empty
        bare repository removed on exit."""
        git = [*GIT, "-C", str(self.root)]
        probe = subprocess.run([*git, "rev-parse", "--git-dir"], capture_output=True, check=False)
        if not probe.returncode:
            yield git
        elif NOT_A_REPOSITORY not in probe.stderr:
            raise RuntimeError(f"git in {self.root}: {probe.stderr.decode()}")
        else:
            root = str(self.root)
            with tempfile.TemporaryDirectory() as bare:
                init = [*GIT, "init", "--bare", "-q", bare]
                subprocess.run(init, capture_output=True, check=True)
                yield [*git, "-c", "core.ignorecase=false", "--git-dir", bare, "--work-tree", root]

    def _ignored(self, git: list[str], names: list[str]) -> set[str]:
        """The entries the ignore patterns exclude. Git is given the paths as `./name` (a name such
        as `:!x` would be read as pathspec magic, which `check-ignore` refuses) and looks at the
        file system for what each is: with a trailing slash `data/*` would match `data/` itself."""
        run = subprocess.run(
            [*git, "check-ignore", "--no-index", "-z", "--stdin"],
            input="\0".join(f"./{name}" for name in names).encode(),
            capture_output=True,
            check=False,
        )
        if run.returncode > 1:
            raise RuntimeError(f"git check-ignore in {self.root}: {run.stderr.decode()}")
        paths = run.stdout.decode().split("\0")
        return {path.removeprefix("./").rstrip("/") for path in paths if path}


class ProjectParent(Function):
    """The directory holding `entry`, empty for a top-level entry."""

    entry: str

    expression = r"regexp_extract(entry, '^(.*)/[^/]*$', 1)"


class ProjectName(Function):
    """The last component of `entry`."""

    entry: str

    expression = r"regexp_extract(entry, '([^/]*)$', 1)"


class ProjectIsSource(Function):
    """Whether `entry` is a file mcmr reads as source: the suffixes of its Python, Rust,
    TypeScript, C, C++ and CUDA frontends."""

    entry: str

    expression = (
        "regexp_matches(entry, "
        "'[.](py|pyi|rs|ts|tsx|c|h|cc|cpp|cxx|hpp|hh|hxx|inl|ipp|tpp|cu|cuh)$')"
    )


class ProjectDepth(Function):
    """How many directory levels `directory` sits below `root`, its source root."""

    directory: str
    root: str

    expression = (
        "len(list_filter(string_split(substr(directory, length(root) + 1), '/'), "
        "lambda part: part <> ''))"
    )


class Directory(Family):
    family = "DirectoryFact"
    verified = True
    relations = {
        "facts": r"""
            WITH directories AS (
              SELECT '' AS directory UNION ALL
              SELECT entry FROM project_entries WHERE is_directory
            ), held AS (
              SELECT project_parent(entry) AS directory,
                     count(*) AS entry_count,
                     count(*) FILTER (WHERE NOT is_directory
                                      AND project_name(entry) <> '__init__.py') AS file_count,
                     count(*) FILTER (WHERE is_directory) AS directory_count,
                     any_value(project_name(entry)) FILTER (WHERE is_directory) AS child,
                     count(*) FILTER (WHERE NOT is_directory AND project_is_source(entry)
                                      AND project_name(entry) <> '__init__.py') AS module_count
              FROM project_entries GROUP BY directory
            ), packages AS (
              SELECT project_parent(entry) AS directory FROM project_entries
              WHERE NOT is_directory AND project_name(entry) = '__init__.py'
            ), documents AS (
              SELECT DISTINCT unnest(list_transform(
                       range(1, length(entry) - length(replace(entry, '/', '')) + 1),
                       lambda i: array_to_string(string_split(entry, '/')[:i], '/'))) AS directory
              FROM project_entries WHERE NOT is_directory AND project_is_source(entry)
            ), names AS (
              SELECT DISTINCT project_name(p.directory) AS name FROM packages p
              WHERE p.directory <> '' AND NOT EXISTS (
                SELECT 1 FROM packages a WHERE a.directory <> '' AND a.directory <> p.directory
                  AND starts_with(p.directory, a.directory || '/'))
            ), spaces AS (
              SELECT directory FROM packages WHERE directory <> ''
              UNION
              SELECT d.directory FROM documents d
              JOIN names n ON n.name = project_name(d.directory)
              WHERE NOT EXISTS (SELECT 1 FROM packages p WHERE p.directory <> ''
                AND (p.directory = d.directory OR starts_with(p.directory, d.directory || '/'))
                AND project_name(p.directory) = n.name)
            ), roots AS (
              SELECT directory AS root FROM directories
              WHERE directory = 'src' OR ends_with(directory, '/src')
              UNION
              SELECT project_parent(s.directory) FROM spaces s
              WHERE NOT EXISTS (SELECT 1 FROM spaces a WHERE a.directory <> s.directory
                                AND starts_with(s.directory, a.directory || '/'))
                AND NOT EXISTS (SELECT 1 FROM packages WHERE directory = '')
              UNION
              SELECT '' FROM packages WHERE directory = ''
            ), catalogs AS (
              SELECT project_parent(path) AS directory FROM module_facts
              WHERE NOT is_package_initializer
              GROUP BY directory HAVING bool_and(class_count + function_count = 1)
            ), placed AS (
              SELECT d.directory, arg_max(r.root, length(r.root)) AS root
              FROM directories d LEFT JOIN roots r
                ON r.root = '' OR d.directory = r.root OR starts_with(d.directory, r.root || '/')
              GROUP BY d.directory
            )
            SELECT row_number() OVER (ORDER BY d.directory) - 1 AS fact_order,
                   'directory:' || d.label AS fact_id, d.label AS path, 1 AS start_line,
                   0 AS start_column, 1 AS end_line, 0 AS end_column, NULL::VARCHAR AS language,
                   coalesce(h.directory_count, 0) AS direct_directory_count,
                   coalesce(h.file_count, 0) AS direct_file_count,
                   coalesce(h.module_count, 0) AS direct_module_count,
                   coalesce(h.entry_count, 0) AS entry_count, 0 AS evidence_length,
                   true AS evidence_present, c.directory IS NOT NULL AS is_definition_catalog,
                   CASE WHEN h.directory_count = 1 THEN h.child END AS only_child_directory,
                   project_depth(d.directory, coalesce(p.root, '')) AS source_depth
            FROM (SELECT directory, coalesce(nullif(directory, ''), '.') AS label
                  FROM directories) d
            LEFT JOIN held h USING (directory) LEFT JOIN catalogs c USING (directory)
            LEFT JOIN placed p USING (directory)
            ORDER BY d.directory
        """,
    }

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Load the tree the relation reads."""
        ProjectTree(root=root).register(facts)
