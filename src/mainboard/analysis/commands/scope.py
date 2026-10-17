"""The units of a package and the imports between them, for `matrix` and `diagram`.

A directory is read as a package whose units are its children: each subpackage (with everything
below it) and each module. A dependency is a pair of files, one importing the other through an
`import` statement (the initializers a parent package runs first are not counted, or every package
would depend on itself).
"""

from pathlib import Path

from patos import FrozenModel

from ...core.errors import MissionError
from ...git.process import Git
from .workspace import Workspace

_UNIT = r"regexp_replace(split_part(substr(path, $skip), '/', 1), '\.py$', '')"

_COMPONENTS = """
    WITH RECURSIVE reach(src, dst) AS (
      SELECT src, dst FROM {edges}
      UNION
      SELECT r.src, e.dst FROM reach r JOIN {edges} e ON e.src = r.dst
    ), mutual AS (
      SELECT a.src AS node, a.dst AS peer
      FROM reach a JOIN reach b ON b.src = a.dst AND b.dst = a.src
    )
    SELECT n.node, least(n.node, coalesce(min(m.peer), n.node)) AS component,
           count(m.peer) > 0 AS cyclic,
           (SELECT count(*) FROM reach r WHERE r.src = n.node) AS depends
    FROM {nodes} n LEFT JOIN mutual m ON m.node = n.node
    GROUP BY n.node
"""


class Scope(FrozenModel):
    """A directory of a project: the project root (the git checkout holding it) and the directory
    below the root, empty for the root itself."""

    root: Path
    prefix: str

    @classmethod
    def of(cls, path: Path) -> Scope:
        """The scope of `path`, its project being the git checkout that holds it (itself outside
        git)."""
        path = path.resolve()
        if not path.is_dir():
            raise MissionError(f"{path} is not a directory")
        found = Git(path).run("rev-parse", "--show-toplevel")
        root = Path(found.stdout.strip()).resolve() if found.succeeded else path
        return cls(root=root, prefix="" if path == root else path.relative_to(root).as_posix())

    @property
    def parameters(self) -> dict[str, str | int]:
        """What the unit SQL binds: the path prefix to match and where a unit's name starts."""
        if not self.prefix:
            return {"prefix": "", "skip": 1}
        return {"prefix": f"{self.prefix}/", "skip": len(self.prefix) + 2}


class Unit(FrozenModel):
    """One child of the scope.

    files: the Python files in it.
    internal: dependencies between its own files.
    depends: how many other units it reaches through imports, directly or not.
    cyclic: it takes part in an import cycle with other units.
    """

    name: str
    files: int
    internal: int
    depends: int
    cyclic: bool


class Dependency(FrozenModel):
    """`importer` imports `imported` in this many file pairs."""

    importer: str
    imported: str
    count: int


class Graph(FrozenModel):
    """The units of a scope, ordered with what they depend on first, their dependencies, and the
    loops among the files and among the units."""

    scope: str
    units: list[Unit]
    dependencies: list[Dependency]
    file_cycles: list[list[str]]
    unit_cycles: list[list[str]]


class UnitGraph:
    """The import graph of one scope, by unit and by file."""

    def __init__(self, workspace: Workspace, scope: Scope) -> None:
        self.workspace, self.scope = workspace, scope
        self._materialize()

    def fields(self) -> dict:
        """The `Graph` fields of the scope."""
        units = self.workspace.rows(
            """
            SELECT unit AS name, count(*) AS files,
                   coalesce((SELECT count FROM unit_edges e
                             WHERE e.src = unit AND e.dst = unit), 0) AS internal
            FROM scope_files GROUP BY unit
            """
        )
        units_by = self._components("unit_nodes", "unit_links")
        files_by = self._components("file_nodes", "file_links")
        links = self.workspace.rows("SELECT src, dst, count FROM unit_edges ORDER BY src, dst")
        return {
            "scope": self.scope.prefix or ".",
            "units": [
                Unit(
                    name=unit["name"],
                    files=unit["files"],
                    internal=unit["internal"],
                    depends=units_by[unit["name"]]["depends"],
                    cyclic=units_by[unit["name"]]["cyclic"],
                )
                for unit in sorted(
                    units, key=lambda unit: (units_by[unit["name"]]["depends"], unit["name"])
                )
            ],
            "dependencies": [
                Dependency(importer=link["src"], imported=link["dst"], count=link["count"])
                for link in links
            ],
            "file_cycles": self._cycles(files_by),
            "unit_cycles": self._cycles(units_by),
        }

    def _components(self, nodes: str, edges: str) -> dict[str, dict]:
        """Each node of a graph with its component, cyclic flag and reach count."""
        rows = self.workspace.rows(_COMPONENTS.format(nodes=nodes, edges=edges))
        return {row["node"]: row for row in rows}

    @staticmethod
    def _cycles(components: dict[str, dict]) -> list[list[str]]:
        """The groups of two or more nodes sharing a component."""
        grouped: dict[str, list[str]] = {}
        for node, row in components.items():
            if row["cyclic"]:
                grouped.setdefault(row["component"], []).append(node)
        return sorted(sorted(members) for members in grouped.values() if len(members) > 1)

    def _materialize(self) -> None:
        """Create the scope's files, file dependencies and unit dependencies as tables."""
        db = self.workspace.db
        self.workspace.install("SELECT 1 FROM design_edges()")
        db.execute(
            f"""
            CREATE OR REPLACE TABLE scope_files AS
            SELECT path, {_UNIT} AS unit FROM module_files() WHERE starts_with(path, $prefix)
            """,
            self.scope.parameters,
        )
        db.execute(
            """
            CREATE OR REPLACE TABLE file_links AS
            SELECT DISTINCT e.importer AS src, e.imported AS dst FROM design_edges() e
            JOIN scope_files a ON a.path = e.importer JOIN scope_files b ON b.path = e.imported
            """
        )
        db.execute("CREATE OR REPLACE TABLE file_nodes AS SELECT path AS node FROM scope_files")
        db.execute(
            """
            CREATE OR REPLACE TABLE unit_edges AS
            SELECT a.unit AS src, b.unit AS dst, count(*) AS count
            FROM file_links l
            JOIN scope_files a ON a.path = l.src JOIN scope_files b ON b.path = l.dst
            GROUP BY a.unit, b.unit
            """
        )
        db.execute(
            "CREATE OR REPLACE TABLE unit_links AS "
            "SELECT src, dst FROM unit_edges WHERE src <> dst"
        )
        db.execute(
            "CREATE OR REPLACE TABLE unit_nodes AS SELECT DISTINCT unit AS node FROM scope_files"
        )
