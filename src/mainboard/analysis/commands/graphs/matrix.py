"""`matrix`: the import design structure matrix of a package, and its import cycles.

Rows are the units of the directory (its subpackages and modules), ordered with the units that
depend on the fewest others first, so a layered design fills only below the diagonal; a mark above
the diagonal is a dependency running against that order. A cell counts the file pairs in which the
row's unit imports the column's; the diagonal counts a unit's own internal pairs. Cycles are the
groups of files, and of units, that import each other around a loop.
"""

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from ....log import logger
from ..report import Report
from ..workspace import Workspace
from .scope import Graph, Scope, UnitGraph

_GRID = 32


class Matrix(Graph, Report):
    """The dependency matrix of one directory."""

    @staticmethod
    def _loops(title: str, cycles: list[list[str]]) -> list[str]:
        if not cycles:
            return [f"no {title}"]
        return [f"{len(cycles)} {title}:", *(f"  {' <-> '.join(members)}" for members in cycles)]

    def _grid(self) -> list[str]:
        names = [unit.name for unit in self.units]
        cells = {(d.importer, d.imported): d.count for d in self.dependencies}
        width = max(2, len(str(max(cells.values(), default=0))))
        label = max(map(len, names))
        header = " " * (label + 5) + " ".join(f"{n + 1:>{width}}" for n in range(len(names)))
        rows = [
            f"{number + 1:>3} {name:<{label}} "
            + " ".join(f"{cells.get((name, other), 0) or '.':>{width}}" for other in names)
            for number, name in enumerate(names)
        ]
        return [header, *rows]

    def _list(self) -> list[str]:
        """Too many units for a grid: each with the units it imports, most files first."""
        found: dict[str, list[str]] = {}
        for dependency in sorted(self.dependencies, key=lambda d: (-d.count, d.imported)):
            if dependency.importer != dependency.imported:
                found.setdefault(dependency.importer, []).append(
                    f"{dependency.imported} {dependency.count}"
                )
        return [
            f"{unit.name} ({unit.files} files, depends on {unit.depends}): "
            f"{', '.join(found.get(unit.name, ['-']))}"
            for unit in self.units
        ]

    def text(self) -> str:
        lines = [
            f"{len(self.units)} units under {self.scope}, "
            f"{sum(d.count for d in self.dependencies if d.importer != d.imported)} "
            f"cross-unit file pairs (row imports column)"
        ]
        lines += self._grid() if len(self.units) <= _GRID else self._list()
        lines += self._loops("unit cycles", self.unit_cycles)
        lines += self._loops("file cycles", self.file_cycles)
        return "\n".join(lines)


def matrix(
    path: Path = Path(),
    *,
    as_json: Annotated[bool, Parameter(name="--json")] = False,
    memory: str = "2GB",
) -> None:
    """The import design structure matrix of a package, by subpackage or module, with its cycles.

    path: a directory of the project; its subpackages and modules are the rows.
    as_json: print JSON for a program.
    memory: a DuckDB memory limit such as `4GB`.
    """
    scope = Scope.of(path)
    graph = UnitGraph(Workspace(scope.root, memory=memory), scope)
    logger.result(Matrix(**graph.fields()).render(as_json=as_json))
