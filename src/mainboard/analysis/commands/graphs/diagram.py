"""`diagram`: a Mermaid diagram of a package's dependencies or of its classes.

`packages` draws the units of the directory (the rows of `matrix`) with an arrow from each unit to
the units it imports, labeled by the file pairs, units in an import cycle shaded. `classes` draws
the classes of the directory that take part in inheritance, each with its bases; a base from
outside the project is drawn as a plain box.
"""

import re
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from patos import FrozenModel

from ....log import logger
from ..report import Report
from ..workspace import Workspace
from .scope import Scope, UnitGraph

_CLASSES = """
    SELECT module, qualname, base, target_module, target FROM class_bases()
    WHERE starts_with(path, $prefix) AND base <> 'object'
    ORDER BY module, qualname, base
"""


class DiagramKind(StrEnum):
    """What a diagram draws."""

    PACKAGES = auto()
    CLASSES = auto()


class Node(FrozenModel):
    """A box: a unit of the scope or a class."""

    id: str
    label: str
    shaded: bool = False


class Edge(FrozenModel):
    """An arrow from `source` to `target`, labeled when it counts something."""

    source: str
    target: str
    label: str = ""


class Diagram(Report):
    """A Mermaid diagram, its nodes and edges kept for programs."""

    kind: DiagramKind
    nodes: list[Node]
    edges: list[Edge]

    def _classes(self) -> str:
        lines = ["classDiagram"]
        lines += [f'  class {node.id}["{node.label}"]' for node in self.nodes]
        lines += [f"  {edge.target} <|-- {edge.source}" for edge in self.edges]
        return "\n".join(lines)

    def _packages(self) -> str:
        lines = ["flowchart LR", "  classDef cyclic fill:#fde8e8,stroke:#c0392b"]
        lines += [f'  {node.id}["{node.label}"]' for node in self.nodes]
        lines += [f"  {edge.source} -->|{edge.label}| {edge.target}" for edge in self.edges]
        shaded = [node.id for node in self.nodes if node.shaded]
        return "\n".join([*lines, *([f"  class {','.join(shaded)} cyclic"] if shaded else [])])

    def text(self) -> str:
        return self._packages() if self.kind is DiagramKind.PACKAGES else self._classes()


def _identifier(text: str) -> str:
    """A Mermaid-safe id: word characters only, never a reserved word such as `end`."""
    return "n_" + re.sub(r"\W", "_", text)


class Drawing:
    """The diagrams of one directory."""

    def __init__(self, workspace: Workspace, scope: Scope) -> None:
        self.workspace, self.scope = workspace, scope

    def classes(self) -> Diagram:
        """The classes with bases under the scope, and the bases they name."""
        rows = self.workspace.rows(_CLASSES, {"prefix": self.scope.parameters["prefix"]})
        nodes: dict[str, Node] = {}
        edges: list[Edge] = []
        for row in rows:
            child = f"{row['module']}.{row['qualname']}"
            base = f"{row['target_module']}.{row['target']}" if row["target"] else row["base"]
            nodes.setdefault(child, Node(id=_identifier(child), label=row["qualname"]))
            nodes.setdefault(base, Node(id=_identifier(base), label=row["target"] or row["base"]))
            edges.append(Edge(source=_identifier(child), target=_identifier(base)))
        return Diagram(kind=DiagramKind.CLASSES, nodes=list(nodes.values()), edges=edges)

    def packages(self) -> Diagram:
        """The scope's units and their dependencies."""
        graph = UnitGraph(self.workspace, self.scope).fields()
        return Diagram(
            kind=DiagramKind.PACKAGES,
            nodes=[
                Node(
                    id=_identifier(unit.name),
                    label=f"{unit.name} ({unit.files})",
                    shaded=unit.cyclic,
                )
                for unit in graph["units"]
            ],
            edges=[
                Edge(
                    source=_identifier(dep.importer),
                    target=_identifier(dep.imported),
                    label=str(dep.count),
                )
                for dep in graph["dependencies"]
                if dep.importer != dep.imported
            ],
        )


def diagram(
    path: Path = Path(),
    *,
    kind: DiagramKind = DiagramKind.PACKAGES,
    as_json: Annotated[bool, Parameter(name="--json")] = False,
    memory: str = "2GB",
) -> None:
    """A Mermaid diagram of a package's dependencies or of its classes and their bases.

    path: a directory of the project.
    kind: `packages` for the units and what they import, `classes` for the inheritance.
    as_json: print the nodes and edges as JSON instead of Mermaid.
    memory: a DuckDB memory limit such as `4GB`.
    """
    scope = Scope.of(path)
    drawing = Drawing(Workspace(scope.root, memory=memory), scope)
    shown = drawing.packages() if kind is DiagramKind.PACKAGES else drawing.classes()
    logger.result(shown.render(as_json=as_json))
