from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

import pytest
from duckdb import DuckDBPyConnection

from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.operations.nodes import Dialect, Node, RewriteNode, Site
from mainboard.analysis.operations.nodes.dialects import Python

from ....engine.builders import written
from .builders import sources
from .readers import Rewritten

_NODE = (
    "SELECT min(node_id) FROM ast_nodes WHERE path = ? AND type = ? AND (? = 0 OR start_line = ?)"
)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project holding the files the rewrite tests read."""
    return written(tmp_path_factory.mktemp("rewriting"), sources())


@pytest.fixture(scope="module")
def parse(grammar: Callable[[type[Dialect], str], list[Node]]) -> Callable[[str], list[Node]]:
    """Parse a text with the Python grammar, pruning nothing: a pattern's nodes, or a target's."""
    return partial(grammar, Python)


@pytest.fixture(scope="module")
def site(cursor: DuckDBPyConnection) -> Callable[..., Site]:
    """The site of the outermost node of a type in a file (on a line, when one is given)."""

    def gathered(path: str, *, kind: str, line: int = 0) -> Site:
        row = cursor.execute(_NODE, [path, kind, line, line]).fetchone()
        assert row is not None
        return Site.gathered(cursor, [(path, row[0])])[path, row[0]]

    return gathered


@pytest.fixture(scope="module")
def plan(project: Path) -> Callable[[Mapping], Rewritten]:
    """Run a `rewrite-node` row as the engine does and write its edits in one plan."""

    def planned(row: Mapping) -> Rewritten:
        found: Site = row["site"]
        operation = RewriteNode()
        edits = operation.edits(row)
        writer = Plan(project)
        if edits is not None:
            writer.add(edits)
        data = writer.written().get(found.path, (project / found.path).read_bytes())
        return Rewritten(
            kinds=[edit.kind for edit in edits or ()], why=operation.why, text=data.decode()
        )

    return planned
