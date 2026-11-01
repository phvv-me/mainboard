import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import pytest
from duckdb import DuckDBPyConnection

from mainboard.analysis import Example, Rule
from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.engine.lint import Lint, load_catalog, selected, verbs
from mainboard.analysis.operations.nodes import DeleteNode, Site

from ....engine.builders import written
from .readers import Deletion

_NODE = "SELECT min(node_id) FROM ast_nodes WHERE path = ? AND start_line = ? AND type = ?"


def _node(cursor: DuckDBPyConnection, path: str, target: tuple[int, str]) -> int:
    """The outermost node of the target's type starting on the target's line of `path`."""
    row = cursor.execute(_NODE, [path, *target]).fetchone()
    assert row is not None
    return row[0]


def _planned(root: Path, path: str, sites: Sequence[Site]) -> Deletion:
    """The deletions of `sites` added to one plan, in order, and the file the plan writes."""
    plan, made, refusals = Plan(root), [], []
    for site in sites:
        operation = DeleteNode()
        made.append(operation.edits({"site": site}))
        refusals.append(operation.why)
    taken = [edits is not None and plan.add(edits) for edits in made]
    data = plan.written().get(path, (root / path).read_bytes())
    kinds = [[edit.kind for edit in edits or ()] for edits in made]
    return Deletion(edits=kinds, refusals=refusals, taken=taken, text=data.decode())


def _deleted(files: Mapping[str, str], path: str, targets: Sequence[tuple[int, str]]) -> Deletion:
    """Delete the nodes `targets` name, each a line and a type, of the file `path` of a project."""
    load_catalog()
    with tempfile.TemporaryDirectory(prefix="mb-delete-") as scratch:
        root = written(Path(scratch), files)
        with Lint(root, selected(("ALL-CONT0002",))) as lint:
            cursor = lint.facts(root).cursor(None)
            nodes = [_node(cursor, path, target) for target in targets]
            sites = Site.gathered(cursor, [(path, node) for node in nodes])
        return _planned(root, path, [sites[path, node] for node in nodes])


@pytest.fixture(scope="module")
def delete() -> Callable[[Mapping[str, str], str, Sequence[tuple[int, str]]], Deletion]:
    """Delete nodes of a scratch project's file as the engine does: from the facts of its tree."""
    return _deleted


@pytest.fixture
def verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[[type[Rule], Example], str]:
    """What `lint test` says of one example of a rule, in place of the rule's own."""

    def run(rule: type[Rule], example: Example) -> str:
        monkeypatch.setattr(rule, "examples", (example,))
        with suppress(SystemExit):
            verbs.test(selection=verbs.Selection(rule=(rule.id,)))
        return capsys.readouterr().out.splitlines()[0].partition(": ")[2]

    return run
