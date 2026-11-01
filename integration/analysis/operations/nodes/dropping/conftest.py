import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import pytest
from duckdb import DuckDBPyConnection

from mainboard.analysis import Example, Rule
from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.engine.lint import Lint, load_catalog, selected, verbs
from mainboard.analysis.operations.nodes import DropArgument, Site

from ....engine.builders import written
from .readers import Dropped

_NODES = "SELECT node_id, start_byte, end_byte FROM ast_nodes WHERE path = ? AND start_line = ?"


def _node(cursor: DuckDBPyConnection, data: bytes, path: str, target: tuple[int, str]) -> int:
    """The outermost node written as the target's text and starting on its line of `path`.

    data: the bytes of the file, which the facts measure the nodes in.
    """
    line, text = target
    rows = cursor.execute(_NODES, [path, line]).fetchall()
    return min(node for node, start, end in rows if data[start:end].decode() == text)


def _planned(root: Path, path: str, sites: Sequence[Site]) -> Dropped:
    """The drops of `sites` added to one plan, in order, and the file the plan writes."""
    plan, made, refusals = Plan(root), [], []
    for site in sites:
        operation = DropArgument()
        made.append(operation.edits({"site": site}))
        refusals.append(operation.why)
    taken = [edits is not None and plan.add(edits) for edits in made]
    data = plan.written().get(path, (root / path).read_bytes())
    kinds = [[edit.kind for edit in edits or ()] for edits in made]
    return Dropped(edits=kinds, refusals=refusals, taken=taken, text=data.decode())


def _dropped(files: Mapping[str, str], path: str, targets: Sequence[tuple[int, str]]) -> Dropped:
    """Drop the nodes `targets` name, each a line and a text, of the file `path` of a project."""
    load_catalog()
    with tempfile.TemporaryDirectory(prefix="mb-drop-") as scratch:
        root = written(Path(scratch), files)
        with Lint(root, selected(("ALL-CONT0002",))) as lint:
            cursor = lint.facts(root).cursor(None)
            data = files[path].encode()
            nodes = [_node(cursor, data, path, target) for target in targets]
            sites = Site.gathered(cursor, [(path, node) for node in nodes])
        return _planned(root, path, [sites[path, node] for node in nodes])


@pytest.fixture(scope="module")
def drop() -> Callable[[Mapping[str, str], str, Sequence[tuple[int, str]]], Dropped]:
    """Drop nodes of a scratch project's file as the engine does: from the facts of its tree."""
    return _dropped


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
