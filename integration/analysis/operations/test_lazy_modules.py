"""`lazy-modules` lists a finding's modules in the file's `__lazy_modules__` before every import,
keeping the modules it listed that the file still imports."""

import ast
from collections.abc import Sequence
from pathlib import Path

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.operations import Operation

_PATH = "module.py"
_NAMES = ["alpha", "beta", "gamma", "delta", "gone"]


@st.composite
def _modules(draw: st.DrawFn) -> tuple[list[str], int | None, list[str], list[str]]:
    """A file's imports, the place of its list, what the list holds and the finding's entries.

    The list stands before import number `placed` (None for no list) and holds nothing without
    one. Every entry is imported.
    """
    imported = draw(st.lists(st.sampled_from(_NAMES[:4]), min_size=1, unique=True))
    placed = draw(st.none() | st.integers(0, len(imported)))
    listed = draw(st.lists(st.sampled_from(_NAMES), unique=True)) if placed is not None else []
    entries = draw(st.lists(st.sampled_from(imported), min_size=1, unique=True))
    return imported, placed, listed, entries


def _source(imported: Sequence[str], placed: int | None, listed: Sequence[str]) -> str:
    """A module importing `imported`, its list of `listed` standing before import `placed`."""
    lines = [f"import {name}" for name in imported]
    if placed is not None:
        lines.insert(placed, f"__lazy_modules__ = {listed!r}")
    return "\n".join([*lines, "", ""])


def _lists(tree: ast.Module) -> list[tuple[int, set[str]]]:
    """Each module-level assignment of `__lazy_modules__` as its line and the modules it lists."""
    return [
        (node.lineno, set(ast.literal_eval(node.value)))
        for node in tree.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "__lazy_modules__"
    ]


@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@example(drawn=(["alpha", "beta"], 1, ["beta", "gone"], ["alpha"]))
@example(drawn=(["alpha"], None, [], ["alpha"]))
@example(drawn=(["alpha", "beta"], 2, ["alpha", "beta"], ["beta"]))
@given(drawn=_modules())
def test_the_lazy_list_keeps_what_the_file_still_imports_and_stands_before_every_import(
    tmp_path: Path, drawn: tuple[list[str], int | None, list[str], list[str]]
) -> None:
    imported, placed, listed, entries = drawn
    source = _source(imported, placed, listed)
    row = {"operation": "lazy-modules", "path": _PATH, "content": source, "importer": "module"}
    (tmp_path / _PATH).write_text(source, encoding="utf-8")
    plan = Plan(tmp_path)

    plan.add(Operation.rendered({**row, "entries": entries}) or [])

    tree = ast.parse(plan.written()[_PATH])
    kept = (set(listed) & set(imported)) | set(entries)
    assert _lists(tree) == [(1, kept)]
    assert [type(node) for node in tree.body[1:]] == [ast.Import] * len(imported)
