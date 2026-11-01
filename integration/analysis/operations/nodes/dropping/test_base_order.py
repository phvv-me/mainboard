"""Dropping the redundant base of a class never changes the order in which any class of the module
resolves its methods, which is what makes the fix of PY-CLAS0007 safe to review quickly."""

import ast
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.constants import FIXABLE
from mainboard.analysis.engine.fixing import Fixpoint
from mainboard.analysis.engine.lint import Lint, load_catalog, selected
from mainboard.analysis.rules.python.classes.inheritance import (
    HazardousMultipleInheritanceMroCount,
)

from ....engine.builders import written

type Orders = dict[str, list[str]]


@pytest.fixture(scope="module")
def fixed() -> Callable[[str], str]:
    """The text of a module after the fix of PY-CLAS0007 has run on it, pass after pass."""

    def run(source: str) -> str:
        load_catalog()
        with tempfile.TemporaryDirectory(prefix="mb-drop-") as scratch:
            root = written(Path(scratch), {"module.py": source})
            with Lint(root, selected((HazardousMultipleInheritanceMroCount.id,))) as lint:
                list(Fixpoint(lint, root, FIXABLE["review"]).run(write=True))
            return (root / "module.py").read_text(encoding="utf-8")

    return run


@st.composite
def _hierarchies(draw: st.DrawFn) -> tuple[list[list[int]], list[int]]:
    """A module's classes, each with bases drawn from the classes before it, and a last one.

    That is the bases of three to six classes, and the bases (two to four, in any order) of a last
    class that may take any of them.
    """
    count = draw(st.integers(3, 6))
    parents = [
        draw(st.lists(st.integers(0, place - 1), unique=True, max_size=3)) if place else []
        for place in range(count)
    ]
    return parents, draw(st.lists(st.integers(0, count - 1), unique=True, min_size=2, max_size=4))


def _class(name: str, bases: Sequence[int]) -> str:
    """The source of a class named `name` whose bases are the classes `K<n>` for `n` in `bases`."""
    inherited = f"({', '.join(f'K{base}' for base in bases)})" if bases else ""
    return f"class {name}{inherited}:\n    pass\n"


def _orders(source: str) -> Orders | None:
    """The method resolution order of each class of `source`, None when Python refuses one."""
    made: dict[str, type] = {}
    for node in [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]:
        bases = tuple(made[ast.unparse(base)] for base in node.bases)
        try:
            made[node.name] = type(node.name, bases, {})
        except TypeError:
            return None
    return {name: [each.__name__ for each in kind.__mro__] for name, kind in made.items()}


@settings(max_examples=15)
@example(chosen=([[], [0]], [1, 0]))
@example(chosen=([[], [0], []], [1, 2, 0]))
@example(chosen=([[], [], [0], [1]], [3, 2, 0, 1]))
@example(chosen=([[], [0], [1]], [2, 1, 0]))
@given(chosen=_hierarchies())
def test_dropping_a_redundant_base_keeps_the_method_resolution_order_of_every_class(
    fixed: Callable[[str], str], chosen: tuple[list[list[int]], list[int]]
) -> None:
    parents, final = chosen
    source = "\n\n".join(
        [*(_class(f"K{place}", bases) for place, bases in enumerate(parents)), _class("C", final)]
    )
    before = _orders(source)
    assume(before is not None)

    assert _orders(fixed(source)) == before
