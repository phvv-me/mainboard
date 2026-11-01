import string
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import mainboard
from mainboard import cli
from mainboard.analysis.providers.cli import CommandTree

from .builders import Group, declare, expected

_NAMES = st.text(string.ascii_lowercase, min_size=1, max_size=5)
_COMMANDS = st.tuples(_NAMES, st.booleans(), st.booleans())
_GROUPS: st.SearchStrategy[Group] = st.recursive(
    st.tuples(_NAMES, st.booleans(), st.lists(_COMMANDS, max_size=3), st.just([])),
    lambda inner: st.tuples(
        _NAMES,
        st.booleans(),
        st.lists(_COMMANDS, max_size=3),
        st.lists(inner, max_size=3),
    ),
    max_leaves=6,
)


def _commands(declaration: str) -> list[str]:
    """The commands a tree whose root states `declaration` as its keywords holds."""
    source = f"TREE = Node(name='mb', help='h', {declaration})\n"
    return [e.command for e in CommandTree(path="constants.py", source=source).entries()]


def test_the_tree_read_from_source_is_the_running_tree() -> None:
    declaring = Path(mainboard.__file__).parent / "verbs" / "constants.py"
    read = CommandTree(path="constants.py", source=declaring.read_text(encoding="utf-8")).entries()
    running = Counter((" ".join((cli.TREE.name, *path)), e.show) for path, e in cli.TREE.entries())
    assert Counter((e.command, e.shown) for e in read if e.parent) == running


@given(root=_GROUPS)
@example(
    root=(
        "mb",
        True,
        [("run", False, True), ("mb", True, True)],
        [("host", False, [("list", False, False)], [])],
    )
)
def test_a_declared_tree_is_read_back_entry_for_entry(root: Group) -> None:
    read = CommandTree(path="constants.py", source=f"TREE = {declare(root)}\n").entries()
    assert Counter((e.command, e.parent, e.depth, e.shown) for e in read) == Counter(
        expected(root)
    )


@pytest.mark.parametrize(
    ("declaration", "commands"),
    [
        pytest.param(
            "commands=tuple(Command(name=word) for word in ('a', 'b'))",
            ["mb", "mb a", "mb b"],
            id="a-comprehension-over-a-tuple",
        ),
        pytest.param(
            "commands=(Command(name='a'), *(Command(name='b'),))",
            ["mb", "mb a", "mb b"],
            id="a-starred-tuple",
        ),
        pytest.param(
            "commands=(Command(name='a'),) + (Command(name='b'),)",
            ["mb", "mb a", "mb b"],
            id="a-sum-of-tuples",
        ),
        pytest.param(
            "commands=[Command(name='a')]",
            ["mb", "mb a"],
            id="a-list",
        ),
        pytest.param(
            "commands=(Command(name='a', default=True), Command(name='b'))",
            ["mb", "mb b"],
            id="a-default-command-is-no-entry",
        ),
    ],
)
def test_the_forms_a_declaration_may_take_are_read_as_data(
    declaration: str, commands: list[str]
) -> None:
    assert _commands(declaration) == commands


@pytest.mark.parametrize(
    "declaration",
    [
        pytest.param("commands=build()", id="a-call-of-a-function"),
        pytest.param("commands=known", id="a-name-nothing-binds"),
        pytest.param("show=1.5", id="a-float"),
        pytest.param("commands=tuple(Command(name=w) for w in ('a',) if w)", id="a-filter"),
        pytest.param("commands=[Command(name=w) for w in ('a',)]", id="a-list-comprehension"),
        pytest.param("commands=('a',) - ('b',)", id="a-difference"),
        pytest.param("commands='a' + 'b'", id="a-sum-of-strings"),
    ],
)
def test_a_declaration_that_is_not_plain_data_is_refused(declaration: str) -> None:
    with pytest.raises(ValueError, match="constants.py:1: not data"):
        _commands(declaration)


def test_only_the_modules_declaring_a_tree_are_read(tmp_path: Path) -> None:
    (tmp_path / "verbs").mkdir()
    (tmp_path / "verbs" / "constants.py").write_text("TREE = Node(name='mb', help='h')\n")
    (tmp_path / "verbs" / "other.py").write_text("TREE = Tree(name='mb')\n")
    (tmp_path / "notes.md").write_text("TREE = Node(name='mb')\n")
    assert [tree.path for tree in CommandTree.declared_in(tmp_path)] == ["verbs/constants.py"]
