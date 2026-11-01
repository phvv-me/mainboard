"""`move-definitions` moves top-level definitions into another module, the comments that belong to
them going along. Moves into one module that does not exist yet take it in turns, so it gets one
set of imports. A class takes the comments that end its body, and the header comment of the module
it leaves goes with the definitions it names."""

import ast
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import example, given, settings

from mainboard.analysis.commands.tools import move
from mainboard.analysis.operations import MoveDefinitions

from ..engine.builders import written
from .builders import enum_modules, row
from .constants import DESTINATION, PACKAGE
from .readers import moved

_IMPORTS = "from enum import StrEnum, auto\n"
_CARRIED = "from enum import StrEnum\nfrom enum import auto\n"
_MODE = "class Mode(StrEnum):\n    FAST = auto()\n"
_WORD = "class Word:\n    pass\n"
_AFTER = "def describe() -> int:\n    return LIMIT\n"
_TOP = f"{_IMPORTS}\nLIMIT = 3\n\n\n"
_SOURCE = "src/pkg/a.py"
_PROJECT = {"pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n', **PACKAGE}


@settings(max_examples=25)
@example(
    drawn={
        "src/pkg/m0.py": f"{_IMPORTS}\n\n{_MODE.replace('Mode', 'C0')}",
        "src/pkg/m1.py": f"{_IMPORTS}\n\n{_MODE.replace('Mode', 'C1')}",
    }
)
@given(drawn=enum_modules())
def test_moves_into_one_new_module_leave_it_each_definition_once_under_one_set_of_imports(
    drawn: dict[str, str],
) -> None:
    moves = [(path, [f"C{index}"], DESTINATION) for index, path in enumerate(drawn)]

    after = moved({**PACKAGE, **drawn}, *moves)

    tree = ast.parse(after[DESTINATION])
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    bound = Counter(alias.asname or alias.name for _, alias in _imports(tree))
    assert sorted(classes) == [f"C{index}" for index in range(len(drawn))]
    assert set(bound.values()) == {1}


def _imports(tree: ast.Module) -> list[tuple[ast.stmt, ast.alias]]:
    """Every name the top-level imports of `tree` bind, with the statement binding it."""
    return [
        (node, alias)
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    ]


@pytest.mark.parametrize(
    ("before", "old", "new"),
    [
        pytest.param(
            f"{_TOP}{_MODE}    # Breadth kinds belong to a later phase.\n\n\n{_AFTER}",
            f"{_TOP}{_AFTER}",
            f"{_CARRIED}\n\n{_MODE}    # Breadth kinds belong to a later phase.\n",
            id="a-comment-ending-the-body",
        ),
        pytest.param(
            f"{_TOP}{_MODE}\n    # Set apart by a gap.\n    # In two lines.\n\n\n{_AFTER}",
            f"{_TOP}{_AFTER}",
            f"{_CARRIED}\n\n{_MODE}\n    # Set apart by a gap.\n    # In two lines.\n",
            id="comments-set-apart-from-the-body-by-a-gap",
        ),
        pytest.param(
            f"{_TOP}{_MODE}\n\n# About describe.\n{_AFTER}",
            f"{_TOP}# About describe.\n{_AFTER}",
            f"{_CARRIED}\n\n{_MODE}",
            id="a-comment-at-the-left-edge-stays-for-the-next-definition",
        ),
        pytest.param(
            f"{_TOP}def Mode() -> int:\n    return 1\n    # Unreachable.\n\n\n{_AFTER}",
            f"{_TOP}{_AFTER}",
            "def Mode() -> int:\n    return 1\n    # Unreachable.\n",
            id="a-comment-ending-a-function",
        ),
    ],
)
def test_a_definition_takes_the_comments_ending_its_body_along(
    before: str, old: str, new: str
) -> None:
    after = moved({**PACKAGE, _SOURCE: before}, (_SOURCE, ["Mode"], DESTINATION))

    assert (after[_SOURCE], after[DESTINATION]) == (old, new)


@pytest.mark.parametrize(
    ("header", "old", "new"),
    [
        pytest.param(
            "# The words it has.\n#\n# `Mode` is fixed: it worked.\n#\n# A dead one is a gain.\n",
            "# The words it has.\n#\n# A dead one is a gain.\n",
            "# `Mode` is fixed: it worked.\n",
            id="the-paragraph-naming-it-goes-above-it-and-the-others-stay",
        ),
        pytest.param(
            "# `Mode` is fixed.\n#\n# It picks the speed, `Mode.FAST` or `Mode.SLOW`.\n",
            "",
            "# `Mode` is fixed.\n#\n# It picks the speed, `Mode.FAST` or `Mode.SLOW`.\n",
            id="a-header-naming-only-it-leaves-the-module-with-none",
        ),
        pytest.param(
            "# What a receipt carries.\n",
            "# What a receipt carries.\n",
            "",
            id="a-header-naming-nothing-stays",
        ),
        pytest.param(
            "# `Word` is the consumer's.\n",
            "# `Word` is the consumer's.\n",
            "",
            id="a-header-naming-only-what-stays-stays",
        ),
        pytest.param(
            "# `Mode` is fixed.\n#\n# A third paragraph.\n",
            "# A third paragraph.\n",
            "# `Mode` is fixed.\n",
            id="the-first-paragraph-leaves-with-the-bare-line-after-it",
        ),
        pytest.param(
            "# A first paragraph.\n#\n# `Mode` is fixed.\n",
            "# A first paragraph.\n",
            "# `Mode` is fixed.\n",
            id="the-last-paragraph-leaves-with-the-bare-line-before-it",
        ),
    ],
)
def test_the_header_paragraphs_naming_a_moved_definition_go_with_it(
    header: str, old: str, new: str
) -> None:
    body = f"{_IMPORTS}\n\n{_MODE}\n\n{_WORD}"
    stayed = f"{old}\n{_IMPORTS}\n\n{_WORD}" if old else f"{_IMPORTS}\n\n{_WORD}"
    sources = {**PACKAGE, _SOURCE: f"{header}\n{body}"}

    after = moved(sources, (_SOURCE, ["Mode"], DESTINATION))

    assert after[_SOURCE] == stayed
    assert after[DESTINATION] == f"{_CARRIED}\n\n{new}{_MODE}"


def test_a_comment_right_above_a_statement_is_that_statements_own_and_stays() -> None:
    comment = "# `Mode` is fixed.\n"
    sources = {**PACKAGE, _SOURCE: f"{comment}{_IMPORTS}\n\n{_MODE}\n\n{_WORD}"}

    after = moved(sources, (_SOURCE, ["Mode"], DESTINATION))

    assert after[_SOURCE] == f"{comment}{_IMPORTS}\n\n{_WORD}"


def test_a_moved_definition_keeps_its_own_comment_apart_from_the_header_paragraph() -> None:
    sources = {
        **PACKAGE,
        _SOURCE: f"# `Mode` is fixed.\n\n{_IMPORTS}\n\n# The speeds.\n{_MODE}",
    }

    after = moved(sources, (_SOURCE, ["Mode"], DESTINATION))

    assert after[DESTINATION] == f"{_CARRIED}\n\n# `Mode` is fixed.\n#\n# The speeds.\n{_MODE}"


def test_a_header_paragraph_tying_a_moved_definition_to_one_that_stays_refuses_the_move() -> None:
    header = "# `Mode` is fixed and `Word` is the consumer's.\n"
    sources = {**PACKAGE, _SOURCE: f"{header}\n{_IMPORTS}\n\n{_MODE}\n\n{_WORD}"}
    operation = MoveDefinitions()

    edits = operation.edits(row(sources, (_SOURCE, ["Mode"], DESTINATION)))

    assert edits is None
    assert "header comment names a definition that moves and one that stays" in operation.why


def test_a_destination_outside_the_source_root_has_no_module_name_and_refuses_the_move() -> None:
    sources = {**PACKAGE, _SOURCE: f"{_IMPORTS}\n\n{_MODE}"}
    operation = MoveDefinitions()

    edits = operation.edits(
        {**row(sources, (_SOURCE, ["Mode"], "enums.py")), "destination_module": ""}
    )

    assert edits is None
    assert "stands outside the source root" in operation.why


def test_an_import_from_another_top_level_folder_follows_the_move(tmp_path: Path) -> None:
    written(
        tmp_path,
        {
            **_PROJECT,
            _SOURCE: "def helper() -> int:\n    return 1\n\n\nother = 2\n",
            "integration/checks/test_a.py": "from pkg.a import helper\n\nVALUE = helper()\n",
            "scripts/run.py": "from pkg import a\n\nVALUE = a.helper()\n",
        },
    )

    code = move((f"{_SOURCE}::helper",), to=Path("src/pkg/c.py"), root=tmp_path, write=True)

    read = {
        name: (tmp_path / name).read_text()
        for name in ("integration/checks/test_a.py", "scripts/run.py")
    }
    assert code == 0
    assert read == {
        "integration/checks/test_a.py": "from pkg.c import helper\n\nVALUE = helper()\n",
        "scripts/run.py": "from pkg import c\n\nVALUE = c.helper()\n",
    }
