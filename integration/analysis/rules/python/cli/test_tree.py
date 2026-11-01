from pathlib import Path

import pytest

from mainboard.analysis.engine.lint import Lint, load_catalog, verbs
from mainboard.analysis.rules.python.cli.tree import CommandTreeCeiling

from .builders import tree

_PYPROJECT = '[project]\nname = "pkg"\nversion = "0"\n'
_WIDE = "`mb` shows 11 children, past the ceiling of 10"
_DEEP = "`mb g4 g3 g2 leaf` sits 4 words deep, past the ceiling of 3"


def _found(root: Path, declaration: str, *, policy: str = "") -> list[str]:
    files = {
        "pyproject.toml": _PYPROJECT + policy,
        "src/pkg/__init__.py": "",
        "src/pkg/verbs/constants.py": declaration,
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    load_catalog()
    with Lint(root, (CommandTreeCeiling,), fixing=False) as lint:
        return [row[3] for row in lint.findings(lint.facts(root)).get(CommandTreeCeiling.id, [])]


@pytest.mark.parametrize(
    ("declaration", "said"),
    [
        pytest.param(tree(shown=10, depth=3), [], id="ten-children-three-words-deep"),
        pytest.param(tree(shown=11), [_WIDE], id="eleven-children"),
        pytest.param(tree(shown=10, hidden=5), [], id="hidden-commands-are-no-children"),
        pytest.param(tree(shown=10, default=True), [], id="a-default-command-is-no-child"),
        pytest.param(tree(shown=11, depth=1), [_WIDE], id="eleven-commands-and-no-group"),
        pytest.param(tree(depth=4), [_DEEP], id="four-words-deep"),
        pytest.param(tree(shown=11, depth=4), [_WIDE, _DEEP], id="too-wide-and-too-deep"),
        pytest.param("def main() -> None: ...\n", [], id="no-tree-declared"),
        pytest.param("TREE = Node(name='mb')\ndef main(:\n", [], id="a-tree-that-does-not-parse"),
        pytest.param("TREE = Node(name=NAME)\n", [], id="a-tree-that-is-no-plain-data"),
    ],
)
def test_a_command_tree_past_the_default_ceilings_is_flagged(
    tmp_path: Path, declaration: str, said: list[str]
) -> None:
    assert _found(tmp_path, declaration) == said


@pytest.mark.parametrize(
    ("policy", "said"),
    [
        pytest.param("maximum = 11\n", [_DEEP], id="a-raised-width-leaves-the-depth"),
        pytest.param("maximum_depth = 4\n", [_WIDE], id="a-raised-depth-leaves-the-width"),
        pytest.param("maximum = 11\nmaximum_depth = 4\n", [], id="both-raised"),
    ],
)
def test_a_project_raises_each_ceiling_in_its_own_rule_table(
    tmp_path: Path, policy: str, said: list[str]
) -> None:
    table = '\n[tool.mcmr.rules."PY-CLI0002"]\n' + policy
    assert _found(tmp_path, tree(shown=11, depth=4), policy=table) == said


def test_every_example_of_the_command_tree_rule_passes() -> None:
    """Each `bad` is flagged and each `good` is not."""
    verbs.test(selection=verbs.Selection(rule=(CommandTreeCeiling.id,)))
