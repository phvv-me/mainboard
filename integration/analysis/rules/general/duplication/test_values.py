"""ALL-DUPL0005 names the line and column where the module first writes the repeated literal, as
ALL-DUPL0002 does, and does not count the places that hand the text to a callable."""

from collections.abc import Mapping

import pytest

from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.general.duplication.values import ModuleRepeatedStringLiteral

from ....engine.builders import written

_RETURNS = """\
import sys


def first():
    return "observed"


def second():
    return "observed"


def third():
    return "observed"


def fourth():
    return "observed"
"""
_COMPARED_FIRST = """\
def check(mode, other):
    return mode == "observed" or other == "observed"


NAMES = ["observed", "observed"]
"""
_LISTED_FIRST = """\
import sys

NAMES = ["observed", "observed"]


def check(mode, other):
    return mode == "observed" or other == "observed"
"""
_AFTER_A_CALLEE = """\
import sys

sys.stdout.write("observed")
NAMES = ["observed", "observed", "observed", "observed"]
"""
_FILES = {
    "returns.py": _RETURNS,
    "compared_first.py": _COMPARED_FIRST,
    "listed_first.py": _LISTED_FIRST,
    "after_a_callee.py": _AFTER_A_CALLEE,
}


@pytest.fixture(scope="module")
def places(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[int, int]]:
    """The `(line, column)` of the rule's finding in each of `_FILES`, by file."""
    root = written(tmp_path_factory.mktemp("duplication"), _FILES)
    load_catalog()
    with Lint(root, (ModuleRepeatedStringLiteral,), fixing=False) as lint:
        rows = lint.findings(lint.facts(root))[ModuleRepeatedStringLiteral.id]
    return {path: (line, column) for path, line, column, _ in rows}


@pytest.mark.parametrize(
    ("name", "place"),
    [
        pytest.param("returns.py", (5, 11), id="the-first-of-four-returns"),
        pytest.param("compared_first.py", (2, 19), id="a-comparison-before-a-list"),
        pytest.param("listed_first.py", (3, 9), id="a-list-before-a-comparison"),
        pytest.param("after_a_callee.py", (4, 9), id="a-call-argument-is-not-an-occurrence"),
    ],
)
def test_the_finding_sits_where_the_module_first_writes_the_repeated_literal(
    places: Mapping[str, tuple[int, int]], name: str, place: tuple[int, int]
) -> None:
    assert places[name] == place
