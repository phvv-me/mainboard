"""A fixpoint keeps a fix that changes test files only when pytest lists the ids the fix's change
line accounts for: the duplicates it deletes, the tests it merges into cases. A fix that drops
tests without naming them is undone before anything is written."""

from contextlib import suppress
from pathlib import Path

import pytest

from mainboard.analysis import Rule
from mainboard.analysis.engine.fixing import Fixpoint
from mainboard.analysis.engine.lint import Lint, load_catalog, verbs
from mainboard.analysis.rules.python.testing.cases import ParametrizationCandidateGroupCount
from mainboard.analysis.rules.python.testing.redundancy import (
    DuplicateTestIntentClusterCount,
    ProductionReachHotspotCount,
)

from ..builders import written

_LIB = {"lib.py": "def b(i):\n    return i\n"}
_DUPLICATES = """from lib import b


def test_b_is_positive():
    assert b(1) > 0


def test_b_stays_positive():
    assert b(1) > 0


def test_b_is_still_positive():
    assert b(1) > 0


def test_b_is_bounded():
    assert b(1) < 100
"""
_KEPT = """from lib import b


def test_b_is_positive():
    assert b(1) > 0


def test_b_is_bounded():
    assert b(1) < 100
"""
_METHODS = """from lib import b


class TestB:
    def test_one(self):
        assert b(2) > 0

    def test_two(self):
        assert b(2) > 0

    def test_three(self):
        assert b(2) > 0
"""
_METHOD = """from lib import b


class TestB:
    def test_one(self):
        assert b(2) > 0
"""
_TRIO = """def f(x):
    return x + 1


def test_f_one():
    assert f(1) == 2


def test_f_two():
    assert f(2) == 3


def test_f_three():
    assert f(3) == 4
"""
_TRIO_IN_CLASS = """def f(x):
    return x + 1


class TestF:
    def test_f_one(self):
        assert f(1) == 2

    def test_f_two(self):
        assert f(2) == 3

    def test_f_three(self):
        assert f(3) == 4
"""


class DropsWithoutSaying(DuplicateTestIntentClusterCount):
    """The duplicate tests rule with no word about what its fix changed."""

    change = None


def _fixed(root: Path, rule: type[Rule]) -> list[str]:
    """What the fixpoint reports fixing the project at `root` with `rule` and writing it."""
    load_catalog()
    with Lint(root, (rule,)) as lint:
        return list(Fixpoint(lint, root, ("review",)).run(write=True))


@pytest.mark.parametrize(
    ("source", "kept", "listed"),
    [
        pytest.param(_DUPLICATES, _KEPT, 2, id="duplicates-of-the-module"),
        pytest.param(_METHODS, _METHOD, 1, id="duplicates-of-a-class"),
    ],
)
def test_duplicate_tests_a_rule_deletes_are_the_ones_its_change_line_names(
    tmp_path: Path, source: str, kept: str, listed: int
) -> None:
    root = written(tmp_path, {**_LIB, "test_b.py": source})

    report = _fixed(root, DuplicateTestIntentClusterCount)

    assert (root / "test_b.py").read_text(encoding="utf-8") == kept, report
    assert (
        f"collection: {listed} test ids listed from the mirror, 1 components judged, 0 undone"
        in report
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(_TRIO, id="tests-of-the-module"),
        pytest.param(_TRIO_IN_CLASS, id="tests-of-a-class"),
    ],
)
def test_tests_a_rule_merges_are_kept_as_cases_its_change_line_names(
    tmp_path: Path, source: str
) -> None:
    root = written(tmp_path, {"test_f.py": source})

    report = _fixed(root, ParametrizationCandidateGroupCount)

    assert "@pytest.mark.parametrize" in (root / "test_f.py").read_text(encoding="utf-8"), report
    assert "collection: 3 test ids listed from the mirror, 1 components judged, 0 undone" in report


def test_a_fix_that_drops_tests_without_naming_them_is_undone_before_anything_is_written(
    tmp_path: Path,
) -> None:
    root = written(tmp_path, {**_LIB, "test_b.py": _DUPLICATES})

    report = _fixed(root, DropsWithoutSaying)

    assert (root / "test_b.py").read_text(encoding="utf-8") == _DUPLICATES
    assert any(
        line.startswith("collection: undid the fixes of 1 files (test_b.py): lose 2 test ids")
        for line in report
    ), report


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(DuplicateTestIntentClusterCount, id="the-duplicates-of-one-intent"),
        pytest.param(ProductionReachHotspotCount, id="the-hotspot-of-few-intents"),
    ],
)
def test_every_example_of_the_rules_that_delete_or_merge_tests_passes(
    capsys: pytest.CaptureFixture[str], rule: type[Rule]
) -> None:
    with suppress(SystemExit):
        verbs.test(selection=verbs.Selection(rule=(rule.id,)))

    shown = capsys.readouterr().out.splitlines()
    assert shown[:-1] == [f"{rule.id} example {n}: pass" for n in range(1, len(rule.examples) + 1)]
    assert shown[-1].split()[0] == "0"
