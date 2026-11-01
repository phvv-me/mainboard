"""`lint test` runs a rule's examples. A `good` equal to its `bad` states a refusal: the fix must
leave the file as it was, so the example passes only when the rule's fix changes nothing. The
examples that define a test are written as `test_module.py`, where the fixpoint also lists the
test ids the fix leaves, so every rule fixing a test module runs its own examples through it."""

from contextlib import suppress

import pytest

from mainboard.analysis import Example
from mainboard.analysis.engine.lint import selected, verbs
from mainboard.analysis.rules.python.libraries.logging import TextAroundTheLogger

_ID = TextAroundTheLogger.id
# The fixable Python rules with an example that defines a test, which `verbs.test` writes as
# `test_module.py` unless the example names its path.
_TEST_FIXERS = sorted(
    spec.id
    for spec in selected(())
    if spec.fixes() != "none"
    and spec().language in (None, "python")
    and any("def test_" in example.bad for example in spec.examples)
)


@pytest.mark.parametrize(
    ("snippet", "said"),
    [
        pytest.param(
            'import sys\n\nprint("args", sys.argv)\n',
            f"{_ID} example 1: pass",
            id="a-fix-that-refuses",
        ),
        pytest.param(
            'print("hello")\n',
            f"{_ID} example 1: the fix does not give good",
            id="a-fix-that-edits",
        ),
    ],
)
def test_an_example_whose_good_is_its_bad_passes_only_when_the_fix_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], snippet: str, said: str
) -> None:
    monkeypatch.setattr(TextAroundTheLogger, "examples", (Example(bad=snippet, good=snippet),))

    with suppress(SystemExit):
        verbs.test(selection=verbs.Selection(rule=(_ID,)))

    assert said in capsys.readouterr().out


@pytest.mark.parametrize("rule", _TEST_FIXERS)
def test_every_example_of_a_rule_that_fixes_a_test_module_passes_through_the_fixpoint(
    capsys: pytest.CaptureFixture[str], rule: str
) -> None:
    with suppress(SystemExit):
        verbs.test(selection=verbs.Selection(rule=(rule,)))

    assert "0 examples failed" in capsys.readouterr().out
