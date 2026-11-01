"""The rules whose fix deletes a node: each fix example and refusal example the rule states passes
in `lint test`, and the fix leaves alone what it cannot delete whole."""

from collections.abc import Callable
from contextlib import suppress

import pytest

from mainboard.analysis import Example, Rule
from mainboard.analysis.engine.lint import verbs
from mainboard.analysis.rules.general.bundles import InteropReach
from mainboard.analysis.rules.general.control.flow import StatementWithoutEffect
from mainboard.analysis.rules.general.idioms.strings import DecorativeRepeatedSeparatorCount
from mainboard.analysis.rules.python.classes.inheritance import PassThroughInheritanceLayerCount

from .constants import BRACED_IF, UNBRACED_LOOP

type Verdict = Callable[[type[Rule], Example], str]

_BANNER = 'def banner():\n    print("=" * 60)\n    print("-" * 60)\n'
_LAYERS = """class Base:
    def run(self):
        return 1


class Mid(Base):
    pass


class Leaf(Mid):
    def run(self):
        return 2


class Other(Mid):
    def run(self):
        return 3
"""
_ONE_LAYER = (
    "class Base:\n    pass\n\n\nclass Mid(Base):\n    pass\n\n\nclass Leaf(Mid):\n    x = 1\n"
)
_KEYWORD = _ONE_LAYER.replace("Mid(Base)", "Mid(Base, metaclass=type)")
_CHAIN = """class Base:
    x = 1


class Mid1(Base):
    pass


class Mid2(Mid1):
    pass


class Leaf(Mid2):
    y = 2
"""
_KERNEL = "__global__ void dead(int *p) {\n    p[0] = 1;\n}\n"
_HANDLED = (
    "def has(obj):\n    try:\n        run(obj)\n    except AttributeError:\n{}\n    return True\n"
)
_NESTED_PROBE = """def has(obj, deep):
    try:
        if deep:
            obj.attr
    except AttributeError:
        return False
    return True
"""
_TS_PROBE = """function has(obj: any): boolean {
  try {
    obj.attr;
  } catch (error) {
    return false;
  }
  return true;
}
"""


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(StatementWithoutEffect, id="a-statement-without-effect"),
        pytest.param(DecorativeRepeatedSeparatorCount, id="a-decorative-separator"),
        pytest.param(InteropReach, id="a-kernel-no-other-language-reaches"),
        pytest.param(PassThroughInheritanceLayerCount, id="a-pass-through-layer"),
    ],
)
def test_every_example_of_the_rules_fixed_by_deletion_passes(
    capsys: pytest.CaptureFixture[str], rule: type[Rule]
) -> None:
    with suppress(SystemExit):
        verbs.test(selection=verbs.Selection(rule=(rule.id,)))

    said = capsys.readouterr().out.splitlines()
    assert {line.partition(": ")[2] for line in said[:-1]} == {"pass"}
    assert len(said) == len(rule.examples) + 1


@pytest.mark.parametrize(
    ("rule", "example"),
    [
        pytest.param(
            DecorativeRepeatedSeparatorCount,
            Example(bad=_BANNER, good="def banner():\n    pass\n"),
            id="two-separators-of-one-body-go-in-two-passes",
        ),
        pytest.param(
            DecorativeRepeatedSeparatorCount,
            Example(bad='SEPARATOR = "=" * 60\n', good='SEPARATOR = "=" * 60\n'),
            id="a-separator-bound-to-a-name-stays",
        ),
        pytest.param(
            DecorativeRepeatedSeparatorCount,
            Example(
                bad='def banner(out):\n    print("=" * 60, file=out)\n',
                good='def banner(out):\n    print("=" * 60, file=out)\n',
            ),
            id="a-call-with-other-arguments-stays",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(
                bad="void f(int x) {\n    x == 1;\n    return;\n}\n",
                good="void f(int x) {\n    return;\n}\n",
                path="m.c",
            ),
            id="a-c-statement-goes-without-a-filler",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(
                bad="class A:\n    x.a\n    x.b\n\n    def run(self):\n        return 1\n",
                good="class A:\n    def run(self):\n        return 1\n",
            ),
            id="two-statements-of-a-class-body-before-a-method",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(
                bad=BRACED_IF,
                good=BRACED_IF.replace("        x == 1;\n", ""),
                path="m.c",
            ),
            id="a-c-statement-in-the-braces-of-an-if-goes",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(bad=UNBRACED_LOOP, good=UNBRACED_LOOP, path="kernel.cu"),
            id="the-body-of-a-cuda-loop-with-no-braces-stays",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(bad=_NESTED_PROBE, good=_NESTED_PROBE),
            id="a-probe-nested-in-the-body-of-a-try-stays",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(bad=_TS_PROBE, good=_TS_PROBE, path="m.ts"),
            id="a-probe-in-the-body-of-a-typescript-try-stays",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(
                bad=_HANDLED.format("        obj.attr\n        return False"),
                good=_HANDLED.format("        return False"),
            ),
            id="a-statement-of-a-handler-goes",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(
                bad=_HANDLED.format(
                    "        return False\n    else:\n        obj.attr\n        return True"
                ),
                good=_HANDLED.format("        return False\n    else:\n        return True"),
            ),
            id="a-statement-of-the-else-of-a-try-goes",
        ),
        pytest.param(
            StatementWithoutEffect,
            Example(
                bad=_HANDLED.format(
                    "        return False\n    finally:\n        obj.attr\n        close()"
                ),
                good=_HANDLED.format("        return False\n    finally:\n        close()"),
            ),
            id="a-statement-of-the-finally-of-a-try-goes",
        ),
        pytest.param(
            InteropReach,
            Example(bad=_KERNEL + "\nvoid run() {}\n", good="void run() {}\n", path="kernel.cu"),
            id="a-kernel-named-nowhere-else-goes-with-the-blank-line-after-it",
        ),
        pytest.param(
            InteropReach,
            Example(
                bad=_KERNEL.replace("dead", "alpha")
                + "\n"
                + _KERNEL.replace("dead", "beta")
                + "\nvoid run() {}\n",
                good="void run() {}\n",
                path="kernel.cu",
            ),
            id="two-kernels-no-language-reaches",
        ),
        pytest.param(
            InteropReach,
            Example(
                bad=_KERNEL + "\nvoid run(int *p);\n",
                good=_KERNEL + "\nvoid run(int *p);\n",
                path="kernel.cu",
                context={"kernel.h": "__global__ void dead(int *p);\n"},
            ),
            id="a-kernel-a-header-names-stays",
        ),
        pytest.param(
            PassThroughInheritanceLayerCount,
            Example(
                bad=_LAYERS,
                good=_LAYERS.replace("class Mid(Base):\n    pass\n\n\n", "").replace(
                    "(Mid)", "(Base)"
                ),
            ),
            id="every-subclass-of-the-layer-takes-its-base",
        ),
        pytest.param(
            PassThroughInheritanceLayerCount,
            Example(
                bad=_CHAIN,
                good="class Base:\n    x = 1\n\n\nclass Leaf(Base):\n    y = 2\n",
            ),
            id="a-chain-of-two-layers-goes-in-two-passes",
        ),
        pytest.param(
            PassThroughInheritanceLayerCount,
            Example(
                bad=_ONE_LAYER + "\n\nVALUE: Mid = Leaf()\n",
                good=_ONE_LAYER + "\n\nVALUE: Mid = Leaf()\n",
            ),
            id="a-layer-named-in-an-annotation-stays",
        ),
        pytest.param(
            PassThroughInheritanceLayerCount,
            Example(
                bad=_ONE_LAYER + '\n__all__ = ["Mid"]\n', good=_ONE_LAYER + '\n__all__ = ["Mid"]\n'
            ),
            id="a-layer-in-the-export-list-stays",
        ),
        pytest.param(
            PassThroughInheritanceLayerCount,
            Example(bad=_KEYWORD, good=_KEYWORD),
            id="a-layer-with-a-keyword-stays",
        ),
    ],
)
def test_the_fix_of_a_rule_deletes_only_what_the_rule_may_delete(
    verdict: Verdict, rule: type[Rule], example: Example
) -> None:
    assert verdict(rule, example) == "pass"
