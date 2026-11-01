"""The rules whose fix drops an argument: each example a rule states, and each more this module
states, passes in `lint test`; the keyword a CUDA rule names is the compile call's own; and the fix
leaves alone what it cannot drop whole."""

from collections.abc import Callable

import pytest

from mainboard.analysis import Example, Rule
from mainboard.analysis.rules.python.classes.inheritance import (
    HazardousMultipleInheritanceMroCount,
)
from mainboard.analysis.rules.python.cuda.framework import (
    DebugBuildCommitted,
    HardcodedHelionConfig,
)

type Verdict = Callable[[type[Rule], Example], str]

_DROPPING = (DebugBuildCommitted, HardcodedHelionConfig, HazardousMultipleInheritanceMroCount)

_NUMBA = "from numba import cuda\n\n\n{}\ndef kernel(x):\n    x[0] = 1\n"
_HELION = "import helion\n\n\n{}\ndef scale(x):\n    return x\n"
_CHAIN = "class A2:\n    pass\n\n\nclass A1(A2):\n    pass\n\n\nclass B(A1):\n    pass\n\n\n{}\n"
_SPREAD = (
    "@helion.kernel(\n    config=helion.Config(block_sizes=[16]),\n    static_shapes=True,\n)"
)


@pytest.mark.parametrize(
    ("rule", "example"),
    [
        *(
            pytest.param(rule, declared, id=f"{rule.id}-example-{number}")
            for rule in _DROPPING
            for number, declared in enumerate(rule.examples, 1)
        ),
        pytest.param(
            DebugBuildCommitted,
            Example(
                bad=_NUMBA.format("@cuda.jit(debug=True, opt=False)"),
                good=_NUMBA.format("@cuda.jit"),
            ),
            id="two-flags-of-one-call-go-in-two-passes",
        ),
        pytest.param(
            DebugBuildCommitted,
            Example(
                bad=_NUMBA.format("@cuda.jit(device=True, debug=True)"),
                good=_NUMBA.format("@cuda.jit(device=True)"),
            ),
            id="a-flag-after-another-keyword",
        ),
        pytest.param(
            DebugBuildCommitted,
            Example(
                bad=_NUMBA.format("@cuda.jit(\n    debug=True,\n    fastmath=True,\n)"),
                good=_NUMBA.format("@cuda.jit(\n    fastmath=True,\n)"),
            ),
            id="a-flag-on-a-line-of-its-own",
        ),
        pytest.param(
            DebugBuildCommitted,
            Example(
                bad=_NUMBA.format("@cuda.jit(  # slow\n    debug=True)"),
                good=_NUMBA.format("@cuda.jit(  # slow\n    debug=True)"),
            ),
            id="a-comment-in-the-parentheses-stays",
        ),
        pytest.param(
            HardcodedHelionConfig,
            Example(
                bad=_HELION.format(_SPREAD),
                good=_HELION.format("@helion.kernel(\n    static_shapes=True,\n)"),
            ),
            id="a-config-on-a-line-of-its-own",
        ),
        pytest.param(
            HardcodedHelionConfig,
            Example(
                bad=_HELION.format("@helion.kernel(config=helion.Config(block_sizes=[16]))"),
                good=_HELION.format("@helion.kernel"),
            ),
            id="a-config-written-out-as-the-only-argument",
        ),
        pytest.param(
            HazardousMultipleInheritanceMroCount,
            Example(
                bad=_CHAIN.format("class C(B, A1, A2):\n    pass"),
                good=_CHAIN.format("class C(B, A1, A2):\n    pass"),
            ),
            id="two-redundant-bases-stay-since-dropping-one-clears-no-finding",
        ),
        pytest.param(
            HazardousMultipleInheritanceMroCount,
            Example(
                bad=_CHAIN.format("class C(B, A2):\n    pass"),
                good=_CHAIN.format("class C(B):\n    pass"),
            ),
            id="a-base-the-base-before-it-inherits-through-two-levels",
        ),
        pytest.param(
            HazardousMultipleInheritanceMroCount,
            Example(
                bad=_CHAIN.format("class C(B, A2, Exception):\n    pass"),
                good=_CHAIN.format("class C(B, A2, Exception):\n    pass"),
            ),
            id="a-redundant-base-another-base-follows-stays",
        ),
    ],
)
def test_every_example_of_the_rules_fixed_by_dropping_an_argument_passes(
    verdict: Verdict, rule: type[Rule], example: Example
) -> None:
    assert verdict(rule, example) == "pass"


@pytest.mark.parametrize(
    ("rule", "source"),
    [
        pytest.param(
            DebugBuildCommitted,
            _NUMBA.format("@cuda.jit(wrap(debug=True))"),
            id="a-flag-of-a-call-nested-in-the-arguments",
        ),
        pytest.param(
            DebugBuildCommitted,
            _NUMBA.format("@other(debug=True)\n@cuda.jit"),
            id="a-flag-of-another-decorator",
        ),
        pytest.param(
            DebugBuildCommitted,
            _NUMBA.format("@cuda.jit(debug=False)"),
            id="the-debug-flag-turned-off",
        ),
        pytest.param(
            DebugBuildCommitted,
            _NUMBA.format("@cuda.jit(opt=True)"),
            id="the-optimizer-turned-on",
        ),
        pytest.param(
            HardcodedHelionConfig,
            _HELION.format("@helion.kernel(static_shapes=wrap(config=helion.Config()))"),
            id="a-config-of-a-call-nested-in-the-arguments",
        ),
        pytest.param(
            HardcodedHelionConfig,
            _HELION.format("@other(config=helion.Config())\n@helion.kernel"),
            id="a-config-of-another-decorator",
        ),
    ],
)
def test_a_keyword_the_compile_call_does_not_hold_is_no_finding_of_its_rule(
    verdict: Verdict, rule: type[Rule], source: str
) -> None:
    assert verdict(rule, Example(bad=source)) == "bad is not flagged"
