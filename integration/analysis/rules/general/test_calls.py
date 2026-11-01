"""ALL-CALL0002 passes the project's one timeout to a configured blocking call that passes none,
after the arguments it has, and leaves the calls whose arguments it cannot extend."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ..builders import fixed
from .builders import cases
from .constants import CONFIG

_RULE = "ALL-CALL0002"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `cases()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in cases().items()}
    return fixed(tmp_path_factory.mktemp("calls"), sources, rules=(_RULE,), config=CONFIG)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("nested.py", id="a-call-inside-the-arguments-of-another"),
        pytest.param("carriage_returns.py", id="a-call-over-lines-in-a-crlf-file"),
        pytest.param("tuple.py", id="a-tuple-as-the-only-argument"),
        pytest.param("decorator.py", id="a-call-that-decorates"),
        pytest.param("annotated.py", id="comments-among-the-arguments-stay-where-they-are"),
        pytest.param("spaced.py", id="blanks-inside-the-parentheses"),
        pytest.param("generator.py", id="a-bare-generator-has-no-argument-list"),
        pytest.param("multiline.py", id="an-argument-holding-a-string-over-two-lines"),
        pytest.param("splat.py", id="a-call-spreading-keywords"),
        pytest.param("starred.py", id="a-call-spreading-arguments"),
        pytest.param("closing.py", id="a-comment-closing-the-arguments"),
    ],
)
def test_a_call_without_a_bound_gets_one_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == cases()[name][1]


@settings(max_examples=6)
@given(
    positional=st.lists(st.sampled_from(["a", "b.c", "f(x)", "[a, b]"]), max_size=3),
    keywords=st.lists(st.sampled_from(["k=1", 'j="t"', "m=f(x)"]), max_size=2, unique=True),
    trailing=st.booleans(),
)
def test_the_bound_follows_the_arguments_whatever_they_are(
    tmp_path_factory: pytest.TempPathFactory,
    positional: list[str],
    keywords: list[str],
    trailing: bool,
) -> None:
    arguments = ", ".join([*positional, *keywords])
    comma = "," if arguments and trailing else ""
    head = "import requests\n\n\ndef run(a, b, x):\n    return requests.get("
    sources = {"run.py": f"{head}{arguments}{comma})\n"}

    result = fixed(tmp_path_factory.mktemp("bound"), sources, rules=(_RULE,), config=CONFIG)

    assert result["run.py"] == f"{head}{', '.join([*positional, *keywords, 'timeout=2.5'])})\n"
