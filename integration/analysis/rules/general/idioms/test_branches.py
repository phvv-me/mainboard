"""ALL-BRAN0001 turns an if chain over one subject and distinct literals into a match, headers
and indents only, and leaves the chains whose arms a match would read differently."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ...builders import fixed
from .builders import cases, chain, matched

_RULE = "ALL-BRAN0001"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `cases()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in cases().items()}
    return fixed(tmp_path_factory.mktemp("branches"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("two_spaces.py", id="an-indent-of-two-spaces-is-kept"),
        pytest.param("loop.py", id="arms-that-continue-and-break-a-loop"),
        pytest.param("between.py", id="a-comment-between-arms-stays-where-it-was"),
        pytest.param("carriage_returns.py", id="a-chain-in-a-crlf-file"),
        pytest.param("mixed.py", id="strings-integers-and-floats-in-one-chain"),
        pytest.param("nested.py", id="a-chain-inside-an-arm-of-another"),
        pytest.param("noted.py", id="a-comment-after-a-header-colon"),
        pytest.param("parenthesized.py", id="headers-in-parentheses"),
        pytest.param("uneven.py", id="an-arm-indented-by-another-width"),
        pytest.param("other_subject.py", id="an-arm-testing-another-subject"),
    ],
)
def test_a_chain_becomes_a_match_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == cases()[name][1]


@settings(max_examples=6)
@given(
    values=st.lists(st.integers(min_value=0, max_value=99), min_size=3, max_size=6, unique=True),
    default=st.booleans(),
)
def test_every_literal_becomes_a_case_and_the_else_the_wildcard(
    tmp_path_factory: pytest.TempPathFactory, values: list[int], default: bool
) -> None:
    sources = {"label.py": chain(values, default=default)}

    result = fixed(tmp_path_factory.mktemp("chain"), sources, rules=(_RULE,))

    assert result["label.py"] == matched(values, default=default)
