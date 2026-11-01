"""PY-TEST0014 draws a wide parametrized range from Hypothesis integers, whole file or none, and
leaves the ranges and tests Hypothesis would read differently."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ....builders import fixed
from .builders import drawn, ranged, wide_ranges

_RULE = "PY-TEST0014"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `wide_ranges()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in wide_ranges().items()}
    return fixed(tmp_path_factory.mktemp("coverage"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("test_basic.py", id="a-start-and-stop-and-a-second-decorator"),
        pytest.param("test_method.py", id="a-method-taking-self"),
        pytest.param("test_present.py", id="hypothesis-already-imported"),
        pytest.param("test_carriage_returns.py", id="a-range-in-a-crlf-file"),
        pytest.param("test_imported.py", id="the-mark-imported-by-name"),
        pytest.param("test_tuple.py", id="several-argument-names"),
        pytest.param("test_ids.py", id="a-keyword-in-the-decorator"),
        pytest.param("test_noted.py", id="a-comment-in-the-decorator"),
        pytest.param("test_names.py", id="st-is-a-variable"),
        pytest.param("test_fixture.py", id="a-fixture-requested"),
        pytest.param("test_small.py", id="a-range-below-the-minimum"),
        pytest.param("test_mixed.py", id="a-fixable-test-beside-one-requesting-a-fixture"),
    ],
)
def test_a_range_becomes_integers_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == wide_ranges()[name][1]


@settings(max_examples=6)
@given(
    bounds=st.lists(
        st.tuples(st.integers(0, 50), st.integers(10, 40)).map(lambda pair: (pair[0], sum(pair))),
        min_size=1,
        max_size=3,
    )
)
def test_every_range_becomes_integers_ending_before_its_stop(
    tmp_path_factory: pytest.TempPathFactory, bounds: list[tuple[int, int]]
) -> None:
    sources = {"test_ranges.py": ranged(bounds)}

    result = fixed(tmp_path_factory.mktemp("ranges"), sources, rules=(_RULE,))

    assert result["test_ranges.py"] == drawn(bounds)
