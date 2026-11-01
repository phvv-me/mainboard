"""PY-PYDA0001 turns an after validator of a model that reads one field into a validator of that
field, whole file or none, and leaves the validators that use more of the model than a field."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ...builders import fixed
from .builders import cases, narrowed, validated

_RULE = "PY-PYDA0001"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `cases()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in cases().items()}
    return fixed(tmp_path_factory.mktemp("validation"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("typed.py", id="a-return-annotation-of-self-takes-the-field-type"),
        pytest.param("carriage_returns.py", id="a-validator-in-a-crlf-file"),
        pytest.param("two.py", id="two-validators-of-two-fields"),
        pytest.param("qualified.py", id="the-decorator-reached-through-the-module"),
        pytest.param("decorated.py", id="a-second-decorator"),
        pytest.param("nested.py", id="a-nested-function-reading-the-field"),
        pytest.param("other_return.py", id="a-return-of-something-else"),
        pytest.param("annotated.py", id="a-return-annotation-other-than-self"),
        pytest.param("before.py", id="a-before-validator"),
    ],
)
def test_a_validator_is_narrowed_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == cases()[name][1]


@settings(max_examples=6)
@given(fields=st.lists(st.sampled_from(["a", "b", "size", "count"]), min_size=1, unique=True))
def test_every_validator_reads_its_field_as_an_argument(
    tmp_path_factory: pytest.TempPathFactory, fields: list[str]
) -> None:
    sources = {"box.py": validated(fields)}

    result = fixed(tmp_path_factory.mktemp("fields"), sources, rules=(_RULE,))

    assert result["box.py"] == narrowed(fields)
