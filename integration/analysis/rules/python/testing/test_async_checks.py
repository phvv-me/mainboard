"""PY-TEST0009 makes a test that runs its own loop an async test awaiting where it ran, whole file
or none, and leaves the tests the anyio plugin cannot own."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ...builders import fixed
from .builders import asynchronous, cases, synchronous

_RULE = "PY-TEST0009"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `cases()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in cases().items()}
    return fixed(tmp_path_factory.mktemp("async_checks"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("test_attribute.py", id="an-attribute-of-the-result"),
        pytest.param("test_subscript.py", id="an-item-of-the-result"),
        pytest.param("test_comprehension.py", id="a-list-comprehension"),
        pytest.param("test_carriage_returns.py", id="a-test-in-a-crlf-file"),
        pytest.param("test_two.py", id="two-tests-and-two-runs-in-one-test"),
        pytest.param("test_decorated.py", id="the-mark-goes-below-the-other-decorators"),
        pytest.param("test_method.py", id="a-method-of-a-plain-class"),
        pytest.param("test_lambda.py", id="a-run-inside-a-lambda"),
        pytest.param("test_generator.py", id="a-run-inside-a-generator-expression"),
        pytest.param("test_aliased.py", id="run-imported-by-name"),
        pytest.param("test_marked.py", id="a-test-already-under-the-anyio-mark"),
        pytest.param("test_given.py", id="a-test-under-given"),
        pytest.param("test_tabs.py", id="a-method-indented-with-tabs"),
        pytest.param("test_nested.py", id="a-run-in-a-nested-function-is-not-the-tests-own"),
    ],
)
def test_a_test_becomes_async_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == cases()[name][1]


def test_the_rule_for_unowned_async_tests_finds_the_fixed_file_owned(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    sources = {"test_attribute.py": cases()["test_attribute.py"][0]}

    result = fixed(tmp_path_factory.mktemp("owned"), sources, rules=(_RULE, "PY-TEST0010"))

    assert result["test_attribute.py"] == cases()["test_attribute.py"][1]


@settings(max_examples=6)
@given(forms=st.lists(st.sampled_from(["{}", "value = {}", "assert {}"]), min_size=1, max_size=4))
def test_every_form_of_running_a_coroutine_becomes_its_await(
    tmp_path_factory: pytest.TempPathFactory, forms: list[str]
) -> None:
    sources = {"test_forms.py": synchronous(forms)}

    result = fixed(tmp_path_factory.mktemp("forms"), sources, rules=(_RULE,))

    assert result["test_forms.py"] == asynchronous(forms)
