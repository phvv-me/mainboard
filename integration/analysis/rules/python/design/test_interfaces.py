"""PY-INTE0001 checks against the capability a block needs in place of the concrete types it
named, importing each capability, and leaves the checks it cannot respell."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ...builders import fixed
from .builders import abstracted, checked, checks, concrete_types, stands_in

_RULE = "PY-INTE0001"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `checks()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in checks().items()}
    return fixed(tmp_path_factory.mktemp("interfaces"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("carriage_returns.py", id="a-check-in-a-crlf-file"),
        pytest.param("present.py", id="the-capability-is-already-imported"),
        pytest.param("twice.py", id="two-checks-share-one-import"),
        pytest.param("attribute_member.py", id="a-dotted-type-stays-in-its-tuple"),
        pytest.param("taken.py", id="the-capability-name-is-a-local-variable"),
        pytest.param("aliased.py", id="the-capability-is-imported-under-another-name"),
        pytest.param("unknown.py", id="a-call-among-the-types"),
        pytest.param("only_str.py", id="types-with-no-capability"),
    ],
)
def test_a_check_is_respelled_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == checks()[name][1]


@settings(max_examples=6)
@given(
    types=st.lists(st.sampled_from(concrete_types()), min_size=1, max_size=4, unique=True).filter(
        stands_in
    )
)
def test_every_concrete_type_becomes_its_capability_named_once_and_imported(
    tmp_path_factory: pytest.TempPathFactory, types: list[str]
) -> None:
    sources = {"check.py": checked(types)}

    result = fixed(tmp_path_factory.mktemp("types"), sources, rules=(_RULE,))

    assert result["check.py"] == abstracted(types)
