"""PY-MODE0004 dumps a model through `model_dump` in place of a dict repeating its fields, when
the facts prove the object a model whose own scalar fields they are, and leaves the rest."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ...builders import fixed
from .builders import dumped, modeled, projections

_RULE = "PY-MODE0004"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `projections()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in projections().items()}
    return fixed(tmp_path_factory.mktemp("models"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("spot.py", id="self-with-the-keys-in-another-order"),
        pytest.param("annotated.py", id="a-parameter-annotated-with-a-model-of-another-file"),
        pytest.param("carriage_returns.py", id="a-projection-in-a-crlf-file"),
        pytest.param("plain.py", id="a-parameter-without-an-annotation"),
        pytest.param("dashed.py", id="a-key-that-differs-from-its-attribute"),
        pytest.param("spread.py", id="an-entry-spreading-another-dict"),
        pytest.param("union.py", id="an-annotation-that-may-be-none"),
        pytest.param("commented.py", id="a-comment-among-the-entries"),
        pytest.param("keywords.py", id="a-call-in-place-of-a-dict-display"),
    ],
)
def test_a_projection_becomes_a_dump_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == projections()[name][1]


@settings(max_examples=6)
@given(fields=st.lists(st.sampled_from(list("abcde")), min_size=4, max_size=5, unique=True))
def test_every_field_is_kept_in_the_order_the_projection_wrote_it(
    tmp_path_factory: pytest.TempPathFactory, fields: list[str]
) -> None:
    sources = {"row.py": modeled(fields)}

    result = fixed(tmp_path_factory.mktemp("fields"), sources, rules=(_RULE,))

    assert result["row.py"] == dumped(fields)
