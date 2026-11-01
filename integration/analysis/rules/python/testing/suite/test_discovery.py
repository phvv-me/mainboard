"""PY-TEST0002 moves a test from the legacy `tmpdir` to `tmp_path` when every use of `tmpdir` is
one the pathlib API spells the same, whole file or none, and leaves the rest."""

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ....builders import fixed
from .builders import legacy_chain, legacy_tmpdirs, modern_chain

_RULE = "PY-TEST0002"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str]:
    """The files of `legacy_tmpdirs()` after one fix of the rule over them all."""
    sources = {name: source for name, (source, _) in legacy_tmpdirs().items()}
    return fixed(tmp_path_factory.mktemp("discovery"), sources, rules=(_RULE,))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("test_alias.py", id="a-variable-holding-a-joined-path"),
        pytest.param("test_chain.py", id="joins-written-read-and-spelled"),
        pytest.param("test_carriage_returns.py", id="a-tmpdir-test-in-a-crlf-file"),
        pytest.param("test_two.py", id="two-tests-of-one-file"),
        pytest.param("test_closure.py", id="a-nested-function-using-the-directory"),
        pytest.param("test_fixture_home.py", id="a-fixture-handing-the-directory-on-as-text"),
        pytest.param("test_method_dir.py", id="a-method-of-a-class"),
        pytest.param("test_ensure.py", id="a-method-the-pathlib-api-spells-otherwise"),
        pytest.param("test_bare.py", id="the-directory-passed-to-a-function"),
        pytest.param("test_remove.py", id="a-variable-used-in-an-unknown-way"),
        pytest.param("test_factory.py", id="a-test-beside-one-taking-the-factory"),
        pytest.param("test_named.py", id="tmp-path-requested-too"),
        pytest.param("test_annotated.py", id="an-annotated-parameter"),
        pytest.param("test_two_arguments.py", id="a-join-of-two-names"),
        pytest.param("test_sum.py", id="a-join-of-a-sum"),
        pytest.param("test_mode.py", id="a-write-in-binary-mode"),
        pytest.param("test_returned.py", id="a-joined-path-returned-by-a-fixture"),
        pytest.param("test_reassigned.py", id="a-variable-assigned-twice"),
    ],
)
def test_a_test_moves_to_tmp_path_or_stays_exactly_as_the_case_states(
    project: Mapping[str, str], name: str
) -> None:
    assert project[name] == legacy_tmpdirs()[name][1]


@settings(max_examples=6)
@given(names=st.lists(st.sampled_from(["a", "b.txt", "data", "x1"]), min_size=1, max_size=4))
def test_every_joined_write_becomes_a_division_and_a_text_write(
    tmp_path_factory: pytest.TempPathFactory, names: list[str]
) -> None:
    sources = {"test_files.py": legacy_chain(names)}

    result = fixed(tmp_path_factory.mktemp("chain"), sources, rules=(_RULE,))

    assert result["test_files.py"] == modern_chain(names)
