"""The groups the merging rules offer: the tests of one holder, shape, marks and reach whose cases
all differ, merged under the words their names share, and left alone when no name can be given."""

import os
from collections.abc import Callable, Iterator

import duckdb
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.analysis.rules.python.testing.cases import CommonPrefix, CommonSuffix

from .builders import named, preamble, suite, trio
from .readers import Merged, outcomes, parametrized

type Fix = Callable[..., Merged]
type Share = Callable[..., tuple[str, str]]

_ASSERT = "    assert f({a}) == [{a}, None]"
_WORD = st.sampled_from(["a", "b", "c", "d"])
_NAME = st.lists(_WORD, min_size=1, max_size=5).map("_".join)
_LIB = {"lib.py": "def b(i):\n    return i\n"}
_SLOW = '@pytest.mark.skipif(False, reason="slow")\ndef test_f_two'
_METHODS = "\n".join(
    f"    def test_f_{word}(self):\n        assert f({number}) == [{number}, None]\n"
    for word, number in (("one", 4), ("two", 5), ("three", 6))
)
_POSITIVE = "\n\n".join(
    f"def test_positive_{word}():\n    assert b({number}) > 0\n"
    for word, number in (("one", 1), ("two", 2), ("three", 3))
)
_BOUNDED = "\n\n".join(
    f"def test_bounded_{word}():\n    assert b({number}) > 0\n    assert b({number}) < 100\n"
    for word, number in (("one", 1), ("two", 2))
)
_SINGLE = "def test_single():\n    assert b(9) > 0\n    assert b(9) < 99\n    assert b(9) < 98\n"


@pytest.fixture(scope="module")
def shared() -> Iterator[Share]:
    """The words two names share at their start and at their end, as SQL finds them."""
    connection = duckdb.connect()
    CommonPrefix.create(connection)
    CommonSuffix.create(connection)

    def ask(first: str, *, second: str) -> tuple[str, str]:
        row = connection.execute(
            "SELECT common_prefix(?, ?), common_suffix(?, ?)", [first, second, first, second]
        ).fetchone()
        assert row is not None
        return row

    yield ask
    connection.close()


@given(first=_NAME, second=_NAME)
def test_the_words_two_names_share_at_each_end_are_found_by_word_and_not_by_letter(
    shared: Share, first: str, second: str
) -> None:
    words = [first.split("_"), second.split("_")]

    found = shared(first, second=second)

    assert found == (
        "_".join(os.path.commonprefix(words)),
        "_".join(reversed(os.path.commonprefix([list(reversed(each)) for each in words]))),
    )


@pytest.mark.parametrize(
    ("names", "renamed"),
    [
        pytest.param(
            ("test_f_one", "test_f_two", "test_f_three"),
            (
                "test_f_one -> test_f[one]",
                "test_f_two -> test_f[two]",
                "test_f_three -> test_f[three]",
            ),
            id="the-words-the-names-start-with",
        ),
        pytest.param(
            (
                "test_admin_health_is_absent",
                "test_admin_doctor_is_absent",
                "test_admin_usage_is_absent",
            ),
            (
                "test_admin_health_is_absent -> test_admin_is_absent[health]",
                "test_admin_doctor_is_absent -> test_admin_is_absent[doctor]",
                "test_admin_usage_is_absent -> test_admin_is_absent[usage]",
            ),
            id="the-words-the-names-start-and-end-with",
        ),
        pytest.param(
            ("test_f_a_x", "test_f_a_y", "test_f_a_z"),
            (
                "test_f_a_x -> test_f_a[x]",
                "test_f_a_y -> test_f_a[y]",
                "test_f_a_z -> test_f_a[z]",
            ),
            id="a-word-every-case-starts-with-joins-the-name",
        ),
        pytest.param(
            ("test_f_a_b_c", "test_f_d_c", "test_f_e_c"),
            (
                "test_f_a_b_c -> test_f_c[a-b]",
                "test_f_d_c -> test_f_c[d]",
                "test_f_e_c -> test_f_c[e]",
            ),
            id="cases-of-more-than-one-word-keep-their-hyphens",
        ),
        pytest.param(
            ("test_alpha", "test_beta", "test_gamma"),
            (),
            id="names-sharing-only-the-prefix-of-every-test",
        ),
        pytest.param(
            ("test_f", "test_f_one", "test_f_two"),
            (),
            id="a-name-that-is-all-the-merged-name",
        ),
        pytest.param(
            ("test_f_one", "test_g_two", "test_f_three"),
            (),
            id="a-name-that-shares-only-the-prefix-of-the-others",
        ),
    ],
)
def test_names_that_share_words_merge_under_them_and_a_group_with_no_name_is_left_alone(
    fix: Fix, names: tuple[str, ...], renamed: tuple[str, ...]
) -> None:
    source = named(names)

    fixed = fix(source, rule="PY-TEST0003")

    assert fixed.changes == ([f"parametrized {', '.join(renamed)}"] if renamed else [])
    assert (fixed.text == source) == (not renamed)


def test_two_groups_that_would_take_one_name_are_both_left_alone(fix: Fix) -> None:
    other = ["    assert f(4) == []", "    assert f(5) == []", "    assert f(6) == []"]
    source = suite([_ASSERT.format(a=a) for a in (1, 2, 3)] + other)

    fixed = fix(source, rule="PY-TEST0003")

    assert fixed.changes == []
    assert fixed.text == source


def test_tests_of_a_class_are_a_group_apart_from_the_tests_of_the_module(fix: Fix) -> None:
    source = trio(_ASSERT) + f"\n\nclass TestF:\n{_METHODS}"

    fixed = fix(source, rule="PY-TEST0003")

    assert fixed.text.count("@pytest.mark.parametrize") == 2
    assert fixed.text.count("def test_f(") == 2
    assert "class TestF:\n    @pytest.mark.parametrize" in fixed.text


def test_a_test_with_another_mark_stays_and_the_rest_of_its_shape_merges(fix: Fix) -> None:
    source = suite([_ASSERT.format(a=a) for a in (1, 2, 3, 4)]).replace("def test_f_two", _SLOW)

    fixed = fix(source, rule="PY-TEST0003")

    renamed = (
        "test_f_one -> test_f[one]",
        "test_f_three -> test_f[three]",
        "test_f_four -> test_f[four]",
    )
    assert fixed.changes == [f"parametrized {', '.join(renamed)}"]
    assert fixed.text.count(_SLOW) == 1
    assert set(outcomes(fixed.text)) == {
        "test_f[one]",
        "test_f[three]",
        "test_f[four]",
        "test_f_two",
    }


def test_a_shape_left_with_fewer_tests_than_the_rule_asks_for_is_left_alone(fix: Fix) -> None:
    source = suite([_ASSERT.format(a=a) for a in (1, 2, 3)]).replace("def test_f_two", _SLOW)

    fixed = fix(source, rule="PY-TEST0003")

    assert fixed.changes == []
    assert fixed.text == source


@pytest.mark.parametrize(
    "holder",
    [
        pytest.param("class Suite:\n{methods}", id="a-class-pytest-does-not-collect"),
        pytest.param(
            "class TestF:\n    def __init__(self):\n        pass\n\n{methods}",
            id="a-class-with-an-initializer",
        ),
    ],
)
def test_tests_pytest_does_not_collect_are_left_alone(fix: Fix, holder: str) -> None:
    source = preamble() + holder.format(methods=_METHODS)

    fixed = fix(source, rule="PY-TEST0003")

    assert fixed.changes == []
    assert fixed.text == source


def test_a_cluster_of_few_intents_merges_each_intent_that_two_or_more_tests_hold(fix: Fix) -> None:
    source = "\n\n".join(["from lib import b\n", _POSITIVE, _BOUNDED, _SINGLE])

    fixed = fix(source, rule="PY-TEST0016", context=_LIB)

    positive = parametrized(fixed.text, name="test_positive")
    assert positive.endswith("def test_positive(case_0):\n    assert b(case_0) > 0")
    assert "pytest.param(3, id='three')" in positive
    assert parametrized(fixed.text, name="test_bounded").endswith(
        "def test_bounded(case_0):\n    assert b(case_0) > 0\n    assert b(case_0) < 100"
    )
    assert "def test_single()" in fixed.text
    assert len(fixed.changes) == 1
