"""Tests that differ only in literals merge into one parametrized test with a case for each, the
cases being the tests that were, and a merge the tests cannot carry leaves the file as it was."""

from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.analysis import Rule
from mainboard.analysis.engine.lint import verbs
from mainboard.analysis.operations.python.suites import MergeTests
from mainboard.analysis.rules.python.testing.cases import ParametrizationCandidateGroupCount
from mainboard.analysis.rules.python.testing.redundancy import ConcentratedTestReachClusterCount

from .builders import merged_test, preamble, suite, trio, vectored
from .readers import Merged, outcomes, parametrized

type Merge = Callable[..., Merged]
type Fix = Callable[..., Merged]

_LITERAL = st.one_of(
    st.integers(0, 9).map(repr),
    st.text(alphabet=st.characters(codec="utf-8"), max_size=4).map(repr),
    st.floats(allow_nan=False, allow_infinity=False).map(lambda value: repr(abs(value))),
    st.sampled_from(["True", "False", "None"]),
)
_ASSERT = "    assert f({a}) == [{a}, None]"
_KEYWORD = '    assert f({a}, key="k{a}") == [{a}, "k{a}"]'
_CASES = ("1", "2", "3")
_KEYED = ("1, 'k1'", "2, 'k2'", "3, 'k3'")
_COMMENTED = """# the first, kept
@pytest.mark.slow
def test_f_one():
    assert f(1) == []


# about two
# and more
@pytest.mark.slow
def test_f_two():
    assert f(2) == []


@pytest.mark.slow
def test_f_three():
    assert f(3) == []


def test_after():
    assert f(0) == []
"""
_CLASS = """class TestF:
    def test_other(self):
        assert f(0) == [0, None]

    # the first
    def test_f_one(self, tmp_path):
        assert f(1)[0] == 2

    # the second
    def test_f_two(self, tmp_path):
        assert f(2)[0] == 3

    def test_f_three(self, tmp_path):
        assert f(3)[0] == 4

    def test_last(self):
        assert f(5) == [5, None]
"""


@given(
    vectors=st.lists(st.tuples(_LITERAL, _LITERAL, _LITERAL), min_size=3, max_size=6, unique=True)
)
@example(vectors=[("1", "'é'", "1"), ("2", '"""a\nb"""', "3"), ("3", "'it\\'s'", "3")])
@example(vectors=[("0", "None", "0"), ("1", "True", "1"), ("1.5", "'\\x00'", "2")])
@settings(max_examples=12)
def test_tests_differing_only_in_literals_merge_into_one_with_every_case(
    fix: Fix, vectors: list[tuple[str, str, str]]
) -> None:
    source = vectored(vectors)

    fixed = fix(source, rule="PY-TEST0003")

    before = {
        name.replace("test_f_", "test_f[") + "]": passes
        for name, passes in outcomes(source).items()
    }
    assert outcomes(fixed.text) == before
    assert len(fixed.changes) == 1


@pytest.mark.parametrize(
    ("body", "params", "expected"),
    [
        pytest.param(
            _KEYWORD,
            "",
            merged_test(
                ("case_0", "key"),
                cases=_KEYED,
                parameters="case_0, key",
                body=["assert f(case_0, key=key) == [case_0, key]"],
            ),
            id="a-keyword-argument-takes-its-keyword-as-the-name",
        ),
        pytest.param(
            "    value = {a}\n    assert f(value) == [{b}, None]",
            "",
            merged_test(
                ("value", "case_1"),
                cases=("1, 2", "2, 3", "3, 4"),
                parameters="value, case_1",
                body=["value = value", "assert f(value) == [case_1, None]"],
            ),
            id="an-assigned-literal-takes-the-variable-as-the-name",
        ),
        pytest.param(
            "    assert f({a})[0] == {b}",
            "",
            merged_test(
                ("case_0", "expected"),
                cases=("1, 2", "2, 3", "3, 4"),
                parameters="case_0, expected",
                body=["assert f(case_0)[0] == expected"],
            ),
            id="the-right-side-of-an-asserted-equality-is-the-expected-value",
        ),
        pytest.param(
            "    assert f({a}, {b}) == [{a}, {b}]",
            "",
            merged_test(
                ("case_0", "case_1"),
                cases=("1, 2", "2, 3", "3, 4"),
                parameters="case_0, case_1",
                body=["assert f(case_0, case_1) == [case_0, case_1]"],
            ),
            id="positions-that-differ-alike-are-one-parameter",
        ),
        pytest.param(
            "    key = 5\n" + _KEYWORD,
            "",
            merged_test(
                ("case_0", "key_2"),
                cases=_KEYED,
                parameters="case_0, key_2",
                body=["key = 5", "assert f(case_0, key=key_2) == [case_0, key_2]"],
            ),
            id="a-name-the-body-already-uses-gets-a-number",
        ),
        pytest.param(
            _KEYWORD,
            "key",
            merged_test(
                ("case_0", "key_2"),
                cases=_KEYED,
                parameters="key, case_0, key_2",
                body=["assert f(case_0, key=key_2) == [case_0, key_2]"],
            ),
            id="a-name-a-parameter-already-has-gets-a-number",
        ),
        pytest.param(
            _KEYWORD.replace("key=", "request="),
            "",
            merged_test(
                ("case_0", "request_2"),
                cases=_KEYED,
                parameters="case_0, request_2",
                body=["assert f(case_0, request=request_2) == [case_0, request_2]"],
            ),
            id="the-name-of-a-pytest-fixture-gets-a-number",
        ),
        pytest.param(
            "    assert f(-{a}) == [-{a}, None]",
            "",
            merged_test(
                ("case_0",),
                cases=_CASES,
                parameters="case_0",
                body=["assert f(-case_0) == [-case_0, None]"],
            ),
            id="a-sign-stays-in-the-body-and-the-number-is-the-case",
        ),
        pytest.param(
            _ASSERT,
            "a=1",
            merged_test(
                ("case_0",),
                cases=_CASES,
                parameters="case_0, a=1",
                body=["assert f(case_0) == [case_0, None]"],
            ),
            id="parameters-go-before-the-first-one-with-a-default",
        ),
        pytest.param(
            _ASSERT,
            "self, tmp_path, *args, **kwargs",
            merged_test(
                ("case_0",),
                cases=_CASES,
                parameters="self, tmp_path, case_0, *args, **kwargs",
                body=["assert f(case_0) == [case_0, None]"],
            ),
            id="parameters-go-before-a-splat",
        ),
        pytest.param(
            '    """one doc"""\n' + _ASSERT,
            "",
            merged_test(
                ("case_0",),
                cases=_CASES,
                parameters="case_0",
                body=['"""one doc"""', "assert f(case_0) == [case_0, None]"],
            ),
            id="a-docstring-the-tests-share-stays",
        ),
        pytest.param(
            '    assert f("a{a}" "b") == ["a{a}b", None]',
            "",
            merged_test(
                ("case_0", "case_1"),
                cases=("'a1b', 'a1b'", "'a2b', 'a2b'", "'a3b', 'a3b'"),
                parameters="case_0, case_1",
                body=["assert f(case_0) == [case_1, None]"],
            ),
            id="an-implicit-concatenation-is-one-literal",
        ),
    ],
)
def test_each_position_that_differs_is_a_parameter_named_after_what_it_stands_in(
    merge: Merge, body: str, params: str, expected: str
) -> None:
    merged = merge(trio(body, params=params))

    assert merged.refusal == ""
    assert merged.edits == ["replace", "delete", "delete", "import"]
    assert parametrized(merged.text, name="test_f") == expected
    assert merged.text.startswith(preamble())


def test_a_merge_returns_tests_of_every_literal_kind_as_cases(merge: Merge) -> None:
    source = suite(
        [
            "    assert f(1, key=None) == [1, None]",
            "    assert f(2.5, key=True) == [2.5, True]",
            '    assert f("s", key="u") == ["s", "u"]',
        ]
    )

    merged = merge(source)

    cases = ("1, None", "2.5, True", "'s', 'u'")
    assert outcomes(merged.text) == {f"test_f[{word}]": True for word in ("one", "two", "three")}
    expected = merged_test(
        ("case_0", "key"),
        cases=cases,
        parameters="case_0, key",
        body=["assert f(case_0, key=key) == [case_0, key]"],
    )
    assert parametrized(merged.text, name="test_f") == expected


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        pytest.param(
            trio(_ASSERT).replace("def test_f_two", "@pytest.mark.slow\ndef test_f_two"),
            "the tests differ in decorators or signature",
            id="a-decorator-only-one-test-carries",
        ),
        pytest.param(
            trio(_ASSERT).replace("test_f_two()", "test_f_two(tmp_path)"),
            "the tests differ in decorators or signature",
            id="a-signature-that-differs",
        ),
        pytest.param(
            trio(_ASSERT, deco='@pytest.mark.parametrize("y", [1, 2])\n'),
            "the tests are parametrized or property based already",
            id="tests-parametrized-already",
        ),
        pytest.param(
            trio(_ASSERT, deco="@given(st.integers())\n"),
            "the tests are parametrized or property based already",
            id="tests-that-are-properties",
        ),
        pytest.param(
            trio(_ASSERT, params="a, /"),
            "the tests take positional-only parameters",
            id="positional-only-parameters",
        ),
        pytest.param(
            suite(["    assert f(1) == []", "    assert f(2) == []", "    assert f(x) == []"]),
            "the tests differ in more than literals",
            id="a-name-where-the-others-have-a-literal",
        ),
        pytest.param(
            suite(["    assert f(1) == []", "    assert f(2) != []", "    assert f(3) == []"]),
            "the tests differ in more than literals",
            id="an-operator-that-differs",
        ),
        pytest.param(
            suite(['    assert f(f"a{1}")', '    assert f(f"b{1}")', '    assert f(f"c{1}")']),
            "the tests differ in more than literals",
            id="the-text-of-an-f-string",
        ),
        pytest.param(
            trio('    """doc {w}"""\n' + _ASSERT),
            "a literal differs in a bare string such as a docstring",
            id="a-docstring-that-differs",
        ),
        pytest.param(
            trio('    assert f("{a}" f"{{1}}") == []'),
            "a literal differs in a piece of an implicit string concatenation",
            id="a-piece-of-a-concatenation-with-an-f-string",
        ),
        pytest.param(
            trio("    match f(1):\n        case {a}:\n            pass"),
            "a literal differs in a pattern or an annotation",
            id="a-pattern-of-a-match",
        ),
        pytest.param(
            trio('    y: "T{a}" = f(1)'),
            "a literal differs in a pattern or an annotation",
            id="an-annotation",
        ),
        pytest.param(
            trio(_ASSERT + "  # {w}"),
            "a comment stands in a test the merge deletes or in a signature",
            id="a-comment-in-a-test-the-merge-deletes",
        ),
        pytest.param(
            trio("    global counter\n" + _ASSERT),
            "a test names a variable of an outer scope",
            id="a-test-declaring-a-name-of-the-module",
        ),
        pytest.param(
            trio(_ASSERT) + "\n\ndef test_f():\n    pass\n",
            "the merged name is used outside the tests",
            id="a-merged-name-the-module-defines",
        ),
    ],
)
def test_a_merge_the_tests_cannot_carry_is_refused_and_changes_no_byte(
    merge: Merge, source: str, reason: str
) -> None:
    merged = merge(source)

    assert merged.refusal == reason
    assert merged.edits == []
    assert merged.text == source


@pytest.mark.parametrize(
    ("names", "ids", "reason"),
    [
        pytest.param(
            ("test_f_one", "test_f_two", "test_f_three"),
            ("one", "two"),
            "the row names tests and ids that differ",
            id="more-tests-than-ids",
        ),
        pytest.param(
            ("test_f_one",),
            ("one",),
            "the row does not name two or more Python functions",
            id="a-single-test",
        ),
    ],
)
def test_a_row_that_does_not_describe_a_merge_of_tests_is_refused(
    merge: Merge, names: tuple[str, ...], ids: tuple[str, ...], reason: str
) -> None:
    source = trio(_ASSERT)

    merged = merge(source, names=names, ids=ids)

    assert merged.refusal == reason
    assert merged.text == source


@pytest.mark.parametrize(
    "altered",
    [
        pytest.param({"tests": [9990, 9991, 9992]}, id="nodes-the-holder-does-not-hold"),
        pytest.param({"tests": [0, 1, 2]}, id="nodes-that-are-no-functions"),
        pytest.param({"ids": [1, 2, 3]}, id="ids-that-are-no-text"),
        pytest.param({"name": None}, id="no-merged-name"),
    ],
)
def test_a_row_with_nodes_or_values_of_the_wrong_kind_is_refused(
    stated: Callable[..., AbstractContextManager[tuple[Path, dict]]], altered: dict
) -> None:
    with stated(trio(_ASSERT)) as (_, row):
        operation = MergeTests()

        edits = operation.edits({**row, **altered})

    assert edits is None
    assert operation.why == "the row does not name tests of the node"


def test_a_test_that_the_deleter_refuses_withholds_the_whole_merge(
    stated: Callable[..., AbstractContextManager[tuple[Path, dict]]],
) -> None:
    with stated(trio(_ASSERT)) as (_, row):
        site = row["site"]
        error = site.target.model_copy(update={"type": "ERROR"})
        broken = site.model_copy(update={"ancestors": (error, *site.ancestors)})
        operation = MergeTests()

        edits = operation.edits({**row, "site": broken})

    assert edits is None
    assert operation.why == "the node sits in a syntax error"


def test_a_deleted_test_takes_its_decorators_and_the_comment_lines_above_it(merge: Merge) -> None:
    merged = merge(preamble() + _COMMENTED)

    assert merged.text.startswith(
        f"{preamble()}# the first, kept\n@pytest.mark.slow\n@pytest.mark"
    )
    assert merged.text.endswith("\n\n\ndef test_after():\n    assert f(0) == []\n")
    assert "about two" not in merged.text
    assert merged.text.count("@pytest.mark.slow") == 1


def test_tests_of_a_class_merge_in_place_and_keep_self_the_fixtures_and_their_neighbors(
    merge: Merge,
) -> None:
    merged = merge(preamble() + _CLASS)

    expected = merged_test(
        ("case_0", "expected"),
        cases=("1, 2", "2, 3", "3, 4"),
        parameters="self, tmp_path, case_0, expected",
        body=["assert f(case_0)[0] == expected"],
    )
    assert parametrized(merged.text, name="test_f") == expected
    assert "# the first" in merged.text
    assert "# the second" not in merged.text
    assert merged.text.index("def test_other") < merged.text.index("def test_f(")
    assert merged.text.index("def test_f(") < merged.text.index("def test_last")
    assert merged.edits == ["replace", "delete", "claim", "delete", "claim", "import"]


def test_an_async_test_with_a_shared_mark_merges_under_the_same_mark(merge: Merge) -> None:
    source = trio(_ASSERT, deco="@pytest.mark.asyncio\n").replace("def test", "async def test")

    merged = merge(source)

    assert "@pytest.mark.asyncio\n@pytest.mark.parametrize" in parametrized(
        merged.text, name="test_f"
    )
    assert "async def test_f(case_0)" in merged.text


def test_a_module_with_windows_line_endings_and_multibyte_text_keeps_them(merge: Merge) -> None:
    bodies = ['    assert f("é") == ["é", None]', '    assert f("日本") == ["日本", None]']
    source = suite([*bodies, '    assert f("c") == ["c", None]']).replace("\n", "\r\n")

    merged = merge(source)

    assert merged.text.replace("\r\n", "").count("\n") == 0
    assert outcomes(merged.text) == {f"test_f[{word}]": True for word in ("one", "two", "three")}
    assert '"日本"' in merged.text


def test_a_module_without_pytest_imported_gets_the_import(merge: Merge) -> None:
    source = trio(_ASSERT).removeprefix("import pytest\n\n\n")

    merged = merge(source)

    assert merged.text.startswith("import pytest\n")
    assert merged.edits[-1] == "import"


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(ParametrizationCandidateGroupCount, id="the-group-of-distinct-literal-cases"),
        pytest.param(ConcentratedTestReachClusterCount, id="the-cluster-of-few-intents"),
    ],
)
def test_every_example_of_the_merging_rules_passes(
    capsys: pytest.CaptureFixture[str], rule: type[Rule]
) -> None:
    with suppress(SystemExit):
        verbs.test(selection=verbs.Selection(rule=(rule.id,)))

    passes = [f"{rule.id} example {number}: pass" for number in range(1, len(rule.examples) + 1)]
    assert capsys.readouterr().out.splitlines() == [*passes, "0 examples failed"]
