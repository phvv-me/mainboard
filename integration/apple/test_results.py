import json
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.apple import Apple
from mainboard.core import ToolOutput

from .builders import Node, case, node, passing_nodes
from .constants import IDENTIFIER_HOST
from .fakes import ResultTool

_FAILURE: Node = {"failureText": "XCTAssertTrue failed"}
_WARNING = "Invalid frame dimension (negative or non-finite)."
_SKIP = "Test skipped: Requires two independent fixture test hosts"
_RANKED = [("landscape", "Passed", 801, []), ("parameterized", "Passed", 2, ["argument"])]
_INVALID = "invalid xcresulttool test evidence"
_FIXTURE_FIELDS = settings(
    database=None,
    deadline=None,
    max_examples=150,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.mark.parametrize(
    ("arrange", "successful", "slowest"),
    [
        pytest.param({}, True, _RANKED, id="a-passing-run-ranks-cases-not-suites-or-arguments"),
        pytest.param(
            {"summary": {"result": "Failed", "failedTests": 1}},
            False,
            _RANKED,
            id="a-failed-test",
        ),
        pytest.param({"summary": {"result": "unknown"}}, False, _RANKED, id="an-unknown-result"),
        pytest.param(
            {"summary": {"result": "Skipped", "passedTests": 0, "skippedTests": 2}},
            False,
            _RANKED,
            id="only-skips",
        ),
        pytest.param(
            {"summary": {"totalTestCount": 0, "passedTests": 0}},
            False,
            _RANKED,
            id="no-test-ran",
        ),
        pytest.param(
            {"summary": {"testFailures": [{"failureText": "crashed"}]}},
            False,
            _RANKED,
            id="a-recorded-failure-text",
        ),
        pytest.param(
            {"summary": {"totalTestCount": 4, "skippedTests": 1, "expectedFailures": 1}},
            True,
            _RANKED,
            id="skips-and-expected-failures-stay-visible-without-failing",
        ),
        pytest.param(
            {
                "summary": {
                    "result": "Failed",
                    "passedTests": 1,
                    "failedTests": 1,
                    "testFailures": [_FAILURE],
                },
                "nodes": [
                    case(
                        "practice",
                        result="Failed",
                        seconds=25.8,
                        children=[
                            node(
                                "XCTAssertTrue failed",
                                kind="Failure Message",
                                sourceLocation={
                                    "filePath": "/project/ReviewFlowTests.swift",
                                    "lineNumber": 62,
                                },
                            )
                        ],
                    )
                ],
            },
            False,
            [("practice", "Failed", 25.8, ["XCTAssertTrue failed"])],
            id="a-failure-message-leaf-carries-no-verdict",
        ),
        pytest.param(
            {
                "summary": {
                    "runtimeWarnings": [{"issueType": "Runtime Warning", "message": _WARNING}]
                },
                "nodes": [
                    case("editor", seconds=32.9, children=[node(_WARNING, kind="Runtime Warning")])
                ],
            },
            True,
            [("editor", "Passed", 32.9, [_WARNING])],
            id="a-runtime-warning-leaf-is-no-test",
        ),
        pytest.param(
            {
                "summary": {"totalTestCount": 3, "skippedTests": 1},
                "nodes": [
                    *passing_nodes(),
                    case("restart", result="Skipped", children=[node(_SKIP, kind="Skip Message")]),
                ],
            },
            True,
            _RANKED,
            id="a-skip-message-leaf-keeps-the-skip-verdict",
        ),
    ],
)
def test_results_judge_the_native_summary_and_keep_every_diagnostic(
    bundle, arrange, successful, slowest
) -> None:
    tool = ResultTool(**arrange)
    report = Apple(tool).results(bundle)
    summary, raw = report.summary, tool.summary
    assert summary.successful is successful
    assert [summary.skipped, summary.expected_failures, summary.failures] == [
        raw["skippedTests"],
        raw["expectedFailures"],
        raw["testFailures"],
    ]
    assert [
        (ranked.name, ranked.result, ranked.seconds, [child.name for child in ranked.children])
        for ranked in report.slowest
    ] == slowest
    assert report.document()["summary"] == raw
    assert report.document()["bundle"] == str(bundle.resolve())


@pytest.mark.parametrize(
    ("target", "chosen"),
    [
        pytest.param(".", "new.xcresult", id="a-folder-yields-its-newest-bundle"),
        pytest.param("old.xcresult", "old.xcresult", id="a-named-bundle-is-read-even-when-older"),
    ],
)
def test_results_read_the_named_bundle_or_the_newest_in_a_folder(
    bundles: Path, target: str, chosen: str
) -> None:
    tool = ResultTool(summary={"result": "Failed"})
    report = Apple(tool).results(bundles / target)
    path = (bundles / chosen).resolve()
    assert report.bundle == path and not report.summary.successful
    assert tool.calls == [
        ("xcresulttool", "get", "test-results", section, "--path", str(path), "--compact")
        for section in ("summary", "tests")
    ]


@pytest.mark.parametrize(
    ("target", "match"),
    [
        pytest.param(".", "no .xcresult bundles directly inside", id="a-folder-without-bundles"),
        pytest.param("missing.xcresult", "no result bundle or test-log", id="a-missing-bundle"),
    ],
)
def test_a_path_without_a_bundle_fails_before_any_tool_runs(
    tmp_path: Path, target: str, match: str
) -> None:
    tool = ResultTool()
    with pytest.raises(MissionError, match=match):
        Apple(tool).results(tmp_path / target)
    assert tool.calls == []


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param({"reply": ToolOutput(stdout="{}")}, id="an-empty-object"),
        pytest.param({"reply": ToolOutput(stdout="not JSON")}, id="text-that-is-not-json"),
        pytest.param(
            {"reply": ToolOutput(stdout='{"result":"Passed"}')}, id="a-summary-without-counts"
        ),
        pytest.param({"summary": {"passedTests": -1}}, id="a-negative-count"),
        pytest.param({"summary": {"passedTests": "2"}}, id="a-count-spelled-as-text"),
        pytest.param({"summary": {"passedTests": True}}, id="a-boolean-count"),
        *(
            pytest.param(
                {
                    "nodes": [
                        node(
                            "parent",
                            kind="Test Suite",
                            result="Passed",
                            children=[node("child", kind=kind)],
                        )
                    ]
                },
                id=f"a-{kind.lower().replace(' ', '-')}-child-without-a-verdict",
            )
            for kind in ["Test Case", "Test Suite", "Arguments", "Unknown Node"]
        ),
    ],
)
def test_incomplete_or_malformed_test_evidence_is_an_error(bundle, arrange) -> None:
    fake = ResultTool(**arrange)
    with pytest.raises(MissionError, match=_INVALID):
        Apple(fake).results(bundle)
    assert {" ".join(call[:2]) for call in fake.calls} == {"xcresulttool get"}


@st.composite
def _trees(draw: st.DrawFn, depth: int = 3) -> tuple[Node, list[float]]:
    """A test node and the durations of the test cases at or below it that report one."""
    kind = draw(st.sampled_from(["Test Case", "Test Suite", "Arguments"]))
    seconds = draw(st.none() | st.floats(min_value=0, max_value=1e6))
    leaves = st.builds(
        lambda name, found: (node(name, kind=found), []),
        st.text(max_size=5),
        st.sampled_from(["Failure Message", "Runtime Warning", "Skip Message"]),
    )
    below = draw(st.lists(_trees(depth - 1) | leaves, max_size=3)) if depth else []
    own = [seconds] if kind == "Test Case" and seconds is not None else []
    tree = node(
        draw(st.text(max_size=5)),
        kind=kind,
        result="Passed",
        seconds=seconds,
        children=[child for child, _ in below],
    )
    return tree, own + [found for _, nested in below for found in nested]


_TWELVE_CASES = (
    node(
        "Suite",
        kind="Test Suite",
        result="Passed",
        children=[case(f"case{index}", seconds=float(index)) for index in range(12)],
    ),
    [float(index) for index in range(12)],
)


@_FIXTURE_FIELDS
@given(forest=st.lists(_trees(), max_size=4))
@example(forest=[_TWELVE_CASES])
def test_the_slowest_cases_are_the_ten_longest_test_cases_at_any_depth(bundle, forest) -> None:
    report = Apple(ResultTool(nodes=[tree for tree, _ in forest])).results(bundle)
    longest = sorted((found for _, below in forest for found in below), reverse=True)[:10]
    assert [ranked.seconds for ranked in report.slowest] == longest
    assert {ranked.kind for ranked in report.slowest} <= {"Test Case"}


_PARTS = ["a", "b", "ab", "p(_:)"]
_SPELLINGS = st.sampled_from(["{}", "{}()", "/{}/"])


def _spelled(parts: Sequence[str], template: str) -> str:
    """A test selector as a person may write it: plain, called, or between slashes."""
    return template.format("/".join(parts))


def _unanswered(
    selector: Sequence[str], *, idents: Sequence[Sequence[str]], skips: Sequence[Sequence[str]]
) -> bool:
    """Whether a selector of path parts names one case that no case answers and no skip covers."""
    return (
        len(selector) >= 3
        and "*" not in selector
        and not any(list(ident[-len(selector) :]) == list(selector) for ident in idents)
        and not any(list(selector[: len(skip)]) == list(skip) for skip in skips)
    )


@_FIXTURE_FIELDS
@given(
    idents=st.lists(st.lists(st.sampled_from(_PARTS), min_size=3, max_size=4), max_size=4),
    selectors=st.lists(
        st.tuples(st.lists(st.sampled_from([*_PARTS, "*"]), min_size=1, max_size=4), _SPELLINGS),
        max_size=5,
    ),
    skips=st.lists(
        st.tuples(st.lists(st.sampled_from(_PARTS), min_size=1, max_size=3), _SPELLINGS),
        max_size=3,
    ),
)
@example(idents=[["a", "b", "ab", "p(_:)"]], selectors=[(["b", "ab", "p(_:)"], "{}()")], skips=[])
@example(
    idents=[["a", "b", "ab", "ab"]], selectors=[(["b", "ab", "b"], "{}")], skips=[(["b"], "/{}/")]
)
@example(idents=[["ab", "ab", "b"]], selectors=[(["b", "ab", "b"], "{}")], skips=[])
@example(idents=[], selectors=[(["ab", "a", "b"], "{}")], skips=[(["a"], "{}")])
def test_a_selector_is_missing_when_no_case_answers_it_and_no_skip_covers_it(
    bundle, idents, selectors, skips
) -> None:
    nodes = [
        case(f"case{index}", nodeIdentifierURL=f"{IDENTIFIER_HOST}/{quote('/'.join(parts))}")
        for index, parts in enumerate(idents)
    ]
    only = [_spelled(parts, template) for parts, template in selectors]
    skipped = [_spelled(parts, template) for parts, template in skips]
    report = Apple(ResultTool(nodes=nodes)).results(
        bundle, only_testing=only, skip_testing=skipped
    )
    expected = [
        text
        for text, (parts, _) in zip(only, selectors, strict=True)
        if _unanswered(parts, idents=idents, skips=[parts for parts, _ in skips])
    ]
    assert report.missing_selectors == expected
    assert report.successful is (not expected)


@pytest.mark.parametrize(
    ("result", "code"),
    [
        pytest.param("Passed", 0, id="a-passing-run-exits-zero"),
        pytest.param("Failed", 1, id="a-failed-run-exits-one"),
    ],
)
def test_the_results_verb_prints_json_and_exits_with_the_test_verdict(
    cli, bundle, result, code
) -> None:
    exit_code, out = cli(
        Apple(ResultTool(summary={"result": result})), "results", str(bundle), "--json"
    )
    assert exit_code == code
    assert json.loads(out)["successful"] is (code == 0)
