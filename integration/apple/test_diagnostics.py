import json
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.apple import Apple
from mainboard.core import ToolOutput

from .fakes import ResultTool

_INVALID = "invalid xcresulttool build evidence"


@pytest.mark.parametrize(
    ("change", "successful"),
    [
        pytest.param({}, True, id="a-succeeded-build-with-a-located-warning"),
        pytest.param(
            {"warnings": [{"issueType": "Swift Compiler Warning", "message": "no location"}]},
            True,
            id="a-warning-without-a-location-omits-it",
        ),
        pytest.param({"status": "failed"}, False, id="a-failed-build"),
        pytest.param({"status": "cancelled"}, False, id="a-cancelled-build"),
        pytest.param({"status": "unknown"}, False, id="an-unknown-status"),
        pytest.param({"errorCount": 1}, False, id="a-counted-error"),
        pytest.param(
            {"errors": [{"issueType": "Swift Compiler Error", "message": "Missing symbol"}]},
            False,
            id="a-listed-error",
        ),
    ],
)
def test_build_diagnostics_judge_the_status_and_keep_locations_without_reading_tests(
    bundle: Path, change, successful: bool
) -> None:
    tool = ResultTool(build=change)
    report = Apple(tool).diagnostics(bundle)
    assert report.summary.successful is successful
    assert report.document() == {
        "bundle": str(bundle.resolve()),
        "build_successful": successful,
        "summary": tool.build_summary,
    }
    assert tool.calls == [
        ("xcresulttool", "get", "build-results", "--path", str(bundle.resolve()), "--compact")
    ]


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param({"reply": ToolOutput(stdout="not JSON")}, id="text-that-is-not-json"),
        pytest.param({"reply": ToolOutput(stdout="{}")}, id="an-empty-object"),
        pytest.param(
            {"reply": ToolOutput(stdout='{"status":"succeeded"}')}, id="a-status-without-counts"
        ),
        pytest.param({"build": {"errorCount": -1}}, id="a-negative-count"),
        pytest.param({"build": {"errorCount": True}}, id="a-boolean-count"),
        pytest.param({"build": {"warningCount": "1"}}, id="a-count-spelled-as-text"),
        pytest.param({"build": {"errors": [{}]}}, id="an-issue-without-type-or-message"),
        pytest.param(
            {"build": {"warnings": [{"message": "Missing issue type"}]}},
            id="a-warning-without-an-issue-type",
        ),
        pytest.param({"build": {"analyzerWarnings": None}}, id="analyzer-warnings-left-null"),
    ],
)
def test_incomplete_build_evidence_is_never_a_zero_error_success(bundle: Path, arrange) -> None:
    fake = ResultTool(**arrange)
    with pytest.raises(MissionError, match=_INVALID):
        Apple(fake).diagnostics(bundle)
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    ("status", "code"),
    [
        pytest.param("succeeded", 0, id="a-succeeded-build-exits-zero"),
        pytest.param("failed", 1, id="a-failed-build-exits-one"),
        pytest.param("unknown", 1, id="an-unknown-status-exits-one"),
    ],
)
def test_the_diagnostics_verb_reads_the_newest_bundle_and_exits_with_the_build_verdict(
    cli, bundles: Path, status: str, code: int
) -> None:
    exit_code, out = cli(
        Apple(ResultTool(build={"status": status})), "diagnostics", str(bundles), "--json"
    )
    document = json.loads(out)
    assert exit_code == code
    assert document["bundle"] == str(bundles / "new.xcresult")
    assert document["build_successful"] is (code == 0)
