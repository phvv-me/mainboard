"""Apple evidence stays truthful without requiring Xcode in the isolated test worker."""

import json
import os
from pathlib import Path

import pytest
from pydantic import JsonValue

from mainboard.apple import Apple
from mainboard.cli import build
from mainboard.core.errors import MissionError
from mainboard.core.tool import ToolOutput


class ResultTool:
    def __init__(self, summary: dict[str, JsonValue], nodes: list[dict[str, JsonValue]]) -> None:
        self.summary = summary
        self.nodes = nodes
        self.calls: list[tuple[str, ...]] = []
        self.failure: ToolOutput | None = None

    def run(self, tool: str, args: tuple[str, ...], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == "/usr/bin/xcrun" and cwd is None
        self.calls.append(args)
        if self.failure is not None:
            return self.failure
        assert args[:3] == ("xcresulttool", "get", "test-results")
        payload = self.summary if args[3] == "summary" else {"testNodes": self.nodes}
        return ToolOutput(stdout=json.dumps(payload))


@pytest.fixture
def result_tool() -> ResultTool:
    return ResultTool(
        {
            "result": "Passed",
            "totalTestCount": 2,
            "passedTests": 2,
            "failedTests": 0,
            "skippedTests": 0,
            "expectedFailures": 0,
            "testFailures": [],
            "runtimeWarnings": [{"issueType": "Warning", "message": "example"}],
            "devicesAndConfigurations": [{"device": {"deviceName": "iPhone"}}],
        },
        [
            {
                "name": "Suite",
                "nodeType": "Test Suite",
                "result": "Passed",
                "durationInSeconds": 1000,
                "children": [
                    {
                        "name": "landscape",
                        "nodeType": "Test Case",
                        "result": "Passed",
                        "nodeIdentifierURL": "test://suite/landscape",
                        "durationInSeconds": 801,
                    },
                    {
                        "name": "parameterized",
                        "nodeType": "Test Case",
                        "result": "Passed",
                        "durationInSeconds": 2,
                        "children": [
                            {
                                "name": "argument",
                                "nodeType": "Arguments",
                                "result": "Passed",
                                "durationInSeconds": 4,
                            }
                        ],
                    },
                ],
            }
        ],
    )


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    path = tmp_path / "test ' $(literal).xcresult"
    path.mkdir()
    return path


def test_results_preserve_evidence_and_rank_cases_not_suites_or_arguments(
    result_tool: ResultTool,
    bundle: Path,
) -> None:
    report = Apple(result_tool).results(bundle)
    assert report.summary.successful
    assert [(case.name, case.seconds) for case in report.slowest] == [
        ("landscape", 801),
        ("parameterized", 2),
    ]
    document = report.document()
    assert document["bundle"] == str(bundle.resolve())
    summary = document["summary"]
    assert isinstance(summary, dict)
    assert summary["runtimeWarnings"] == result_tool.summary["runtimeWarnings"]
    assert summary["devicesAndConfigurations"] == result_tool.summary["devicesAndConfigurations"]
    assert all(
        call[4:] == ("--path", str(bundle.resolve()), "--compact") for call in result_tool.calls
    )


@pytest.mark.parametrize(
    "change",
    [
        {"result": "Failed", "failedTests": 1},
        {"result": "unknown"},
        {"result": "Skipped", "passedTests": 0, "skippedTests": 2},
        {"totalTestCount": 0, "passedTests": 0},
        {"testFailures": [{"failureText": "crashed"}]},
    ],
)
def test_unsuccessful_runs_never_pass(
    result_tool: ResultTool,
    bundle: Path,
    change: dict[str, JsonValue],
) -> None:
    result_tool.summary.update(change)
    assert not Apple(result_tool).results(bundle).summary.successful


def test_skips_and_expected_failures_remain_visible(result_tool: ResultTool, bundle: Path) -> None:
    result_tool.summary.update(totalTestCount=4, skippedTests=1, expectedFailures=1)
    summary = Apple(result_tool).results(bundle).summary
    assert summary.successful and summary.skipped == 1 and summary.expected_failures == 1


@pytest.mark.parametrize("payload", ["{}", "not JSON", '{"result":"Passed"}'])
def test_malformed_or_incomplete_evidence_is_an_error(
    result_tool: ResultTool,
    bundle: Path,
    payload: str,
) -> None:
    result_tool.failure = ToolOutput(stdout=payload)
    with pytest.raises(MissionError, match="invalid xcresulttool"):
        Apple(result_tool).results(bundle)


@pytest.mark.parametrize(
    "output",
    [
        ToolOutput(returncode=127, stderr="Xcode missing"),
        ToolOutput(returncode=124, truncated=True),
        ToolOutput(stdout='{"result":"Passed"}', truncated=True),
        ToolOutput(returncode=1, stderr="bundle incomplete"),
    ],
)
def test_tool_failures_and_truncation_are_errors(
    result_tool: ResultTool,
    bundle: Path,
    output: ToolOutput,
) -> None:
    result_tool.failure = output
    with pytest.raises(MissionError, match="xcrun .* failed"):
        Apple(result_tool).results(bundle)


def test_directory_selects_newest_bundle_without_success_fallback(
    result_tool: ResultTool,
    bundle: Path,
) -> None:
    newest = bundle.parent / "newest.xcresult"
    newest.mkdir()
    os.utime(bundle, (10, 10))
    os.utime(newest, (20, 20))
    result_tool.summary["result"] = "Failed"
    report = Apple(result_tool).results(bundle.parent)
    assert report.bundle == newest and not report.summary.successful
    assert all(str(newest) in call for call in result_tool.calls)


def test_missing_bundles_fail_before_running_tools(
    result_tool: ResultTool, tmp_path: Path
) -> None:
    for path in (tmp_path, tmp_path / "missing.xcresult"):
        with pytest.raises(MissionError, match="no .*bundle"):
            Apple(result_tool).results(path)
    assert result_tool.calls == []


@pytest.mark.parametrize("passed", [True, False])
def test_cli_returns_test_verdict_and_prints_json(
    result_tool: ResultTool,
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    passed: bool,
) -> None:
    result_tool.summary["result"] = "Passed" if passed else "Failed"
    apple = Apple(result_tool)
    monkeypatch.setattr("mainboard.cli.Apple", lambda: apple)
    with pytest.raises(SystemExit) as stopped:
        build()(["apple", "results", str(bundle), "--json"])
    assert stopped.value.code == (0 if passed else 1)
    assert json.loads(capsys.readouterr().out)["successful"] is passed


def test_toolchain_reads_selected_tools_without_mutating_selection(
    result_tool: ResultTool,
) -> None:
    result_tool.failure = ToolOutput(stdout="selected\n")
    assert Apple(result_tool).toolchain() == {
        "xcode": "selected",
        "swift": "selected",
        "swift_path": "selected",
        "xcresulttool_path": "selected",
    }
    assert result_tool.calls == [
        ("xcodebuild", "-version"),
        ("swift", "--version"),
        ("--find", "swift"),
        ("--find", "xcresulttool"),
    ]


@pytest.mark.parametrize("count", [-1, "2", True])
def test_invalid_counts_are_not_coerced_to_success(
    result_tool: ResultTool,
    bundle: Path,
    count: JsonValue,
) -> None:
    result_tool.summary["passedTests"] = count
    with pytest.raises(MissionError, match="invalid xcresulttool"):
        Apple(result_tool).results(bundle)


def test_simulators_keep_runtime_identity_and_exclude_unavailable(
    result_tool: ResultTool,
) -> None:
    device = {"name": "Same name", "udid": "device-a", "state": "Shutdown", "isAvailable": True}
    result_tool.failure = ToolOutput(
        stdout=json.dumps(
            {
                "devices": {
                    "runtime-new": [device, {**device, "udid": "gone", "isAvailable": False}],
                    "runtime-old": [{**device, "udid": "device-b", "state": "Booted"}],
                }
            }
        )
    )
    assert Apple(result_tool).simulators() == [
        {"runtime": "runtime-new", "name": "Same name", "udid": "device-a", "state": "Shutdown"},
        {"runtime": "runtime-old", "name": "Same name", "udid": "device-b", "state": "Booted"},
    ]
    assert result_tool.calls == [("simctl", "list", "devices", "available", "--json")]


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        "{}",
        '{"devices": []}',
        '{"devices": {"runtime": [{"name": "Device", "udid": "id", "state": "Booted"}]}}',
        '{"devices": {"runtime": [{"name": "Device", "udid": "id", '
        '"state": "Booted", "isAvailable": "true"}]}}',
    ],
)
def test_malformed_simulator_inventory_fails(result_tool: ResultTool, payload: str) -> None:
    result_tool.failure = ToolOutput(stdout=payload)
    with pytest.raises(MissionError, match="invalid simctl"):
        Apple(result_tool).simulators()


@pytest.mark.parametrize(
    "failure",
    [
        ToolOutput(returncode=72, stderr="simctl not available"),
        ToolOutput(stdout='{"devices": {}}', truncated=True),
    ],
)
def test_unavailable_or_truncated_simulator_tools_fail(
    result_tool: ResultTool,
    failure: ToolOutput,
) -> None:
    result_tool.failure = failure
    with pytest.raises(MissionError, match="xcrun simctl .* failed"):
        Apple(result_tool).simulators()


def test_empty_simulator_inventory_is_valid_and_cli_read_only(
    result_tool: ResultTool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_tool.failure = ToolOutput(stdout='{"devices": {}}')
    monkeypatch.setattr("mainboard.cli.Apple", lambda: Apple(result_tool))
    with pytest.raises(SystemExit) as stopped:
        build()(["apple", "simulators", "--json"])
    assert stopped.value.code == 0
    assert json.loads(capsys.readouterr().out) == []
    assert result_tool.calls == [("simctl", "list", "devices", "available", "--json")]
