"""Apple evidence stays truthful without requiring Xcode in the isolated test worker."""

import json
import os
import sys
from pathlib import Path
from typing import Literal

import pytest
from pydantic import JsonValue

from mainboard.apple import Apple
from mainboard.cli import build
from mainboard.core.errors import MissionError
from mainboard.core.tool import ToolOutput
from mainboard.proc import Processes


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


@pytest.fixture
def build_summary() -> dict[str, JsonValue]:
    return {
        "status": "succeeded",
        "errorCount": 0,
        "warningCount": 1,
        "analyzerWarningCount": 0,
        "errors": [],
        "warnings": [
            {
                "issueType": "Swift Compiler Warning",
                "message": "Actor-isolated property cannot be referenced here",
                "targetName": "StudyUI",
                "sourceURL": "file:///tmp/Study%20View.swift#StartingLineNumber=42",
                "className": "DVTTextDocumentLocation",
            }
        ],
        "analyzerWarnings": [],
        "destination": {"platform": "macOS", "architecture": "arm64"},
        "actionTitle": "Building project Example with scheme Example",
    }


def test_build_diagnostics_preserve_locations_and_metadata_without_reading_tests(
    result_tool: ResultTool,
    build_summary: dict[str, JsonValue],
    bundle: Path,
) -> None:
    result_tool.failure = ToolOutput(stdout=json.dumps(build_summary))
    report = Apple(result_tool).diagnostics(bundle)
    assert report.summary.successful
    assert report.document() == {
        "bundle": str(bundle.resolve()),
        "build_successful": True,
        "summary": build_summary,
    }
    assert result_tool.calls == [
        ("xcresulttool", "get", "build-results", "--path", str(bundle.resolve()), "--compact")
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {"status": "cancelled"},
        {"status": "unknown"},
        {"errorCount": 1},
        {"errors": [{"issueType": "Swift Compiler Error", "message": "Missing symbol"}]},
    ],
)
def test_build_failures_cannot_be_reported_as_success(
    result_tool: ResultTool,
    build_summary: dict[str, JsonValue],
    bundle: Path,
    change: dict[str, JsonValue],
) -> None:
    build_summary.update(change)
    result_tool.failure = ToolOutput(stdout=json.dumps(build_summary))
    assert not Apple(result_tool).diagnostics(bundle).summary.successful


@pytest.mark.parametrize(
    "change",
    [
        {"errorCount": -1},
        {"errorCount": True},
        {"warningCount": "1"},
        {"errors": [{}]},
        {"warnings": [{"message": "Missing issue type"}]},
        {"analyzerWarnings": None},
    ],
)
def test_invalid_build_evidence_fails(
    result_tool: ResultTool,
    build_summary: dict[str, JsonValue],
    bundle: Path,
    change: dict[str, JsonValue],
) -> None:
    build_summary.update(change)
    result_tool.failure = ToolOutput(stdout=json.dumps(build_summary))
    with pytest.raises(MissionError, match="invalid xcresulttool build evidence"):
        Apple(result_tool).diagnostics(bundle)


@pytest.mark.parametrize("payload", ["not JSON", "{}", '{"status":"succeeded"}'])
def test_incomplete_build_evidence_is_not_a_zero_error_success(
    result_tool: ResultTool, bundle: Path, payload: str
) -> None:
    result_tool.failure = ToolOutput(stdout=payload)
    with pytest.raises(MissionError, match="invalid xcresulttool build evidence"):
        Apple(result_tool).diagnostics(bundle)


@pytest.mark.parametrize(
    "failure",
    [ToolOutput(returncode=1, stderr="unreadable"), ToolOutput(stdout="{}", truncated=True)],
)
def test_build_tool_failures_remain_errors(
    result_tool: ResultTool, bundle: Path, failure: ToolOutput
) -> None:
    result_tool.failure = failure
    with pytest.raises(MissionError, match="xcrun .* failed"):
        Apple(result_tool).diagnostics(bundle)


@pytest.mark.parametrize("status", ["succeeded", "failed", "unknown"])
def test_build_cli_verdict_and_newest_bundle_selection(
    result_tool: ResultTool,
    build_summary: dict[str, JsonValue],
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    newest = bundle.parent / "newest.xcresult"
    newest.mkdir()
    os.utime(bundle, (10, 10))
    os.utime(newest, (20, 20))
    build_summary["status"] = status
    result_tool.failure = ToolOutput(stdout=json.dumps(build_summary))
    monkeypatch.setattr("mainboard.cli.Apple", lambda: Apple(result_tool))
    with pytest.raises(SystemExit) as stopped:
        build()(["apple", "diagnostics", str(bundle.parent), "--json"])
    assert stopped.value.code == (0 if status == "succeeded" else 1)
    document = json.loads(capsys.readouterr().out)
    assert document["bundle"] == str(newest)
    assert document["build_successful"] == (status == "succeeded")


def test_failed_case_accepts_xcode_diagnostic_leaves_without_verdicts(
    result_tool: ResultTool,
    bundle: Path,
) -> None:
    failure: dict[str, JsonValue] = {"failureText": "XCTAssertTrue failed"}
    result_tool.summary.update(
        result="Failed", passedTests=1, failedTests=1, testFailures=[failure]
    )
    result_tool.nodes = [
        {
            "name": "practice",
            "nodeType": "Test Case",
            "result": "Failed",
            "durationInSeconds": 25.8,
            "children": [
                {
                    "name": "XCTAssertTrue failed",
                    "nodeType": "Failure Message",
                    "sourceLocation": {
                        "filePath": "/project/ReviewFlowTests.swift",
                        "lineNumber": 62,
                    },
                }
            ],
        }
    ]
    report = Apple(result_tool).results(bundle)
    assert not report.summary.successful
    assert report.summary.failures == [failure]
    assert [(case.name, case.result) for case in report.slowest] == [("practice", "Failed")]


def test_passing_case_retains_runtime_warning_without_treating_it_as_a_test(
    result_tool: ResultTool,
    bundle: Path,
) -> None:
    message = "Invalid frame dimension (negative or non-finite)."
    result_tool.summary["runtimeWarnings"] = [{"issueType": "Runtime Warning", "message": message}]
    result_tool.nodes = [
        {
            "name": "editor",
            "nodeType": "Test Case",
            "result": "Passed",
            "durationInSeconds": 32.9,
            "children": [{"name": message, "nodeType": "Runtime Warning"}],
        }
    ]
    report = Apple(result_tool).results(bundle)
    assert report.summary.successful
    assert [(case.name, case.seconds) for case in report.slowest] == [("editor", 32.9)]
    assert report.slowest[0].children[0].name == message
    summary = report.document()["summary"]
    assert isinstance(summary, dict)
    assert summary["runtimeWarnings"] == result_tool.summary["runtimeWarnings"]


@pytest.mark.parametrize("kind", ["Test Case", "Test Suite", "Arguments", "Unknown Node"])
def test_non_diagnostic_nodes_still_require_verdicts(
    result_tool: ResultTool,
    bundle: Path,
    kind: str,
) -> None:
    result_tool.nodes = [
        {
            "name": "parent",
            "nodeType": "Test Suite",
            "result": "Passed",
            "children": [{"name": "missing verdict", "nodeType": kind}],
        }
    ]
    with pytest.raises(MissionError, match="invalid xcresulttool"):
        Apple(result_tool).results(bundle)


@pytest.mark.parametrize("kind", ["xcworkspace", "xcodeproj", "package"])
@pytest.mark.parametrize("action", ["build", "test", "build-for-testing", "resolve"])
def test_execution_uses_literal_native_arguments_and_package_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    action: Literal["build", "test", "build-for-testing", "resolve"],
) -> None:
    project = tmp_path / ("Package" if kind == "package" else f"App '$literal.{kind}")
    project.mkdir()
    if kind == "package":
        (project / "Package.swift").write_text("// swift-tools-version: 6.4\n")
    bundle = tmp_path / "results" / "fresh.xcresult"
    calls: list[tuple[float, list[str], Path | None]] = []

    def execute(_self: Apple, seconds: float, command: list[str], *, cwd: Path | None) -> int:
        calls.append((seconds, command, cwd))
        return 65

    monkeypatch.setattr("mainboard.apple.Processes.timeout", execute)
    code = Apple().execute(
        project,
        scheme="An app",
        action=action,
        destination="platform=macOS",
        derived_data=tmp_path / "build",
        bundle=bundle,
        only_testing=("UITests/Flow/testA",),
        skip_testing=("UITests/Flow/testB",),
        settings=("DISPLAY_NAME=日本 '$()",),
        timeout=123,
    )
    assert code == 65 and len(calls) == 1
    seconds, command, cwd = calls[0]
    assert seconds == 123 and cwd == (project if kind == "package" else project.parent)
    assert command[:3] == [
        "/usr/bin/xcrun",
        "xcodebuild",
        "-resolvePackageDependencies" if action == "resolve" else action,
    ]
    assert "-onlyUsePackageVersionsFromResolvedFile" in command
    assert command.count("-hideShellScriptEnvironment") == 1
    if action != "resolve":
        assert command[command.index("-resultBundlePath") + 1] == str(bundle)
    else:
        assert "-resultBundlePath" not in command
    if action == "test":
        assert command[command.index("-parallel-testing-enabled") + 1] == "NO"
    else:
        assert "-parallel-testing-enabled" not in command
    assert command[-3:] == [
        "-only-testing:UITests/Flow/testA",
        "-skip-testing:UITests/Flow/testB",
        "DISPLAY_NAME=日本 '$()",
    ]
    if kind != "package":
        assert command[3:5] == [
            "-workspace" if kind == "xcworkspace" else "-project",
            str(project),
        ]
    assert bundle.parent.is_dir() == (action != "resolve")
    assert not bundle.exists()


def test_execution_refuses_old_evidence_and_invalid_inputs(tmp_path: Path) -> None:
    project = tmp_path / "App.xcodeproj"
    project.mkdir()
    old = tmp_path / "old.xcresult"
    old.mkdir()
    with pytest.raises(MissionError, match="overwrite"):
        Apple().execute(project, scheme="App", bundle=old)
    with pytest.raises(MissionError, match="positive timeout"):
        Apple().execute(project, scheme="App", timeout=0)
    with pytest.raises(MissionError, match="fresh result"):
        Apple().execute(project, scheme="App")
    with pytest.raises(MissionError, match="no Apple project"):
        Apple().execute(tmp_path / "missing", scheme="App")
    with pytest.raises(MissionError, match="expected an"):
        Apple().execute(tmp_path, scheme="App")


@pytest.mark.parametrize("native_code, passed", [(0, True), (0, False), (65, True)])
def test_execution_cli_uses_actual_test_verdict(
    result_tool: ResultTool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_code: int,
    passed: bool,
) -> None:
    project = tmp_path / "App.xcodeproj"
    project.mkdir()
    result_tool.summary["result"] = "Passed" if passed else "Failed"
    apple = Apple(result_tool)

    def execute(*_args: Path, **kwargs: str | Path | None | tuple[str, ...] | float) -> int:
        result = kwargs["bundle"]
        assert isinstance(result, Path)
        result.mkdir(parents=True)
        return native_code

    monkeypatch.setattr(apple, "execute", execute)
    monkeypatch.setattr("mainboard.cli.Apple", lambda: apple)
    with pytest.raises(SystemExit) as stopped:
        build()(
            [
                "apple",
                "execute",
                str(project),
                "--scheme",
                "App",
                "--action",
                "test",
                "--results",
                str(tmp_path / "out"),
            ]
        )
    assert stopped.value.code == (native_code or (0 if passed else 1))


def test_process_timeout_runs_package_commands_in_the_requested_directory(tmp_path: Path) -> None:
    code = Processes().timeout(
        10,
        [sys.executable, "-c", "from pathlib import Path; Path('receipt').write_text('ok')"],
        cwd=tmp_path,
    )
    assert code == 0 and (tmp_path / "receipt").read_text() == "ok"


@pytest.mark.parametrize(
    "marker,routes",
    [
        ("../escape", ["fixture:///reviews"]),
        ("/absolute", ["fixture:///reviews"]),
        ("tmp/request.json", ["fixture:///reviews?credential=x"]),
        ("tmp/request.json", ["fixture://host/reviews"]),
        ("tmp/request.json", ["fixture:///reviews", "fixture:///reviews"]),
    ],
)
def test_url_host_rejects_escaped_markers_and_unexpected_routes(
    marker: str,
    routes: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "App.xcodeproj"
    project.mkdir()
    configuration = tmp_path / "url.json"
    configuration.write_text(
        json.dumps({"runner_bundle": "example.runner", "marker": marker, "routes": routes})
    )

    def forbidden(*_args: str) -> list[dict[str, str]]:
        pytest.fail("invalid configuration must not inspect or launch a simulator")

    apple = Apple()
    monkeypatch.setattr(apple, "simulators", forbidden)
    with pytest.raises(MissionError):
        apple.execute(
            project,
            scheme="App",
            action="test",
            bundle=tmp_path / "test.xcresult",
            url_test=configuration,
        )


def test_url_host_uses_exact_destination_and_temporary_literal_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "App.xcodeproj"
    project.mkdir()
    configuration = tmp_path / "url.json"
    configuration.write_text(
        json.dumps(
            {
                "runner_bundle": "example.runner",
                "marker": "tmp/request.json",
                "routes": ["fixture:///reviews"],
            }
        )
    )
    apple = Apple()
    monkeypatch.setattr(
        apple,
        "simulators",
        lambda: [
            {
                "name": "Example",
                "udid": "chosen",
                "runtime": "com.apple.CoreSimulator.SimRuntime.iOS-27-0",
                "state": "Booted",
            }
        ],
    )
    descriptors: list[Path] = []

    def execute(
        _self: Processes, seconds: float, command: list[str], *, cwd: Path | None = None
    ) -> int:
        assert seconds == 12 and cwd == project.parent
        assert command[:2] == ["/usr/bin/xcrun", "swift"]
        assert Path(command[2]).name == "apple_url_tests.swift"
        descriptor = Path(command[3])
        descriptors.append(descriptor)
        data = json.loads(descriptor.read_text(encoding="utf-8"))
        assert data["device"] == "chosen" and data["routes"] == ["fixture:///reviews"]
        assert data["command"][0:3] == ["/usr/bin/xcrun", "xcodebuild", "test"]
        assert data["command"].count("-hideShellScriptEnvironment") == 1
        assert "-only-testing:Suite/Case" in data["command"]
        return 65

    monkeypatch.setattr(Processes, "timeout", execute)
    assert (
        apple.execute(
            project,
            scheme="App",
            action="test",
            destination="platform=iOS Simulator,name=Example,OS=27.0",
            bundle=tmp_path / "test.xcresult",
            url_test=configuration,
            only_testing=("Suite/Case",),
            timeout=12,
        )
        == 65
    )
    assert len(descriptors) == 1 and not descriptors[0].exists()
    with pytest.raises(MissionError, match="exact simulator"):
        apple.execute(
            project,
            scheme="App",
            action="test",
            destination="platform=iOS Simulator,name=Other,OS=27.0",
            bundle=tmp_path / "other.xcresult",
            url_test=configuration,
        )
