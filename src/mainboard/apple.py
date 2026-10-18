"""Read Apple's installed toolchain and test evidence without launching the IDE."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from patos import FrozenModel
from pydantic import ConfigDict, Field, JsonValue, ValidationError

from .core.errors import MissionError
from .core.tool import Runner, ToolRunner
from .proc import Processes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .render.values import Node


class Summary(FrozenModel):
    """The verdict fields are required; additional Xcode metadata remains in the report."""

    model_config = ConfigDict(extra="allow", strict=True)

    result: str
    total: int = Field(alias="totalTestCount", ge=0)
    passed: int = Field(alias="passedTests", ge=0)
    failed: int = Field(alias="failedTests", ge=0)
    skipped: int = Field(alias="skippedTests", ge=0)
    expected_failures: int = Field(alias="expectedFailures", ge=0)
    failures: list[dict[str, JsonValue]] = Field(alias="testFailures")

    @property
    def successful(self) -> bool:
        return (
            self.result == "Passed"
            and 0 < self.passed <= self.total
            and not (self.failed or self.failures)
        )


class DiagnosticNode(FrozenModel):
    """Xcode diagnostic leaves lack verdicts, including warnings in passing cases."""

    model_config = ConfigDict(extra="allow", strict=True)

    name: str
    kind: Literal["Failure Message", "Runtime Warning"] = Field(alias="nodeType")


class TestNode(FrozenModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str
    kind: str = Field(alias="nodeType")
    result: str
    identifier: str = Field(default="", alias="nodeIdentifierURL")
    seconds: float | None = Field(default=None, alias="durationInSeconds", ge=0)
    children: list[TestNode | DiagnosticNode] = Field(default_factory=list)

    def cases(self) -> Iterator[TestNode]:
        if self.kind == "Test Case":
            yield self
        for child in self.children:
            if isinstance(child, TestNode):
                yield from child.cases()


class TestTree(FrozenModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    nodes: list[TestNode] = Field(alias="testNodes")


class TestReport(FrozenModel):
    bundle: Path
    summary: Summary
    slowest: list[TestNode]

    def document(self) -> dict[str, Node]:
        return {
            "bundle": str(self.bundle),
            "successful": self.summary.successful,
            "summary": self.summary.model_dump(mode="json", by_alias=True),
            "slowest": [
                node.model_dump(mode="json", exclude={"children", "kind"}) for node in self.slowest
            ],
        }


class BuildIssue(FrozenModel):
    model_config = ConfigDict(extra="allow", strict=True)

    kind: str = Field(alias="issueType")
    message: str
    target: str | None = Field(default=None, alias="targetName")
    source: str | None = Field(default=None, alias="sourceURL")


class BuildSummary(FrozenModel):
    """Keep compiler locations and unknown metadata; warnings do not fail a build."""

    model_config = ConfigDict(extra="allow", strict=True)

    status: str
    error_count: int = Field(alias="errorCount", ge=0)
    warning_count: int = Field(alias="warningCount", ge=0)
    analyzer_warning_count: int = Field(alias="analyzerWarningCount", ge=0)
    errors: list[BuildIssue]
    warnings: list[BuildIssue]
    analyzer_warnings: list[BuildIssue] = Field(alias="analyzerWarnings")

    @property
    def successful(self) -> bool:
        return self.status == "succeeded" and not (self.error_count or self.errors)


class BuildReport(FrozenModel):
    bundle: Path
    summary: BuildSummary

    def document(self) -> dict[str, Node]:
        return {
            "bundle": str(self.bundle),
            "build_successful": self.summary.successful,
            "summary": self.summary.model_dump(mode="json", by_alias=True, exclude_none=True),
        }


class Simulator(FrozenModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)
    udid: str = Field(min_length=1)
    state: str = Field(min_length=1)
    available: bool = Field(alias="isAvailable")


class SimulatorInventory(FrozenModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    devices: dict[str, list[Simulator]]


class URLTestConfiguration(FrozenModel):
    """Routes delivered by a host while an isolated simulator UI test runs."""

    runner_bundle: str = Field(min_length=1)
    marker: str = Field(min_length=1)
    routes: tuple[str, ...] = Field(min_length=1)

    def validate_routes(self) -> None:
        marker = Path(self.marker)
        if marker.is_absolute() or ".." in marker.parts:
            raise MissionError("URL test marker must stay inside the runner data container")
        if len(set(self.routes)) != len(self.routes):
            raise MissionError("URL test routes must be unique")
        for route in self.routes:
            value = urlsplit(route)
            if (
                not value.scheme
                or value.netloc
                or not value.path.startswith("/")
                or value.query
                or value.fragment
            ):
                raise MissionError("URL tests require explicit local custom-scheme paths")


class Apple:
    """Use xcrun's selected Xcode (including DEVELOPER_DIR), with bounded tool output."""

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner if runner is not None else ToolRunner(timeout=30)

    def toolchain(self) -> dict[str, str]:
        return {
            "xcode": self._run("xcodebuild", "-version").strip(),
            "swift": self._run("swift", "--version").strip(),
            "swift_path": self._run("--find", "swift").strip(),
            "xcresulttool_path": self._run("--find", "xcresulttool").strip(),
        }

    def simulators(self) -> list[dict[str, str]]:
        """List available devices without booting, creating or erasing a simulator."""
        payload = self._run("simctl", "list", "devices", "available", "--json")
        try:
            inventory = SimulatorInventory.model_validate_json(payload)
        except ValidationError as fault:
            raise MissionError(f"invalid simctl device inventory: {fault}") from fault
        return [
            {"runtime": runtime, "name": device.name, "udid": device.udid, "state": device.state}
            for runtime, devices in sorted(inventory.devices.items())
            for device in sorted(devices, key=lambda item: (item.name, item.udid))
            if device.available
        ]

    def execute(
        self,
        project: Path,
        *,
        scheme: str,
        action: Literal["build", "test", "build-for-testing", "resolve"] = "build",
        destination: str = "",
        configuration: str = "Debug",
        derived_data: Path | None = None,
        bundle: Path | None = None,
        only_testing: tuple[str, ...] = (),
        skip_testing: tuple[str, ...] = (),
        settings: tuple[str, ...] = (),
        timeout: float = 900,
        url_test: Path | None = None,
    ) -> int:
        """Build/test a workspace, project or Swift package with the selected Xcode.

        Keep native stdout and exit status, use committed package versions, and stop the
        entire process tree on timeout. Result bundles must be fresh, never overwritten.
        """
        project = project.expanduser().resolve()
        if timeout <= 0 or not scheme.strip():
            raise MissionError("Apple execution needs a scheme and a positive timeout")
        if not project.is_dir():
            raise MissionError(f"no Apple project at {project}")
        if project.suffix in {".xcworkspace", ".xcodeproj"}:
            source: tuple[str, ...] = (
                "-workspace" if project.suffix == ".xcworkspace" else "-project",
                str(project),
            )
            cwd = project.parent
        elif (project / "Package.swift").is_file():
            source = ()
            cwd = project
        else:
            raise MissionError("expected an .xcworkspace, .xcodeproj or Swift package directory")
        command = [
            "/usr/bin/xcrun",
            "xcodebuild",
            "-resolvePackageDependencies" if action == "resolve" else action,
            *source,
            "-scheme",
            scheme,
            "-onlyUsePackageVersionsFromResolvedFile",
        ]
        if action != "resolve":
            if bundle is None:
                raise MissionError("Apple builds/tests need a fresh result bundle path")
            bundle = bundle.expanduser().resolve()
            if bundle.exists():
                raise MissionError(f"refusing to overwrite result evidence at {bundle}")
            bundle.parent.mkdir(parents=True, exist_ok=True)
            command += ["-resultBundlePath", str(bundle), "-configuration", configuration]
            if destination:
                command += ["-destination", destination]
            if action == "test":
                command += ["-parallel-testing-enabled", "NO"]
        if derived_data is not None:
            command += ["-derivedDataPath", str(derived_data.expanduser().resolve())]
        command += [f"-only-testing:{value}" for value in only_testing]
        command += [f"-skip-testing:{value}" for value in skip_testing]
        command += list(settings)
        if url_test is None:
            return Processes().timeout(timeout, command, cwd=cwd)
        if action != "test":
            raise MissionError("URL delivery requires the test action")
        try:
            config = URLTestConfiguration.model_validate_json(url_test.read_text())
        except (OSError, ValidationError) as fault:
            raise MissionError(f"invalid URL test configuration: {fault}") from fault
        config.validate_routes()
        fields = dict(part.split("=", 1) for part in destination.split(",") if "=" in part)
        if fields.get("platform") != "iOS Simulator":
            raise MissionError("URL delivery is only supported on an iOS simulator")
        candidates = [
            device
            for device in self.simulators()
            if (not fields.get("id") or device["udid"] == fields["id"])
            and (not fields.get("name") or device["name"] == fields["name"])
            and (
                not fields.get("OS")
                or device["runtime"].endswith(".iOS-" + fields["OS"].replace(".", "-"))
            )
        ]
        if len(candidates) != 1:
            raise MissionError("URL tests require one exact simulator destination")
        with TemporaryDirectory(prefix="mainboard-apple-url-") as temporary:
            descriptor = Path(temporary) / "request.json"
            descriptor.write_text(
                json.dumps(
                    {
                        **config.model_dump(mode="json"),
                        "device": candidates[0]["udid"],
                        "command": command,
                    }
                )
            )
            helper = Path(__file__).with_name("apple_url_tests.swift")
            return Processes().timeout(
                timeout,
                ["/usr/bin/xcrun", "swift", str(helper), str(descriptor)],
                cwd=cwd,
            )

    def results(self, path: Path) -> TestReport:
        """Read an exact bundle, or the newest direct child of a Logs/Test directory.

        Never fall back to an older successful bundle when the newest one is incomplete.
        Timings are Xcode's test-case durations, not sums of parallel parameterized runs.
        """
        bundle = self._bundle(path)
        summary_text = self._result("summary", bundle)
        tests_text = self._result("tests", bundle)
        try:
            summary = Summary.model_validate_json(summary_text)
            tree = TestTree.model_validate_json(tests_text)
        except ValidationError as fault:
            raise MissionError(
                f"invalid xcresulttool test evidence for {bundle}: {fault}"
            ) from fault
        cases = (case for node in tree.nodes for case in node.cases() if case.seconds is not None)
        return TestReport(
            bundle=bundle,
            summary=summary,
            slowest=sorted(cases, key=lambda case: case.seconds or 0, reverse=True)[:10],
        )

    def diagnostics(self, path: Path) -> BuildReport:
        """Read the build action, including bundles that failed before tests could start.

        A successful build action says nothing about subsequent test results. Incremental
        builds report only the diagnostics Xcode retained for that action.
        """
        bundle = self._bundle(path)
        payload = self._run(
            "xcresulttool", "get", "build-results", "--path", str(bundle), "--compact"
        )
        try:
            summary = BuildSummary.model_validate_json(payload)
        except ValidationError as fault:
            raise MissionError(
                f"invalid xcresulttool build evidence for {bundle}: {fault}"
            ) from fault
        return BuildReport(bundle=bundle, summary=summary)

    def _result(self, section: str, bundle: Path) -> str:
        return self._run(
            "xcresulttool", "get", "test-results", section, "--path", str(bundle), "--compact"
        )

    def _run(self, *arguments: str) -> str:
        found = self.runner.run("/usr/bin/xcrun", arguments)
        if found.returncode or found.truncated:
            raise MissionError(
                f"xcrun {' '.join(arguments[:4])} failed "
                f"(exit {found.returncode}, truncated={found.truncated}): "
                f"{found.stderr.strip() or 'no complete output; check Xcode and DEVELOPER_DIR'}"
            )
        return found.stdout

    @staticmethod
    def _bundle(path: Path) -> Path:
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise MissionError(f"no result bundle or test-log directory at {path}")
        if path.suffix == ".xcresult":
            return path
        candidates = [item for item in path.glob("*.xcresult") if item.is_dir()]
        if not candidates:
            raise MissionError(f"no .xcresult bundles directly inside {path}")
        return max(candidates, key=lambda item: (item.stat().st_mtime_ns, item.name))
