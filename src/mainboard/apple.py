"""Read Apple's installed toolchain and test evidence without launching the IDE."""

from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import ConfigDict, Field, JsonValue, ValidationError

from .core.errors import MissionError
from .core.tool import Runner, ToolRunner

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


class TestNode(FrozenModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str
    kind: str = Field(alias="nodeType")
    result: str
    identifier: str = Field(default="", alias="nodeIdentifierURL")
    seconds: float | None = Field(default=None, alias="durationInSeconds", ge=0)
    children: list[TestNode] = Field(default_factory=list)

    def cases(self) -> Iterator[TestNode]:
        if self.kind == "Test Case":
            yield self
        for child in self.children:
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


class Simulator(FrozenModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)
    udid: str = Field(min_length=1)
    state: str = Field(min_length=1)
    available: bool = Field(alias="isAvailable")


class SimulatorInventory(FrozenModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    devices: dict[str, list[Simulator]]


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
