"""What Xcode leaves behind: test and build results read from a result bundle, and simulators."""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from patos import FrozenModel
from pydantic import ConfigDict, Field, JsonValue

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..render.values import Node


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
    kind: Literal["Failure Message", "Runtime Warning", "Skip Message"] = Field(alias="nodeType")


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
    missing_selectors: list[str] = Field(default_factory=list)

    @property
    def successful(self) -> bool:
        return self.summary.successful and not self.missing_selectors

    def document(self) -> dict[str, Node]:
        return {
            "bundle": str(self.bundle),
            "successful": self.successful,
            "missing_selectors": tuple(self.missing_selectors),
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

    @property
    def successful(self) -> bool:
        return self.summary.successful

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
