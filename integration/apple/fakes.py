import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import JsonValue

from mainboard.core import Runner, ToolOutput

from .builders import passing_nodes, succeeded_build
from .constants import XCRUN


class ResultTool(Runner):
    """Answers xcrun's xcresulttool from canned evidence, or every call from `reply`.

    summary: test-summary fields overriding the passing run's.
    nodes: the test tree, replacing the passing run's.
    build: build-summary fields overriding the succeeded build's.
    reply: what every call answers instead of the evidence.
    built_only: the bundle holds a build and no test results, so a test read fails.
    """

    @staticmethod
    def _passing_summary() -> dict[str, JsonValue]:
        """The test summary of a run that passed both of its cases."""
        return {
            "result": "Passed",
            "totalTestCount": 2,
            "passedTests": 2,
            "failedTests": 0,
            "skippedTests": 0,
            "expectedFailures": 0,
            "testFailures": [],
            "runtimeWarnings": [{"issueType": "Warning", "message": "example"}],
            "devicesAndConfigurations": [{"device": {"deviceName": "iPhone"}}],
        }

    def __init__(
        self,
        *,
        summary: Mapping[str, JsonValue] | None = None,
        nodes: Sequence[JsonValue] | None = None,
        build: Mapping[str, JsonValue] | None = None,
        reply: ToolOutput | None = None,
        built_only: bool = False,
    ) -> None:
        self.built_only = built_only
        self.summary = {**self._passing_summary(), **(summary or {})}
        self.nodes = passing_nodes() if nodes is None else list(nodes)
        self.build_summary = {**succeeded_build(), **(build or {})}
        self.reply = reply
        self.calls: list[Sequence[str]] = []

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == XCRUN and cwd is None
        self.calls.append(args)
        if self.reply is not None:
            return self.reply
        if self.built_only and args[2] == "test-results":
            return ToolOutput(returncode=1, stderr="Error: no test results in the bundle")
        answers: dict[str, JsonValue] = {
            "xcresulttool get build-results --path": self.build_summary,
            "xcresulttool get test-results summary": self.summary,
            "xcresulttool get test-results tests": {"testNodes": self.nodes},
        }
        return ToolOutput(stdout=json.dumps(answers[" ".join(args[:4])]))
