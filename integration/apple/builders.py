import json
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from mainboard.core import ToolOutput

from .constants import IDENTIFIER_HOST

type Node = dict[str, JsonValue]

_SUITE_SECONDS = 1000
_WARNED_SOURCE = "file:///tmp/Study%20View.swift#StartingLineNumber=42"


def node(
    name: str,
    *,
    kind: str,
    result: str | None = None,
    seconds: float | None = None,
    children: Sequence[JsonValue] = (),
    **extra: JsonValue,
) -> Node:
    """One xcresulttool node; a field left unset is omitted, as Xcode omits it."""
    optional = {
        "result": result,
        "durationInSeconds": seconds,
        "children": list(children) or None,
    }
    present = {key: value for key, value in optional.items() if value is not None}
    return {"name": name, "nodeType": kind, **present, **extra}


def case(
    name: str,
    *,
    result: str = "Passed",
    seconds: float | None = None,
    children: Sequence[JsonValue] = (),
    **extra: JsonValue,
) -> Node:
    """One test case node."""
    return node(name, kind="Test Case", result=result, seconds=seconds, children=children, **extra)


def passing_nodes() -> list[JsonValue]:
    """The test tree of the passing run: a suite of two cases, one with an argument child."""
    argument = node("argument", kind="Arguments", result="Passed", seconds=4)
    landscape = case(
        "landscape",
        seconds=801,
        nodeIdentifierURL=f"{IDENTIFIER_HOST}/App/Tests/Suite/landscape",
    )
    parameterized = case("parameterized", seconds=2, children=[argument])
    return [
        node(
            "Suite",
            kind="Test Suite",
            result="Passed",
            seconds=_SUITE_SECONDS,
            children=[landscape, parameterized],
        )
    ]


def succeeded_build() -> dict[str, JsonValue]:
    """The build summary of a build that succeeded with one located warning."""
    warning: Node = {
        "issueType": "Swift Compiler Warning",
        "message": "Actor-isolated property cannot be referenced here",
        "targetName": "StudyUI",
        "sourceURL": _WARNED_SOURCE,
        "className": "DVTTextDocumentLocation",
    }
    return {
        "status": "succeeded",
        "errorCount": 0,
        "warningCount": 1,
        "analyzerWarningCount": 0,
        "errors": [],
        "warnings": [warning],
        "analyzerWarnings": [],
        "destination": {"platform": "macOS", "architecture": "arm64"},
        "actionTitle": "Building project Example with scheme Example",
    }


def devices_reply(runtimes: Mapping[str, Sequence[JsonValue]]) -> ToolOutput:
    """What `simctl list devices available --json` prints for devices grouped by runtime."""
    return ToolOutput(stdout=json.dumps({"devices": runtimes}))
