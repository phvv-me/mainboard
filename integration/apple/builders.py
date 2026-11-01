import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from patos import FrozenModel
from pydantic import JsonValue

from mainboard.core import ToolOutput
from mainboard.verbs.apps.enums import Kind

from .constants import IDENTIFIER_HOST

type Node = dict[str, JsonValue]

_SUITE_SECONDS = 1000
_WARNED_SOURCE = "file:///tmp/Study%20View.swift#StartingLineNumber=42"
_PLUTIL = ("plutil",)
_XCRESULTTOOL = ("xcresulttool",)


class Layout(FrozenModel):
    """A path as a build or a test run leaves it, and what `mb code app inspect` makes of it.

    suffix: how the path ends.
    kind: the kind inspected, `None` for a path of none.
    children: files, and folders ending in `/`, laid inside the path.
    file: the path is a file, not a folder.
    tools: the tools its inspector asks, in order, before the first one fails.
    """

    suffix: str
    kind: Kind | None
    children: Sequence[str] = ()
    file: bool = False
    tools: Sequence[str] = ()

    def lay(self, root: Path, stem: str) -> Path:
        """Create the layout under `root` as `stem` and its suffix, answering its path."""
        path = root / f"{stem}{self.suffix}"
        if self.file:
            root.mkdir(parents=True)
            path.write_text("text")
            return path
        path.mkdir(parents=True)
        for child in self.children:
            (path / child).parent.mkdir(parents=True, exist_ok=True)
            if child.endswith("/"):
                (path / child).mkdir()
            else:
                (path / child).write_text("")
        return path


def _ignored(command: Sequence[str]) -> None:
    """Do nothing while a command runs."""


class ProcessLog:
    """Stands in for `Processes.timeout`: records each command, answers the planned exit codes.

    commands: every command asked, in order, with its time limit in `seconds` and its working
        directory in `directories`.
    exits: the exit codes of the commands in order; a command past them exits 0.
    during: called with each command while it "runs", before it is answered.
    """

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.seconds: list[float] = []
        self.directories: list[Path | None] = []
        self.exits: Iterator[int] = iter(())
        self.during: Callable[[Sequence[str]], None] = _ignored

    def timeout(self, seconds: float, command: Sequence[str], *, cwd: Path | None = None) -> int:
        self.commands.append(list(command))
        self.seconds.append(seconds)
        self.directories.append(cwd)
        self.during(command)
        return next(self.exits, 0)


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


def layouts() -> list[Layout]:
    """Every layout, a bundle's, a result bundle's or neither's."""
    return [
        Layout(suffix=".app", kind=Kind.BUNDLE, tools=_PLUTIL),
        Layout(suffix=".xcarchive", kind=Kind.BUNDLE, tools=_PLUTIL),
        Layout(suffix="", kind=Kind.BUNDLE, children=("Info.plist",), tools=_PLUTIL),
        Layout(suffix="", kind=Kind.BUNDLE, children=("Contents/Info.plist",), tools=_PLUTIL),
        Layout(suffix=".app", kind=Kind.BUNDLE, file=True),
        Layout(suffix=".xcresult", kind=Kind.RESULTS, tools=_XCRESULTTOOL),
        Layout(suffix="", kind=Kind.RESULTS, children=("Run.xcresult/",), tools=_XCRESULTTOOL),
        Layout(suffix=".xcresult", kind=Kind.RESULTS, file=True),
        Layout(suffix="", kind=None),
        Layout(suffix=".txt", kind=None, file=True),
        Layout(suffix="", kind=None, children=("notes.txt", "Sources/")),
    ]
