import json
import os
import plistlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from plumbum import local

from mainboard.apple import Apple, toolchain
from mainboard.core import Runner, ToolOutput

from .constants import XCRUN

_UNSET = "<unset>"
_APPLICATIONS = "<applications>"
_COMMAND_LINE_TOOLS = "/Library/Developer/CommandLineTools"
_XCODE = "/Applications/Xcode.app/Contents/Developer"
_ANSWERS = {
    "xcodebuild -version": "Xcode 27.0",
    "swift --version": "Apple Swift 6.4",
    "--find swift": "/Xcode/usr/bin/swift",
    "--find xcresulttool": "/Xcode/usr/bin/xcresulttool",
}


def _plist(version: str) -> bytes:
    """An Info.plist declaring `version`."""
    return plistlib.dumps({"CFBundleShortVersionString": version})


_INSTALLED = {"Xcode.app": _plist("26.4"), "Xcode-beta.app": _plist("27.0")}


class SelectedTools(Runner):
    """Answers each xcrun query with the text registered for it."""

    def __init__(self, answers: Mapping[str, str]) -> None:
        self.answers = answers
        self.calls: list[Sequence[str]] = []

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == XCRUN and cwd is None
        self.calls.append(args)
        return ToolOutput(stdout=self.answers[" ".join(args)] + "\n")


def _install(root: Path, apps: Mapping[str, bytes | None]) -> None:
    """Make `root` hold the named Xcode apps; a `None` Info.plist leaves the app without one."""
    for name, info in apps.items():
        (root / name / "Contents").mkdir(parents=True)
        if info is not None:
            (root / name / "Contents" / "Info.plist").write_bytes(info)


@pytest.fixture
def xcode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """Run `Apple.select_xcode` over Xcode apps and an xcode-select link made here."""

    def select(
        apps: Mapping[str, bytes | None],
        selected: str | None,
        *,
        environment: Mapping[str, str] | None = None,
        system: str = "Darwin",
    ) -> str:
        _install(tmp_path, apps)
        if selected is not None:
            (tmp_path / "link").symlink_to(selected)
        monkeypatch.setattr(toolchain.platform, "system", lambda: system)
        monkeypatch.setenv("DEVELOPER_DIR", "")
        monkeypatch.delenv("DEVELOPER_DIR")
        for key, value in (environment or {}).items():
            monkeypatch.setenv(key, value)
        # plumbum's copy of the environment, which the selection also sets, is restored afterwards.
        with local.env():
            local.env.pop("DEVELOPER_DIR", "")
            Apple.select_xcode(link=str(tmp_path / "link"), applications=tmp_path)
        return os.environ.get("DEVELOPER_DIR", _UNSET).replace(str(tmp_path), _APPLICATIONS)

    return select


@pytest.mark.parametrize(
    ("arrange", "expected"),
    [
        pytest.param(
            {"apps": _INSTALLED, "selected": _COMMAND_LINE_TOOLS},
            f"{_APPLICATIONS}/Xcode-beta.app/Contents/Developer",
            id="the-command-line-tools-give-way-to-the-newest-xcode",
        ),
        pytest.param(
            {
                "apps": {"Xcode.app": _plist("26.9"), "Xcode-next.app": _plist("26.10")},
                "selected": _COMMAND_LINE_TOOLS,
            },
            f"{_APPLICATIONS}/Xcode-next.app/Contents/Developer",
            id="versions-compare-as-numbers",
        ),
        pytest.param(
            {
                "apps": {**_INSTALLED, "Xcode-damaged.app": None, "Xcode-junk.app": b"junk"},
                "selected": _COMMAND_LINE_TOOLS,
            },
            f"{_APPLICATIONS}/Xcode-beta.app/Contents/Developer",
            id="an-unreadable-xcode-never-outranks-a-readable-one",
        ),
        pytest.param(
            {"apps": _INSTALLED, "selected": _XCODE}, _UNSET, id="a-selected-xcode-is-left-alone"
        ),
        pytest.param(
            {
                "apps": _INSTALLED,
                "selected": _COMMAND_LINE_TOOLS,
                "environment": {"DEVELOPER_DIR": "/elsewhere"},
            },
            "/elsewhere",
            id="the-callers-developer-dir-is-kept",
        ),
        pytest.param(
            {"apps": _INSTALLED, "selected": None},
            _UNSET,
            id="no-xcode-select-link-changes-nothing",
        ),
        pytest.param(
            {"apps": {}, "selected": _COMMAND_LINE_TOOLS}, _UNSET, id="no-installed-xcode"
        ),
        pytest.param(
            {"apps": _INSTALLED, "selected": _COMMAND_LINE_TOOLS, "system": "Linux"},
            _UNSET,
            id="another-platform-is-left-alone",
        ),
    ],
)
def test_select_xcode_replaces_only_the_command_line_tools(xcode, arrange, expected) -> None:
    assert xcode(**arrange) == expected


def test_the_toolchain_verb_reads_the_selected_tools_without_changing_the_selection(cli) -> None:
    tools = SelectedTools(_ANSWERS)
    code, out = cli(Apple(tools), "toolchain", "--json")
    assert code == 0
    assert json.loads(out) == {
        "xcode": "Xcode 27.0",
        "swift": "Apple Swift 6.4",
        "swift_path": "/Xcode/usr/bin/swift",
        "xcresulttool_path": "/Xcode/usr/bin/xcresulttool",
    }
    assert [" ".join(call) for call in tools.calls] == list(_ANSWERS)
