"""Launching a built app must keep explicit identity and native failure boundaries."""

import json
import plistlib
from pathlib import Path

import pytest

from mainboard.apple import Apple
from mainboard.cli import build
from mainboard.core import MissionError
from mainboard.core.tool import ToolOutput
from mainboard.proc import Processes


class LaunchTools:
    def __init__(self, root: Path) -> None:
        self.simctl = root / "Selected Xcode.app/Contents/Developer/usr/bin/simctl"
        self.viewer = self.simctl.parents[3] / "Applications/DeviceHub.app"
        self.viewer.mkdir(parents=True, exist_ok=True)
        self.devices = {
            "com.apple.CoreSimulator.SimRuntime.iOS-27-0": [
                {"name": "Phone", "udid": "one", "state": "Shutdown", "isAvailable": True}
            ]
        }

    def run(self, tool: str, args: tuple[str, ...], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == "/usr/bin/xcrun" and cwd is None
        if args == ("--find", "simctl"):
            return ToolOutput(stdout=str(self.simctl))
        assert args == ("simctl", "list", "devices", "available", "--json")
        return ToolOutput(stdout=json.dumps({"devices": self.devices}))


@pytest.fixture
def built_app(tmp_path: Path) -> Path:
    app = tmp_path / "Literal '$() app.app"
    app.mkdir()
    (app / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": "example.study"}))
    return app


@pytest.mark.parametrize("selection", ["Phone", "one"])
def test_launch_exact_device(
    built_app: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    calls: list[list[str]] = []

    def run(_self: Processes, seconds: float, command: list[str]) -> int:
        assert 0 < seconds <= 180
        calls.append(command)
        return 0

    monkeypatch.setattr(Processes, "timeout", run)
    result = Apple(LaunchTools(built_app.parent)).launch(built_app, simulator=selection)
    assert result["udid"] == "one" and result["bundle_identifier"] == "example.study"
    assert calls == [
        ["/usr/bin/xcrun", "simctl", "bootstatus", "one", "-b"],
        ["/usr/bin/xcrun", "simctl", "install", "one", str(built_app)],
        ["/usr/bin/xcrun", "simctl", "launch", "one", "example.study"],
        [
            "/usr/bin/open",
            "-a",
            str(built_app.parent / "Selected Xcode.app/Contents/Applications/DeviceHub.app"),
            "devices://device/open?id=one",
        ],
    ]


@pytest.mark.parametrize("stage", range(4))
def test_launch_stops_at_native_failure(
    built_app: Path, monkeypatch: pytest.MonkeyPatch, stage: int
) -> None:
    calls: list[list[str]] = []

    def run(_self: Processes, seconds: float, command: list[str]) -> int:
        calls.append(command)
        return 124 if len(calls) == stage + 1 else 0

    monkeypatch.setattr(Processes, "timeout", run)
    with pytest.raises(MissionError, match="exit 124"):
        Apple(LaunchTools(built_app.parent)).launch(built_app, simulator="one")
    assert len(calls) == stage + 1


def test_launch_requires_unique_available_match(
    built_app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = LaunchTools(built_app.parent)
    tools.devices["com.apple.CoreSimulator.SimRuntime.iOS-26-0"] = [
        {"name": "Phone", "udid": "two", "state": "Booted", "isAvailable": True}
    ]
    apple = Apple(tools)
    with pytest.raises(MissionError, match="matched 2"):
        apple.launch(built_app, simulator="Phone")
    with pytest.raises(MissionError, match="matched 0"):
        apple.launch(built_app, simulator="missing")
    tools.devices["com.apple.CoreSimulator.SimRuntime.iOS-26-0"][0]["isAvailable"] = False
    with pytest.raises(MissionError, match="matched 0"):
        apple.launch(built_app, simulator="two")
    monkeypatch.setattr(Processes, "timeout", lambda *_args: 0)
    assert apple.launch(built_app, simulator="Phone", os_version="27.0")["udid"] == "one"


@pytest.mark.parametrize(
    "content", [b"not a plist", plistlib.dumps({}), plistlib.dumps(["wrong"])]
)
def test_launch_rejects_invalid_bundle(built_app: Path, content: bytes) -> None:
    (built_app / "Info.plist").write_bytes(content)
    with pytest.raises(MissionError, match="Info.plist"):
        Apple().launch(built_app, simulator="one")


def test_launch_rejects_missing_bundle_and_invalid_request(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="built simulator"):
        Apple().launch(tmp_path / "missing.app", simulator="one")
    with pytest.raises(MissionError, match="positive timeout"):
        Apple().launch(tmp_path, simulator="one", timeout=0)
    with pytest.raises(MissionError, match="explicit simulator"):
        Apple().launch(tmp_path, simulator="")


def test_launch_cli_forwards_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[tuple[Path, str, str, float]] = []

    def launch(
        _self: Apple, path: Path, *, simulator: str, os_version: str, timeout: float
    ) -> dict[str, str]:
        captured.append((path, simulator, os_version, timeout))
        return {"udid": simulator}

    monkeypatch.setattr(Apple, "launch", launch)
    with pytest.raises(SystemExit) as result:
        build()(
            [
                "apple",
                "launch",
                str(tmp_path),
                "--simulator",
                "one",
                "--os-version",
                "27.0",
                "--timeout",
                "60",
            ]
        )
    assert result.value.code == 0
    assert captured == [(tmp_path, "one", "27.0", 60)]


def test_launch_deadline_prevents_later_steps(
    built_app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([0.0, 5.0, 181.0])
    calls: list[list[str]] = []

    def run(_self: Processes, seconds: float, command: list[str]) -> int:
        assert seconds == 175
        calls.append(command)
        return 0

    monkeypatch.setattr("mainboard.apple.monotonic", lambda: next(ticks))
    monkeypatch.setattr(Processes, "timeout", run)
    with pytest.raises(MissionError, match="timed out before installation"):
        Apple(LaunchTools(built_app.parent)).launch(built_app, simulator="one")
    assert len(calls) == 1


def test_launch_requires_selected_xcode_viewer_before_mutations(built_app: Path) -> None:
    tools = LaunchTools(built_app.parent)
    tools.viewer.rmdir()
    with pytest.raises(MissionError, match="requires Xcode 27 Device Hub"):
        Apple(tools).launch(built_app, simulator="one")


def test_launch_rejects_invalid_toolchain_path(built_app: Path) -> None:
    tools = LaunchTools(built_app.parent)
    tools.simctl = Path("simctl")
    with pytest.raises(MissionError, match="invalid selected simctl path"):
        Apple(tools).launch(built_app, simulator="one")
