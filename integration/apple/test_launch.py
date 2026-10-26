import json
import plistlib
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

import pytest
from pydantic import JsonValue

from mainboard import MissionError
from mainboard.apple import Apple
from mainboard.core import Runner, ToolOutput

from .builders import Node, devices_reply
from .constants import SIMCTL_LIST, XCRUN

_IOS27 = "com.apple.CoreSimulator.SimRuntime.iOS-27-0"
_IOS26 = "com.apple.CoreSimulator.SimRuntime.iOS-26-0"
_PHONE: Node = {"name": "Phone", "udid": "one", "state": "Shutdown", "isAvailable": True}
_TWIN: Node = {"name": "Phone", "udid": "two", "state": "Booted", "isAvailable": True}
_ONE_PHONE = {"simulator": "one"}
_ONE = {_IOS27: [_PHONE]}
_TWINS = {_IOS27: [_PHONE], _IOS26: [_TWIN]}
_BUILT = "Literal '$() app.app"
_VALID = plistlib.dumps({"CFBundleIdentifier": "example.study"})
_NO_IDENTIFIER = "needs a nonempty CFBundleIdentifier"
_REQUEST = "Apple launch needs an explicit simulator and a positive timeout"


class LaunchTools(Runner):
    """Answers xcrun's simctl location and device inventory for a made-up Xcode under `root`.

    devices: the inventory, by runtime; one available phone when unset.
    viewer: whether the Xcode holds a Device Hub.
    simctl: the path `xcrun --find simctl` answers; the made-up Xcode's when unset.
    """

    def __init__(
        self,
        root: Path,
        *,
        devices: Mapping[str, Sequence[JsonValue]] | None = None,
        viewer: bool = True,
        simctl: str | None = None,
    ) -> None:
        contents = root / "Selected Xcode.app/Contents"
        self.viewer = contents / "Applications/DeviceHub.app"
        self.simctl = simctl or str(contents / "Developer/usr/bin/simctl")
        self.devices = devices or _ONE
        if viewer:
            self.viewer.mkdir(parents=True)

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        assert tool == XCRUN and cwd is None
        if args == ("--find", "simctl"):
            return ToolOutput(stdout=self.simctl)
        assert args == SIMCTL_LIST
        return devices_reply(self.devices)


@pytest.fixture
def built_app(tmp_path: Path) -> Path:
    app = tmp_path / _BUILT
    app.mkdir()
    (app / "Info.plist").write_bytes(_VALID)
    return app


@pytest.fixture
def stalling(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], list[float]]:
    """Replace `subprocess.run` and the clocks: the inventory takes two seconds, a call stalls.

    Answers `stall(hang)`, which arranges it and answers the time limit each tool run was given.
    hang: `inventory` stalls the first run; `lookup` lets the inventory finish, then stalls.
    """

    def stall(hang: str) -> list[float]:
        now, limits = [0.0], []

        def run(
            command: tuple[str, ...],
            *,
            stdout: BinaryIO,
            timeout: float,
            **_ignored: Path | BinaryIO | bool | None,
        ) -> subprocess.CompletedProcess[bytes]:
            limits.append(timeout)
            if hang == "inventory" or command[1:] == ("--find", "simctl"):
                raise subprocess.TimeoutExpired(command, timeout)
            assert command[1:] == SIMCTL_LIST
            stdout.write(devices_reply(_ONE).stdout.encode())
            now[0] = 2.0
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr("mainboard.apple.bundle.launch.monotonic", lambda: now[0])
        monkeypatch.setattr("mainboard.core.execution.tool.monotonic", lambda: now[0])
        monkeypatch.setattr(subprocess, "run", run)
        return limits

    return stall


@pytest.mark.parametrize(
    ("devices", "simulator", "os_version"),
    [
        pytest.param(_ONE, "Phone", "", id="a-device-by-name"),
        pytest.param(_ONE, "one", "", id="a-device-by-udid"),
        pytest.param(_TWINS, "Phone", "27.0", id="a-shared-name-told-apart-by-runtime-version"),
    ],
)
def test_launch_boots_installs_launches_and_opens_the_device_hub_for_the_one_matching_device(
    built_app: Path, processes, devices, simulator: str, os_version: str
) -> None:
    tools = LaunchTools(built_app.parent, devices=devices)
    launched = Apple(tools).launch(built_app, simulator=simulator, os_version=os_version)
    assert launched == {
        "simulator": "Phone",
        "udid": "one",
        "runtime": _IOS27,
        "bundle_identifier": "example.study",
        "app": str(built_app),
    }
    assert all(0 < seconds <= 180 for seconds in processes.seconds)
    assert processes.commands == [
        [XCRUN, "simctl", "bootstatus", "one", "-b"],
        [XCRUN, "simctl", "install", "one", str(built_app)],
        [XCRUN, "simctl", "launch", "one", "example.study"],
        ["/usr/bin/open", "-a", str(tools.viewer), "devices://device/open?id=one"],
    ]


@pytest.mark.parametrize(
    ("stage", "name"),
    [
        pytest.param(0, "boot readiness", id="boot-readiness"),
        pytest.param(1, "installation", id="installation"),
        pytest.param(2, "launch", id="launch"),
        pytest.param(3, "Device Hub window", id="the-device-hub-window"),
    ],
)
def test_launch_stops_at_the_first_failing_native_stage(
    built_app: Path, processes, stage: int, name: str
) -> None:
    processes.exits = iter([0] * stage + [124])
    with pytest.raises(MissionError, match=rf"failed during {name} \(exit 124\)"):
        Apple(LaunchTools(built_app.parent)).launch(built_app, simulator="one")
    assert len(processes.commands) == stage + 1


@pytest.mark.parametrize(
    ("info", "app", "call", "match"),
    [
        pytest.param(
            b"not a plist", _BUILT, _ONE_PHONE, "cannot read app Info.plist", id="no-plist"
        ),
        pytest.param(plistlib.dumps({}), _BUILT, _ONE_PHONE, _NO_IDENTIFIER, id="no-identifier"),
        pytest.param(
            plistlib.dumps(["wrong"]), _BUILT, _ONE_PHONE, _NO_IDENTIFIER, id="no-dictionary"
        ),
        pytest.param(
            plistlib.dumps({"CFBundleIdentifier": " "}),
            _BUILT,
            _ONE_PHONE,
            _NO_IDENTIFIER,
            id="blank-identifier",
        ),
        pytest.param(
            plistlib.dumps({"CFBundleIdentifier": 5}),
            _BUILT,
            _ONE_PHONE,
            _NO_IDENTIFIER,
            id="numeric-identifier",
        ),
        pytest.param(
            _VALID, "missing.app", _ONE_PHONE, "expected a built simulator .app", id="missing-app"
        ),
        pytest.param(_VALID, _BUILT, {**_ONE_PHONE, "timeout": 0}, _REQUEST, id="zero-timeout"),
        pytest.param(
            _VALID, _BUILT, {**_ONE_PHONE, "timeout": float("inf")}, _REQUEST, id="endless-timeout"
        ),
        pytest.param(_VALID, _BUILT, {"simulator": " "}, _REQUEST, id="blank-simulator"),
        pytest.param(
            b"not a plist",
            _BUILT,
            {"simulator": " "},
            _REQUEST,
            id="a-blank-simulator-is-refused-before-the-bundle-is-read",
        ),
    ],
)
def test_launch_refuses_an_unreadable_bundle_or_an_invalid_request(
    built_app: Path, info: bytes, app: str, call, match: str
) -> None:
    (built_app / "Info.plist").write_bytes(info)
    with pytest.raises(MissionError, match=match):
        Apple().launch(built_app.parent / app, **call)


@pytest.mark.parametrize(
    ("world", "call", "match"),
    [
        pytest.param({"devices": _TWINS}, {"simulator": "Phone"}, "matched 2", id="a-shared-name"),
        pytest.param({}, {"simulator": "missing"}, "matched 0", id="an-unknown-simulator"),
        pytest.param(
            {"devices": {_IOS27: [_PHONE], _IOS26: [{**_TWIN, "isAvailable": False}]}},
            {"simulator": "two"},
            "matched 0",
            id="an-unavailable-device",
        ),
        pytest.param(
            {"viewer": False}, _ONE_PHONE, "requires Xcode 27 Device Hub", id="no-device-hub"
        ),
        pytest.param(
            {"simctl": "simctl"}, _ONE_PHONE, "invalid selected simctl path", id="relative-simctl"
        ),
    ],
)
def test_launch_refuses_an_unresolvable_device_or_viewer_before_any_native_stage(
    built_app: Path, processes, world, call, match: str
) -> None:
    with pytest.raises(MissionError, match=match):
        Apple(LaunchTools(built_app.parent, **world)).launch(built_app, **call)
    assert processes.commands == []


def test_launch_deadline_prevents_later_steps(
    built_app: Path, processes, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([0.0, 5.0, 181.0])
    monkeypatch.setattr("mainboard.apple.bundle.launch.monotonic", lambda: next(ticks))
    with pytest.raises(MissionError, match="timed out before installation"):
        Apple(LaunchTools(built_app.parent)).launch(built_app, simulator="one")
    assert processes.seconds == [175]


@pytest.mark.parametrize(
    ("hang", "limits"),
    [
        pytest.param("inventory", [3.0], id="the-inventory-stalls"),
        pytest.param("lookup", [3.0, 1.0], id="the-lookup-gets-what-the-inventory-left"),
    ],
)
def test_launch_metadata_processes_share_deadline(
    built_app: Path, stalling, hang: str, limits: list[float]
) -> None:
    asked = stalling(hang)
    with pytest.raises(MissionError, match="exit 124"):
        Apple().launch(built_app, simulator="one", timeout=3)
    assert asked == limits


def test_the_launch_verb_forwards_its_options(cli, built_app: Path, processes) -> None:
    tools = LaunchTools(built_app.parent, devices=_TWINS)
    code, out = cli(
        Apple(tools),
        "launch",
        str(built_app),
        *["--simulator", "Phone", "--os-version", "27.0", "--timeout", "60", "--json"],
    )
    assert code == 0 and json.loads(out)["udid"] == "one"
    assert 0 < max(processes.seconds) <= 60
