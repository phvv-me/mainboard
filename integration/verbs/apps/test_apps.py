"""`[apps.<name>]` is found wherever the workspace keeps it, and a suite resolves to the exact
xcodebuild inputs its old task spelled out."""

import subprocess
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.verbs.apps.placed import Apps, PlacedApp, SuiteName

_APP = """
[workspace]
name = "phone"

[apps.phone]
project = "App/Phone.xcworkspace"
scheme = "Phone"
runtime = "27.0"
settings = ["-skipPackagePluginValidation"]
env = { PROTOC_PATH = false }

[apps.phone.devices]
ipad = "iPad Pro 11-inch (M5)"
mac = { destination = "platform=macOS", settings = ["CODE_SIGN_IDENTITY=-"] }

[apps.phone.devices.iphone]
destination = "iPhone 18 Pro"
derived-data = ".build/Xcode/iOS"
settings = ["CODE_SIGNING_ALLOWED=NO"]

[apps.phone.suites.fonts]
scheme = "ReviewValidation"
only = ["UI/FlowTests/testFonts"]
timeout = 600

[apps.phone.suites.shared]
project = "Shared"
derived-data = ".build/Xcode/Shared"
devices = ["mac"]
"""


def _workspace(root: Path) -> Path:
    """A git workspace whose app lives in a repository nested untracked inside it."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "mainboard.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
    nested = root / "apps" / "phone"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / "mainboard.toml").write_text(_APP, encoding="utf-8")
    return nested


def test_an_app_in_an_untracked_nested_repository_is_found(tmp_path: Path) -> None:
    nested = _workspace(tmp_path)

    placed = Apps.at(tmp_path).named("phone")

    assert placed.home == nested
    with pytest.raises(MissionError, match="no app 'tablet'.*phone"):
        Apps.at(tmp_path).named("tablet")


@pytest.fixture
def placed(tmp_path: Path) -> PlacedApp:
    """The `phone` app of `_APP`, found in its nested repository."""
    _workspace(tmp_path)
    return Apps.at(tmp_path).named("phone")


@pytest.mark.parametrize(
    ("suite", "named", "devices"),
    [
        pytest.param("fonts", "", ["ipad"], id="the-apps-first-device"),
        pytest.param("fonts", "all", ["ipad", "mac", "iphone"], id="every-device-of-the-app"),
        pytest.param("shared", "", ["mac"], id="the-devices-the-suite-lists"),
    ],
)
def test_a_suite_runs_on_the_devices_its_task_spelled(
    placed: PlacedApp, suite: str, named: str, devices: list[str]
) -> None:
    assert placed.devices(placed.suite(SuiteName(suite)), named) == devices


@pytest.mark.parametrize(
    ("device", "destination"),
    [
        pytest.param(
            "ipad",
            "platform=iOS Simulator,name=iPad Pro 11-inch (M5),OS=27.0",
            id="a-simulator-at-the-apps-runtime",
        ),
        pytest.param("mac", "platform=macOS", id="a-destination-spelled-whole"),
    ],
)
def test_a_device_resolves_to_the_destination_its_task_spelled(
    placed: PlacedApp, device: str, destination: str
) -> None:
    assert placed.destination(device) == destination


def test_a_device_adds_its_settings_to_the_apps(placed: PlacedApp) -> None:
    settings = placed.settings(placed.suite(SuiteName("fonts")), "iphone")

    assert settings == ("-skipPackagePluginValidation", "CODE_SIGNING_ALLOWED=NO")


@pytest.mark.parametrize(
    ("suite", "device", "folder"),
    [
        pytest.param("fonts", "iphone", ".build/Xcode/iOS", id="the-device-folder-it-declares"),
        pytest.param("fonts", "ipad", ".build/Xcode/ipad", id="a-folder-named-for-the-device"),
        pytest.param("shared", "mac", ".build/Xcode/Shared", id="the-folder-the-suite-declares"),
    ],
)
def test_a_build_keeps_its_products_where_its_task_spelled(
    placed: PlacedApp, suite: str, device: str, folder: str
) -> None:
    derived = placed.derived_data(placed.suite(SuiteName(suite)), device)

    assert derived == placed.home / folder


def test_a_suite_builds_the_project_it_declares(placed: PlacedApp) -> None:
    assert placed.project(placed.suite(SuiteName("shared"))) == placed.home / "Shared"


def test_a_device_the_app_does_not_declare_is_refused(placed: PlacedApp) -> None:
    with pytest.raises(MissionError, match="no device 'watch'"):
        placed.devices(placed.suite(SuiteName("fonts")), "watch")


def test_a_suite_the_app_does_not_declare_is_refused(placed: PlacedApp) -> None:
    with pytest.raises(MissionError, match="no suite 'audio'.*fonts, shared"):
        placed.suite(SuiteName("audio"))
