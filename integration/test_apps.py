"""`[apps.<name>]` is found wherever the workspace keeps it, and a suite resolves to the exact
xcodebuild inputs its old task spelled out."""

import subprocess
from pathlib import Path

import pytest

from mainboard.apps import Apps
from mainboard.core.errors import MissionError

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


def test_a_suite_resolves_to_the_inputs_its_task_spelled(tmp_path: Path) -> None:
    _workspace(tmp_path)
    placed = Apps.at(tmp_path).named("phone")
    fonts = placed.suite("fonts")

    assert placed.devices(fonts, "") == ["ipad"], "the app's first device"
    assert placed.devices(fonts, "all") == ["ipad", "mac", "iphone"]
    assert placed.destination("ipad") == (
        "platform=iOS Simulator,name=iPad Pro 11-inch (M5),OS=27.0"
    )
    assert placed.destination("mac") == "platform=macOS"
    assert placed.settings(fonts, "iphone") == (
        "-skipPackagePluginValidation",
        "CODE_SIGNING_ALLOWED=NO",
    )
    assert placed.derived_data(fonts, "iphone") == placed.home / ".build/Xcode/iOS"
    assert placed.derived_data(fonts, "ipad") == placed.home / ".build/Xcode/ipad"

    shared = placed.suite("shared")
    assert placed.devices(shared, "") == ["mac"]
    assert placed.project(shared) == placed.home / "Shared"
    assert placed.derived_data(shared, "mac") == placed.home / ".build/Xcode/Shared"
    with pytest.raises(MissionError, match="no device 'watch'"):
        placed.devices(fonts, "watch")
    with pytest.raises(MissionError, match="no suite 'audio'.*fonts, shared"):
        placed.suite("audio")
