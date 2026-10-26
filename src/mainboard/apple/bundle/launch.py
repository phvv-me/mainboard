"""Install a built app and open it in one explicitly selected simulator."""

import math
import plistlib
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import quote

from ...core.errors import MissionError
from ...proc import Processes
from ..xcrun import bounded, simulators, xcrun

if TYPE_CHECKING:
    from ...core.execution.tool import Runner


class Launch:
    """One app launched on one simulator under a single deadline.

    The deadline bounds native discovery, boot readiness, installation, launch and opening the
    selected Xcode's Device Hub. Nothing is built, no device is chosen arbitrarily and no data
    is erased.
    """

    @staticmethod
    def _identifier(app: Path) -> str:
        """The bundle identifier a built `.app` declares in its Info.plist."""
        if app.suffix != ".app" or not app.is_dir():
            raise MissionError(f"expected a built simulator .app directory: {app}")
        try:
            info = plistlib.loads((app / "Info.plist").read_bytes())
        except (OSError, ValueError, plistlib.InvalidFileException) as fault:
            raise MissionError(f"cannot read app Info.plist: {fault}") from fault
        match info:
            case {"CFBundleIdentifier": str() as identifier} if identifier.strip():
                return identifier
            case _:
                raise MissionError("app Info.plist needs a nonempty CFBundleIdentifier")

    def __init__(self, runner: Runner, *, timeout: float) -> None:
        """runner: what the launch reads its devices and tools through, held to its deadline."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise MissionError("Apple launch needs an explicit simulator and a positive timeout")
        self.started = monotonic()
        self.timeout = timeout
        self.runner = bounded(runner, self.started + timeout)

    def _device(self, simulator: str, *, os_version: str) -> dict[str, str]:
        """The one available device `simulator` names, on `os_version` when one is given."""
        runtime = "-" + os_version.replace(".", "-")
        devices = [
            device
            for device in simulators(self.runner)
            if simulator in (device["name"], device["udid"])
            and (not os_version or device["runtime"].endswith(runtime))
        ]
        if len(devices) != 1:
            raise MissionError(
                f"simulator {simulator!r} matched {len(devices)} available devices; "
                "use apple simulators and specify an exact UDID or --os-version"
            )
        return devices[0]

    def _staged(self, stage: str, command: list[str]) -> None:
        """Run one stage in the time left, refusing a stage it cannot start or that fails."""
        remaining = self.timeout - (monotonic() - self.started)
        if remaining <= 0:
            raise MissionError(f"Apple launch timed out before {stage}")
        if code := Processes().timeout(remaining, command):
            raise MissionError(
                f"Apple launch failed during {stage} (exit {code}); see native output above"
            )

    def _viewer(self) -> Path:
        """The selected Xcode's Device Hub, found beside its simctl."""
        simctl = Path(xcrun(self.runner, "--find", "simctl").strip())
        if not simctl.is_absolute() or len(simctl.parents) < 4:
            raise MissionError(f"invalid selected simctl path: {simctl}")
        viewer = simctl.parents[3] / "Applications/DeviceHub.app"
        if not viewer.is_dir():
            raise MissionError(f"Apple launch requires Xcode 27 Device Hub at {viewer}")
        return viewer

    def run(self, app: Path, *, simulator: str, os_version: str = "") -> dict[str, str]:
        """Boot, install, launch `app` and show it in the Device Hub; what was launched where."""
        app = app.expanduser().resolve()
        if not simulator.strip():
            raise MissionError("Apple launch needs an explicit simulator and a positive timeout")
        identifier = Launch._identifier(app)
        device = self._device(simulator, os_version=os_version)
        udid = device["udid"]
        hub = f"devices://device/open?id={quote(udid, safe='')}"
        for stage, command in (
            ("boot readiness", ["/usr/bin/xcrun", "simctl", "bootstatus", udid, "-b"]),
            ("installation", ["/usr/bin/xcrun", "simctl", "install", udid, str(app)]),
            ("launch", ["/usr/bin/xcrun", "simctl", "launch", udid, identifier]),
            ("Device Hub window", ["/usr/bin/open", "-a", str(self._viewer()), hub]),
        ):
            self._staged(stage, command)
        return {
            "simulator": device["name"],
            "udid": udid,
            "runtime": device["runtime"],
            "bundle_identifier": identifier,
            "app": str(app),
        }
