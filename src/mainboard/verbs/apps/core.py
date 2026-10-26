from ...manifest.apps import Apps, PlacedApp, SuiteName
from ...manifest.schema.services.app import Suite
from ..base import Verbs
from ..options import Output
from .apple import AppleVerbs

# The verbs' defaults: compact output.
_COMPACT = Output()
# The scheme's whole test action, no suite named.
_WHOLE = SuiteName("")


class AppVerbs(Verbs):
    """The apps a manifest declares: listed, built and tested from anywhere in the workspace."""

    apple: AppleVerbs

    def app_build(
        self, name: str, *, device: str = "", configuration: str = "Debug", timeout: float = 900
    ) -> int:
        """Build an app's scheme for a device, and retain the build's diagnostics.

        Args:
            name: the app, as `app list` shows it.
            device: one of its devices, `all` for every one; its first when omitted.
            configuration: Debug or Release.
            timeout: hard limit in seconds per device.
        """
        placed = Apps.at(self.session.workspace_root()).named(name)
        return self.app_run(
            placed, Suite(action="build", configuration=configuration, timeout=timeout), device
        )

    def app_list(self, *, output: Output = _COMPACT) -> None:
        """Every app in the workspace and its nested projects, with its devices and suites."""
        root = self.session.workspace_root()
        output.print_rows(
            [
                {
                    "app": name,
                    "home": str(placed.home.relative_to(root)),
                    "devices": " ".join(placed.spec.devices),
                    "suites": " ".join(placed.spec.suites),
                }
                for name, placed in Apps.at(root).found.items()
            ],
            title="apps",
            columns=("app", "home", "devices", "suites"),
        )

    def app_run(self, placed: PlacedApp, suite: Suite, device: str) -> int:
        """Run `suite` of `placed` on each chosen device through `apple execute`.

        The worst device's verdict is the run's.
        """
        placed.enter(suite)
        return max(
            self.apple.apple_execute(
                placed.project(suite),
                run=placed.run(suite, chosen),
                results=placed.home / ".build" / "Results",
            )
            for chosen in placed.devices(suite, device)
        )

    def app_test(self, name: str, suite: SuiteName = _WHOLE, *, device: str = "") -> int:
        """Run an app's suite, or its scheme's whole test action when none is named.

        Args:
            name: the app, as `app list` shows it.
            suite: one of its `[apps.<name>.suites]`.
            device: one of its devices, `all` for every one; the suite's when omitted.
        """
        placed = Apps.at(self.session.workspace_root()).named(name)
        return self.app_run(placed, placed.suite(suite), device)
