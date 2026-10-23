"""The apps a workspace declares, wherever in it: the root's `[apps]` and each nested project's, so
`mb app` reaches an app from any directory without the root repeating its build line per test."""

import os
import tomllib
from pathlib import Path

from patos import FrozenModel
from plumbum import local

from .core.errors import MissionError
from .core.project import Project
from .lint.git import nested_projects
from .manifest.schema.app import App, Suite


class PlacedApp(FrozenModel):
    """An app and the directory its manifest sits in, which its paths are relative to."""

    home: Path
    name: str
    spec: App

    def derived_data(self, suite: Suite, device: str) -> Path:
        """Where a build keeps its products: the suite's folder, else the device's."""
        declared = suite.derived_data or self.spec.devices[device].derived_data
        return self.home / (declared or f".build/Xcode/{device}")

    def destination(self, device: str) -> str:
        """The xcodebuild destination of `device`; a simulator's name runs the app's runtime."""
        declared = self.spec.devices[device].destination
        if declared.startswith("platform="):
            return declared
        runtime = f",OS={self.spec.runtime}" if self.spec.runtime else ""
        return f"platform=iOS Simulator,name={declared}{runtime}"

    def devices(self, suite: Suite, named: str) -> list[str]:
        """The devices a run goes to.

        named: one device, `all` for every one the app declares; the suite's when empty, else
            the app's first.
        """
        if named == "all":
            return list(self.spec.devices)
        chosen = [named] if named else suite.devices or list(self.spec.devices)[:1]
        if unknown := [device for device in chosen if device not in self.spec.devices]:
            raise MissionError(
                f"{self.name} declares no device {unknown[0]!r}; it has "
                f"{', '.join(self.spec.devices) or 'none'}"
            )
        return chosen

    def enter(self, suite: Suite) -> None:
        """Set and clear the app's and the suite's variables for everything this process starts.

        plumbum, which starts the build, copied the environment at import, so it is set too.
        """
        for name, value in (self.spec.env | suite.env).items():
            if value is False:
                os.environ.pop(name, None)
                local.env.pop(name, "")
            else:
                os.environ[name] = local.env[name] = str(value)

    def project(self, suite: Suite) -> Path:
        """The project a suite builds: its own, else the app's."""
        return self.home / (suite.project or self.spec.project)

    def settings(self, suite: Suite, device: str) -> tuple[str, ...]:
        """The app's build settings, then the device's, then the suite's."""
        return (*self.spec.settings, *self.spec.devices[device].settings, *suite.settings)

    def suite(self, name: str) -> Suite:
        """The suite `name`; an empty name is the scheme's whole test action."""
        if not name:
            return Suite()
        try:
            return self.spec.suites[name]
        except KeyError:
            raise MissionError(
                f"{self.name} declares no suite {name!r}; it has "
                f"{', '.join(sorted(self.spec.suites)) or 'none'}"
            ) from None


class Apps(FrozenModel):
    """Every app the workspace at `root` and the projects nested in it declare, by name."""

    found: dict[str, PlacedApp]

    @classmethod
    def at(cls, root: Path) -> Apps:
        """Read the apps of the workspace at `root` and of every project nested in it.

        Only each manifest's `[apps]` table is read, so a nested project whose other tables this
        release does not know still lends its apps; a name declared twice is refused.
        """
        project = Project()
        found: dict[str, PlacedApp] = {}
        for home in [root, *nested_projects(root)]:
            with project.manifest(home).open("rb") as manifest:
                declared = tomllib.load(manifest).get("apps", {})
            for name, table in declared.items():
                if name in found:
                    raise MissionError(
                        f"app {name!r} is declared by both {found[name].home} and {home}"
                    )
                found[name] = PlacedApp(home=home, name=name, spec=App.model_validate(table))
        return cls(found=found)

    def named(self, name: str) -> PlacedApp:
        """The app `name`, refused with the names this workspace has."""
        try:
            return self.found[name]
        except KeyError:
            raise MissionError(
                f"no app {name!r} in this workspace; it has "
                f"{', '.join(sorted(self.found)) or 'none'}"
            ) from None
