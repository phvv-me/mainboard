"""Run Xcode on a workspace, project or Swift package, keeping native output and exit status."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlsplit

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, Field, ValidationError, model_validator

from ..core.errors import MissionError
from ..proc import Processes
from .xcrun import simulators

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..core.execution.tool import Runner

# The project files xcodebuild names by flag; a Swift package is its own directory.
_PROJECTS = {".xcworkspace": "-workspace", ".xcodeproj": "-project"}
_SIMULATOR = "iOS Simulator"


@Parameter(name="*")
class XcodeRun(FrozenModel):
    """One xcodebuild action: the scheme, where it runs, what it selects, how long it may take."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    scheme: str
    """the shared Xcode scheme to build/test."""
    action: Literal["build", "test", "build-for-testing", "resolve"] = "build"
    """build, test, build-for-testing or resolve."""
    destination: str = ""
    """native Xcode destination, including exact simulator runtime."""
    configuration: str = "Debug"
    """build configuration, normally Debug or Release."""
    derived_data: Path | None = None
    """reusable generated build directory."""
    only_testing: tuple[str, ...] = ()
    """repeatable native test identifiers to include."""
    skip_testing: tuple[str, ...] = ()
    """repeatable native test identifiers to exclude."""
    setting: tuple[str, ...] = ()
    """repeatable literal KEY=VALUE Xcode build settings."""
    timeout: float = 900
    """hard limit in seconds, including descendant processes."""
    url_test: Path | None = None
    """JSON runner bundle, relative marker and explicit custom-scheme routes."""

    @model_validator(mode="after")
    def _runnable(self) -> Self:
        """Refuse a run with no scheme or no time to run in."""
        if self.timeout <= 0 or not self.scheme.strip():
            raise MissionError("Apple execution needs a scheme and a positive timeout")
        if self.url_test is not None and self.action != "test":
            raise MissionError("URL delivery requires the test action")
        return self

    @property
    def resolving(self) -> bool:
        """Whether this run only resolves packages, leaving no result bundle."""
        return self.action == "resolve"

    def _built(self, bundle: Path | None) -> list[str]:
        """The flags of a build or test, refusing a result bundle that is missing or exists."""
        if bundle is None:
            raise MissionError("Apple builds/tests need a fresh result bundle path")
        bundle = bundle.expanduser().resolve()
        if bundle.exists():
            raise MissionError(f"refusing to overwrite result evidence at {bundle}")
        bundle.parent.mkdir(parents=True, exist_ok=True)
        flags = ["-resultBundlePath", str(bundle), "-configuration", self.configuration]
        flags += ["-destination", self.destination] if self.destination else []
        return flags + (["-parallel-testing-enabled", "NO"] if self.action == "test" else [])

    def command(self, source: Sequence[str], bundle: Path | None) -> list[str]:
        """The xcodebuild command line for `source`, writing results into a fresh `bundle`."""
        command = [
            "/usr/bin/xcrun",
            "xcodebuild",
            "-resolvePackageDependencies" if self.resolving else self.action,
            *source,
            "-scheme",
            self.scheme,
            "-onlyUsePackageVersionsFromResolvedFile",
            "-hideShellScriptEnvironment",
            *([] if self.resolving else self._built(bundle)),
        ]
        if self.derived_data is not None:
            command += ["-derivedDataPath", str(self.derived_data.expanduser().resolve())]
        command += [f"-only-testing:{value}" for value in self.only_testing]
        command += [f"-skip-testing:{value}" for value in self.skip_testing]
        return command + list(self.setting)


class URLTestConfiguration(FrozenModel):
    """Routes delivered by a host while an isolated simulator UI test runs."""

    runner_bundle: str = Field(min_length=1)
    marker: str = Field(min_length=1)
    routes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _contained(self) -> Self:
        """Refuse a marker outside the runner's container and routes that are not local."""
        marker = Path(self.marker)
        if marker.is_absolute() or ".." in marker.parts:
            raise MissionError("URL test marker must stay inside the runner data container")
        if len(set(self.routes)) != len(self.routes):
            raise MissionError("URL test routes must be unique")
        if not all(URLTestConfiguration._local(route) for route in self.routes):
            raise MissionError("URL tests require explicit local custom-scheme paths")
        return self

    @staticmethod
    def _local(route: str) -> bool:
        """Whether `route` is an explicit local custom-scheme path."""
        value = urlsplit(route)
        return (
            bool(value.scheme)
            and value.path.startswith("/")
            and not (value.netloc or value.query or value.fragment)
        )


class Xcodebuild:
    """xcodebuild under the selected Xcode, the whole process tree stopped at its deadline.

    Committed package versions are used and Xcode's automatic script environment dumps are
    suppressed; result bundles are always fresh, never overwritten.
    """

    @staticmethod
    def _located(project: Path) -> tuple[tuple[str, ...], Path]:
        """The flags naming `project` to xcodebuild, and the directory it runs from."""
        if not project.is_dir():
            raise MissionError(f"no Apple project at {project}")
        if flag := _PROJECTS.get(project.suffix):
            return (flag, str(project)), project.parent
        if (project / "Package.swift").is_file():
            return (), project
        raise MissionError("expected an .xcworkspace, .xcodeproj or Swift package directory")

    def __init__(self, runner: Runner) -> None:
        """runner: what the available simulators are read through, for a URL test only."""
        self.runner = runner

    def _delivered(self, run: XcodeRun, command: list[str], cwd: Path) -> int:
        """`command` run by the URL-delivering Swift helper beside the simulator it targets."""
        path = run.url_test or Path()
        try:
            config = URLTestConfiguration.model_validate_json(path.read_text())
        except (OSError, ValidationError) as fault:
            raise MissionError(f"invalid URL test configuration: {fault}") from fault
        request = {
            **config.model_dump(mode="json"),
            "device": self._device(run.destination),
            "command": command,
        }
        with TemporaryDirectory(prefix="mainboard-apple-url-") as temporary:
            descriptor = Path(temporary) / "request.json"
            descriptor.write_text(json.dumps(request))
            helper = Path(__file__).with_name("url_tests.swift")
            argv = ["/usr/bin/xcrun", "swift", str(helper), str(descriptor)]
            return Processes().timeout(run.timeout, argv, cwd=cwd)

    def _device(self, destination: str) -> str:
        """The one available simulator a URL test's `destination` names."""
        fields = dict(part.split("=", 1) for part in destination.split(",") if "=" in part)
        if fields.get("platform") != _SIMULATOR:
            raise MissionError("URL delivery is only supported on an iOS simulator")
        runtime = ".iOS-" + fields.get("OS", "").replace(".", "-")
        candidates = [
            device["udid"]
            for device in simulators(self.runner)
            if (not fields.get("id") or device["udid"] == fields["id"])
            and (not fields.get("name") or device["name"] == fields["name"])
            and (not fields.get("OS") or device["runtime"].endswith(runtime))
        ]
        if len(candidates) != 1:
            raise MissionError("URL tests require one exact simulator destination")
        return candidates[0]

    def run(self, project: Path, run: XcodeRun, bundle: Path | None) -> int:
        """Build or test `project` as `run` asks, its results in `bundle`; the exit status."""
        source, cwd = Xcodebuild._located(project.expanduser().resolve())
        command = run.command(source, bundle)
        if run.url_test is None:
            return Processes().timeout(run.timeout, command, cwd=cwd)
        return self._delivered(run, command, cwd)
