"""`[apps.<name>]`: an application a manifest declares, which `mb app` builds and tests from any
directory of the workspace, whichever project's manifest declares it."""

from typing import Literal

from pydantic import model_validator

from ...core.base import Declared
from .spec import Json

type Action = Literal["build", "test", "build-for-testing", "resolve"]


class Device(Declared):
    """Where an app runs.

    destination: an xcodebuild destination (`platform=macOS`), or a simulator's name, which runs
        the app's `runtime`.
    derived_data: the build products this device keeps, relative to the app's manifest;
        `.build/Xcode/<device>` when empty.
    settings: build settings its builds add (`CODE_SIGN_IDENTITY=-` for a Mac).
    """

    destination: str
    derived_data: str = ""
    settings: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def from_bare_string(cls, value: Json) -> Json:
        """Accept `iphone = "iPhone 18 Pro"` as shorthand for `{ destination = ... }`."""
        return {"destination": value} if isinstance(value, str) else value


class Suite(Declared):
    """A named build or selection of tests, `mb app test <app> <suite>`.

    project: the app's own when empty, else another project or package beside it.
    scheme: the app's own when empty.
    only, skip: native test identifiers (`Target/Class/testName`) to run or leave out.
    settings: build settings added to the app's.
    devices: those it runs on when the command names none, the app's first when empty.
    derived_data: the device's own when empty, else this one (relative to the manifest), for a
        suite building another project.
    env: variables set (a string) or cleared (`false`) on top of the app's.
    url_test: a JSON runner bundle for custom-scheme route tests, relative to the manifest.
    """

    project: str = ""
    scheme: str = ""
    action: Action = "test"
    configuration: str = "Debug"
    only: list[str] = []
    skip: list[str] = []
    settings: list[str] = []
    devices: list[str] = []
    derived_data: str = ""
    env: dict[str, str | bool] = {}
    timeout: float = 900
    url_test: str = ""


class App(Declared):
    """An application and the ways it is built and tested.

    project: an .xcworkspace, .xcodeproj or Swift package directory, relative to the manifest.
    scheme: the shared scheme builds and suites use unless they name their own.
    runtime: the simulator OS a device named by its simulator runs (`27.0`); xcodebuild's
        newest when empty.
    settings: build settings every build and suite passes (`CODE_SIGNING_ALLOWED=NO`).
    env: variables every build and suite sets (a string) or clears (`false`), as `[env]`.
    """

    project: str
    scheme: str
    runtime: str = ""
    settings: list[str] = []
    env: dict[str, str | bool] = {}
    devices: dict[str, Device] = {}
    suites: dict[str, Suite] = {}
