# Where a harness's binary comes from, and how it is brought to its latest release.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

type Which = Callable[[str], str | None]


class Channel(ABC):
    """Where a harness's binary comes from, and how it is brought to its latest release."""

    @abstractmethod
    def install(self, home: Path, which: Which) -> list[str]: ...

    def update(self, home: Path, which: Which) -> list[str]:
        return self.install(home, which)


class Script(Channel):
    """The vendor's own installer, and the harness's own command to update itself."""

    def __init__(self, *, script: str, update: Sequence[str]) -> None:
        self.script = script
        self.updating = update

    def install(self, home: Path, which: Which) -> list[str]:
        return ["bash", "-c", self.script]

    def update(self, home: Path, which: Which) -> list[str]:
        return [*self.updating]


class PixiGlobal(Channel):
    """A conda-forge package `pixi global` installs and exposes on PATH."""

    def __init__(self, package: str) -> None:
        self.package = package

    def install(self, home: Path, which: Which) -> list[str]:
        return [_found("pixi", which), "global", "install", self.package]

    def update(self, home: Path, which: Which) -> list[str]:
        return [_found("pixi", which), "global", "update", self.package]


class Npm(Channel):
    """An npm package installed under `~/.local`, whose `bin` the vendor installers share.

    scripts: the packages whose install scripts must run, which npm otherwise skips (its
        native terminal and keychain bindings, say).
    """

    def __init__(self, package: str, *, scripts: Sequence[str] = ()) -> None:
        self.package = package
        self.scripts = scripts

    def install(self, home: Path, which: Which) -> list[str]:
        allowed = [f"--allow-scripts={','.join(self.scripts)}"] if self.scripts else []
        return [
            _found("npm", which),
            "install",
            "--global",
            "--prefix",
            str(home / ".local"),
            *allowed,
            f"{self.package}@latest",
        ]


def _found(program: str, which: Which) -> str:
    """`program`'s path, refusing when this machine does not have it."""
    path = which(program)
    if path is None:
        raise MissionError(f"{program} is not on PATH here")
    return path
