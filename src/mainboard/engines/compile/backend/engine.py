import os
import platform
import shutil
import sys
from functools import cached_property
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from plumbum import local
from plumbum.commands.processes import CommandNotFound

from ....core import MissionError, Project
from .process import Process
from .tool import Tool

if TYPE_CHECKING:
    from plumbum.commands.base import BaseCommand

# THE ONE PIXI THE WHOLE FLEET RUNS. Each version rewrites the lock differently (0.77 labels a
# named platform variant `p1`, 0.79 by the manifest's name); on 2026-09-05 that split one artifact
# into two environment addresses across 0.77 and 0.79 hosts, killing a wave with `found no built
# environment`. `pixi_lock.canonical` keeps such a rewrite from moving an address; this pin keeps
# it from happening. To raise it: bump it here, re-solve, and set every host up again.
PIXI_VERSION = "0.79.0"

# `pip install mainboard` brings no pixi, so first use runs the official installer at the pin.
POSIX_INSTALLER = f"curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION={PIXI_VERSION} sh"
WINDOWS_INSTALLER = (
    f"$Env:PIXI_VERSION='{PIXI_VERSION}'; irm -useb https://pixi.sh/install.ps1 | iex"
)

_TOOL = Project().name

# The startup file pixi's installer appends a PATH line to, by `$SHELL` basename, mirroring the
# installer's own case statement (an unlisted shell is left alone there too).
_SHELL_RC = {
    "bash": "~/.bashrc",
    "fish": "~/.config/fish/config.fish",
    "tcsh": "~/.tcshrc",
    "zsh": "~/.zshrc",
}


class PixiEngine(Tool):
    """The pixi binary, found or installed on first use, and what it does without a workspace."""

    name = "pixi"

    @cached_property
    def command(self) -> BaseCommand:
        """pixi on PATH, else `PIXI_HOME/bin` (a non-login shell drops it), else bootstrapped.

        On Windows `HOME` is bound to the real user profile, since tools in a pixi exec
        environment follow it and a launcher may pass another.
        """
        try:
            command = local["pixi"]
        except CommandNotFound:
            command = local[str(self.installed_binary())]
        if platform.system() == "Windows":
            return command.with_env(HOME=str(Path.home()))
        return command

    def version(self) -> str:
        """The pixi this machine runs as `X.Y.Z`, empty when none resolves.

        Never bootstraps: `facts`, `doctor` and host alignment ask this without changing it.
        """
        for candidate in ("pixi", str(self.binary_path())):
            try:
                return str(local[candidate]["--version"]()).split()[-1]
            except CommandNotFound, MissionError, OSError:
                continue
        return ""

    def aligned(self) -> bool:
        """Whether this machine's pixi is the fleet's pin."""
        return self.version() == PIXI_VERSION

    @staticmethod
    def appended_shell_file() -> str:
        """The startup file pixi's installer will append a PATH line to, else empty."""
        if os.environ.get("PIXI_NO_PATH_UPDATE") or platform.system() == "Windows":
            return ""
        return _SHELL_RC.get(PurePath(os.environ.get("SHELL", "")).name, "")

    @staticmethod
    def home() -> Path:
        """pixi's home, where its `bin/` and global `envs/` live."""
        return Path(os.environ.get("PIXI_HOME") or Path.home() / ".pixi")

    def bootstrap(self) -> None:
        """Run pixi's installer into `PIXI_HOME/bin`, first naming the rc file it will edit."""
        sys.stderr.write(f"{_TOOL}: installing pixi engine…\n")
        if appended := self.appended_shell_file():
            sys.stderr.write(f"{_TOOL}: the pixi installer adds a PATH line to {appended}\n")
        if not Process.foreground(self.installer()):
            raise MissionError(
                "the pixi installer failed, install it manually from https://pixi.sh"
            )

    @staticmethod
    def installer() -> BaseCommand:
        """Pixi's official installer command for this operating system."""
        if platform.system() == "Windows":
            executable = shutil.which("powershell") or shutil.which("pwsh")
            if executable is None:
                raise MissionError("PowerShell is required to install pixi on Windows")
            return local[executable][
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                WINDOWS_INSTALLER,
            ]
        executable = shutil.which("sh")
        if executable is None:
            raise MissionError("a POSIX shell is required to install pixi on this platform")
        return local[executable]["-c", POSIX_INSTALLER]

    def installed_binary(self) -> Path:
        """The fallback Pixi binary, bootstrapped when absent."""
        binary = self.binary_path()
        if not binary.exists():
            self.bootstrap()
        return binary

    @staticmethod
    def binary_path() -> Path:
        """The fallback Pixi executable path for this operating system."""
        name = "pixi.exe" if platform.system() == "Windows" else "pixi"
        return PixiEngine.home() / "bin" / name
