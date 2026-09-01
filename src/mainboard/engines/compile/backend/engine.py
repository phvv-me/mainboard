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

# mainboard's engine. `pip install mainboard` brings no `pixi` binary, so it installs one on
# first use with the official installer for the current operating system.
_POSIX_INSTALLER = "curl -fsSL https://pixi.sh/install.sh | sh"
_WINDOWS_INSTALLER = "irm -useb https://pixi.sh/install.ps1 | iex"

# The tool announcing the install, so nothing here spells its name.
_TOOL = Project().name

# The startup file pixi's installer appends its own PATH line to, chosen from the basename of
# `$SHELL` exactly as the installer's own case statement does. A shell it has no rule for is
# absent here, which is the same silence the installer keeps.
_SHELL_RC = {
    "bash": "~/.bashrc",
    "fish": "~/.config/fish/config.fish",
    "tcsh": "~/.tcshrc",
    "zsh": "~/.zshrc",
}


class PixiEngine(Tool):
    """Where the pixi binary is, and what it can do without a workspace to point at.

    Finding it (and installing it the first time) is the one job every workspace-scoped `Pixi`
    shares, so it lives here alone.
    """

    name = "pixi"

    @cached_property
    def command(self) -> BaseCommand:
        """The pixi executable.

        Prefer it on PATH, fall back to `PIXI_HOME/bin` when a non-login remote shell has
        dropped it, and bootstrap the engine when it is absent everywhere. Windows tools in a
        Pixi exec environment still follow the cross-platform ``HOME`` convention, so bind its
        actual user-profile path when the host did not provide one.
        """
        try:
            command = local["pixi"]
        except CommandNotFound:
            command = local[str(self.installed_binary())]
        if platform.system() == "Windows" and "HOME" not in os.environ:
            return command.with_env(HOME=str(Path.home()))
        return command

    @staticmethod
    def appended_shell_file() -> str:
        """The startup file pixi's installer is about to append a PATH line to, else empty.

        Empty in the two cases where nothing is touched, `PIXI_NO_PATH_UPDATE` suppressing the
        edit and a `$SHELL` the installer has no rule for.
        """
        if os.environ.get("PIXI_NO_PATH_UPDATE") or platform.system() == "Windows":
            return ""
        return _SHELL_RC.get(PurePath(os.environ.get("SHELL", "")).name, "")

    @staticmethod
    def home() -> Path:
        """pixi's home, where its `bin/` and global `envs/` live."""
        return Path(os.environ.get("PIXI_HOME") or Path.home() / ".pixi")

    def bootstrap(self) -> None:
        """Install pixi (the engine) when it is missing, so `pip install mainboard` is enough.

        Runs pixi's official installer, which places the binary in `PIXI_HOME/bin` and, unless
        `PIXI_NO_PATH_UPDATE` says otherwise, appends a PATH line to the startup file of
        whatever `$SHELL` names. Editing a personal file is not something a first use should do
        without saying so, so that file is named here before the installer runs.
        """
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
                _WINDOWS_INSTALLER,
            ]
        executable = shutil.which("sh")
        if executable is None:
            raise MissionError("a POSIX shell is required to install pixi on this platform")
        return local[executable]["-c", _POSIX_INSTALLER]

    def installed_binary(self) -> Path:
        """Return the fallback Pixi binary after bootstrapping it when absent."""
        binary = self.binary_path()
        if not binary.exists():
            self.bootstrap()
        return binary

    @staticmethod
    def binary_path() -> Path:
        """The fallback Pixi executable path for this operating system."""
        name = "pixi.exe" if platform.system() == "Windows" else "pixi"
        return PixiEngine.home() / "bin" / name
