import os
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
# first use with the official script that drops `pixi` into `PIXI_HOME/bin`.
_PIXI_INSTALLER = "curl -fsSL https://pixi.sh/install.sh | sh"

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
        dropped it, and bootstrap the engine when it is absent everywhere.
        """
        try:
            return local["pixi"]
        except CommandNotFound:
            return local[str(self.installed_binary())]

    @staticmethod
    def appended_shell_file() -> str:
        """The startup file pixi's installer is about to append a PATH line to, else empty.

        Empty in the two cases where nothing is touched, `PIXI_NO_PATH_UPDATE` suppressing the
        edit and a `$SHELL` the installer has no rule for.
        """
        if os.environ.get("PIXI_NO_PATH_UPDATE"):
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
        if not Process.foreground(local["sh"]["-c", _PIXI_INSTALLER]):
            raise MissionError(
                "the pixi installer failed, install it manually from https://pixi.sh"
            )

    def installed_binary(self) -> Path:
        """Return the fallback Pixi binary after bootstrapping it when absent."""
        binary = self.home() / "bin" / "pixi"
        if not binary.exists():
            self.bootstrap()
        return binary
