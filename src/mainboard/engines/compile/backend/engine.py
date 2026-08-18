import os
import sys
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from plumbum import local
from plumbum.commands.processes import CommandNotFound

from ....core import MissionError
from .process import Process
from .tool import Tool

if TYPE_CHECKING:
    from plumbum.commands.base import BaseCommand

# mainboard's engine. `pip install mainboard` brings no `pixi` binary, so it installs one on
# first use with the official script that drops `pixi` into `PIXI_HOME/bin`.
_PIXI_INSTALLER = "curl -fsSL https://pixi.sh/install.sh | sh"


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
    def home() -> Path:
        """pixi's home, where its `bin/` and global `envs/` live."""
        return Path(os.environ.get("PIXI_HOME") or Path.home() / ".pixi")

    def bootstrap(self) -> None:
        """Install pixi (the engine) when it is missing, so `pip install mainboard` is enough.

        Runs pixi's official installer, which places the binary in `PIXI_HOME/bin`.
        """
        sys.stderr.write("mainboard: installing pixi engine…\n")
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
