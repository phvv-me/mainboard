import os
from contextlib import suppress
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from plumbum import local
from plumbum.commands.processes import CommandNotFound

from .....core import MissionError, Project
from .....log import logger
from ..process import Process
from ..tool import Tool

if TYPE_CHECKING:
    from plumbum.commands.base import BaseCommand

_TOOL = Project().name


class PixiEngine(Tool):
    """The pixi binary, found or installed on first use, and what it does without a workspace."""

    # THE ONE PIXI THE WHOLE FLEET RUNS. Each version rewrites the lock differently (0.77 labels
    # a named platform variant `p1`, 0.79 by the manifest's name); on 2026-09-05 that split one
    # artifact into two environment addresses across 0.77 and 0.79 hosts, killing a wave with
    # `found no built environment`. `pixi.lock.canonical` keeps such a rewrite from moving an
    # address; this pin keeps it from happening. To raise it: bump it here, re-solve, and set
    # every host up again.
    VERSION: ClassVar[str] = "0.79.0"
    # `pip install mainboard` brings no pixi, so first use runs the official installer at the pin.
    # `PIXI_NO_PATH_UPDATE`: the installer would append a line to the user's startup file, which
    # belongs to their dotfiles, and this tool finds pixi by its path.
    INSTALLER: ClassVar[str] = (
        f"curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION={VERSION} PIXI_NO_PATH_UPDATE=1 sh"
    )

    name = "pixi"

    @cached_property
    def command(self) -> BaseCommand:
        """pixi on PATH, else `PIXI_HOME/bin` (a non-login shell drops it), else bootstrapped."""
        try:
            return local["pixi"]
        except CommandNotFound:
            return local[str(self.installed_binary())]

    @staticmethod
    def binary_path() -> Path:
        """The fallback Pixi executable path."""
        return PixiEngine.home() / "bin" / "pixi"

    @staticmethod
    def home() -> Path:
        """pixi's home, where its `bin/` and global `envs/` live."""
        return Path(os.environ.get("PIXI_HOME") or Path.home() / ".pixi")

    @staticmethod
    def installer() -> BaseCommand:
        """Pixi's official installer command."""
        return local["sh"]["-c", PixiEngine.INSTALLER]

    def aligned(self) -> bool:
        """Whether this machine's pixi is the fleet's pin."""
        return self.version() == PixiEngine.VERSION

    def bootstrap(self) -> None:
        """Run pixi's installer into `PIXI_HOME/bin`."""
        logger.info(f"{_TOOL}: installing pixi engine…")
        if not Process.foreground(self.installer()):
            raise MissionError(
                "the pixi installer failed, install it manually from https://pixi.sh"
            )

    def installed_binary(self) -> Path:
        """The fallback Pixi binary, bootstrapped when absent."""
        binary = self.binary_path()
        if not binary.exists():
            self.bootstrap()
        return binary

    def version(self) -> str:
        """The pixi this machine runs as `X.Y.Z`, empty when none resolves.

        Never bootstraps: `facts`, `doctor` and host alignment ask this without changing it.
        """
        for candidate in ("pixi", str(self.binary_path())):
            with suppress(CommandNotFound, MissionError, OSError):
                return local[candidate]["--version"]().split()[-1]
        return ""
