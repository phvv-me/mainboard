import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum import local

from .repair import EnvironmentAudit

if TYPE_CHECKING:
    from collections.abc import Generator

# What pixi writes into a prefix's `conda-meta/` once an installation has finished.
_FINGERPRINT = ".pixi-environment-fingerprint"


class PixiPrefix(FrozenModel):
    """The directory pixi installs one environment into, installed or not yet.

    path: the prefix, `<generated>/.pixi/envs/<env>`.
    """

    path: Path

    @contextmanager
    def activated(self) -> Generator[None]:
        """Prepend the prefix's `bin`, once installed, to PATH for the block."""
        binaries = self.path / "bin"
        leading = [str(binaries)] if binaries.is_dir() else []
        with local.env(PATH=os.pathsep.join([*leading, local.env["PATH"]])):
            yield

    @property
    def audit(self) -> EnvironmentAudit:
        """What the installed prefix holds that is incomplete or can no longer import."""
        return EnvironmentAudit(self.path)

    @property
    def ready(self) -> bool:
        """Whether pixi finished installing: its fingerprint, not a mere prefix, exists."""
        return (self.path / "conda-meta" / _FINGERPRINT).is_file()
