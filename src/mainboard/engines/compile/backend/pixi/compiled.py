import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from .....core.execution.host import current_platform

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ...inputs.toml import Toml

# The env var vouching each virtual-package floor, so a machine that cannot present the package
# (a login node with no GPU driver) still installs the frozen lock its jobs run under.
_FLOOR_OVERRIDES = {
    "archspec": "CONDA_OVERRIDE_ARCHSPEC",
    "cuda": "CONDA_OVERRIDE_CUDA",
    "glibc": "CONDA_OVERRIDE_GLIBC",
    "linux": "CONDA_OVERRIDE_LINUX",
    "macos": "CONDA_OVERRIDE_OSX",
    "osx": "CONDA_OVERRIDE_OSX",
}


class CompiledManifest(FrozenModel):
    """What the generated `pixi.toml` declares, read from the file on every access.

    The compiler rewrites floors mid-run, so nothing is kept. A manifest not compiled yet
    declares nothing.

    path: the `pixi.toml`.
    """

    path: Path

    @property
    def editable(self) -> bool:
        """Whether the manifest carries a mutable editable Python source."""
        return any(
            isinstance(table.get("path"), str) and table.get("editable") is True
            for table in self._nested_tables(tomllib.loads(self._text()))
        )

    @property
    def overrides(self) -> dict[str, str]:
        """The `CONDA_OVERRIDE_*` values for the floors in its platform descriptors.

        An override the process already exports is left to stand, so the caller has the last word.
        """
        return {
            _FLOOR_OVERRIDES[key]: str(value)
            for entry in self._platforms()
            if isinstance(entry, dict)
            for key, value in entry.items()
            if key in _FLOOR_OVERRIDES and _FLOOR_OVERRIDES[key] not in os.environ
        }

    @property
    def runs_here(self) -> bool:
        """Whether it declares this platform, or none, or is not compiled."""
        names = {
            entry if isinstance(entry, str) else str(entry.get("platform", ""))
            for entry in self._platforms()
        }
        return not names or current_platform() in names

    @staticmethod
    def _nested_tables(document: Toml) -> Iterator[Mapping[str, Toml]]:
        """Every table in `document`, `document` itself included, at any depth."""
        pending = [document]
        while pending:
            node = pending.pop()
            if isinstance(node, Mapping):
                yield node
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)

    def _platforms(self) -> list[str | dict[str, Toml]]:
        return tomllib.loads(self._text()).get("workspace", {}).get("platforms", [])

    def _text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
