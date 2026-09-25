from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ...manifest.schema.container import EnvMode, Guardrail
from .base import EnvBackend


class PixiPrefix(EnvBackend):
    """A pixi environment installed at a detached prefix instead of pixi's own `.pixi/envs`.

    The caller redirects pixi at `prefix` before running `provision_argv`, through the
    `detached-environments` config (or `PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR`), and sets
    `PIXI_CACHE_DIR` for the package cache shared across prefixes.
    """

    mode: ClassVar[EnvMode] = EnvMode.PIXI_PREFIX

    @classmethod
    def activation_snippet(cls, prefix: Path, *, guardrails: Sequence[Guardrail] = ()) -> str:
        """Source pixi's shell-hook when present, else prepend `bin` to `PATH`.

        `guardrails` is unused: a pixi environment never inherits the image's `PIP_CONSTRAINT`.
        """
        hook = prefix / "activate.sh"
        return "\n".join(
            [
                f'if [ -f "{hook}" ]; then',
                f'    source "{hook}"',
                "else",
                f'    export PATH="{prefix / "bin"}:$PATH"',
                "fi",
            ]
        )

    @classmethod
    def provision_argv(cls, prefix: Path, *, python: str = "python3") -> list[list[str]]:
        """The `pixi install` argv, `python` unused since pixi pins its own interpreter."""
        return [["pixi", "install", "--locked"]]
