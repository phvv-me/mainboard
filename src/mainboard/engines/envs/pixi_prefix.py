from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ...manifest.schema.container import EnvMode, Guardrail
from .base import EnvBackend


class PixiPrefix(EnvBackend):
    """A pixi environment installed at a detached prefix instead of pixi's own `.pixi/envs`.

    Provisioning still happens through the pixi engine, so `provision_argv` is just the
    `pixi install` invocation. Redirecting pixi at `prefix` is the caller's job before
    running it, pointing pixi's `detached-environments` config (or its
    `PIXI_CACHE_DETACHED_ENVIRONMENTS_DIR` escape hatch) at `prefix` and setting
    `PIXI_CACHE_DIR` for the shared package cache that lives outside any one prefix.
    """

    mode: ClassVar[EnvMode] = EnvMode.PIXI_PREFIX

    @classmethod
    def activation_snippet(cls, prefix: Path, *, guardrails: Sequence[Guardrail] = ()) -> str:
        """Source pixi's own shell-hook when present, else fall back to a bare `PATH` prepend.

        `guardrails` is accepted for interface parity with `VenvSystemSite` but unused,
        since a pixi-managed environment never inherits the image's `PIP_CONSTRAINT` the
        way a `--system-site-packages` venv does, so there is nothing here to unset.
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
