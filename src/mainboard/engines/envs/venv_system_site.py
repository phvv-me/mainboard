from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ...manifest.schema.container import EnvMode, Guardrail
from .base import EnvBackend


class VenvSystemSite(EnvBackend):
    """A stdlib venv layered over the image's system site-packages.

    `--system-site-packages` keeps the image's tuned wheels (torch and friends) visible
    to the venv, so provisioning only ever adds packages on top instead of rebuilding them.
    """

    mode: ClassVar[EnvMode] = EnvMode.VENV_SYSTEM_SITE

    @classmethod
    def activation_snippet(cls, prefix: Path, *, guardrails: Sequence[Guardrail] = ()) -> str:
        lines = [f'source "{prefix / "bin" / "activate"}"']
        if Guardrail.UNSET_PIP_CONSTRAINT in guardrails:
            # NGC base images bake PIP_CONSTRAINT into the image `ENV`, pinning installs
            # to versions that predate whatever this venv is layering on top, so a plain
            # `pip install` inside it fails resolution unless the inherited pin is cleared.
            lines.append("unset PIP_CONSTRAINT")
        return "\n".join(lines)

    @classmethod
    def provision_argv(cls, prefix: Path, *, python: str = "python3") -> list[list[str]]:
        return [[python, "-m", "venv", "--system-site-packages", str(prefix)]]
