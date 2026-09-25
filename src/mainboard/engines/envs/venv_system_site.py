from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ...manifest.schema.container import EnvMode, Guardrail
from .base import EnvBackend


class VenvSystemSite(EnvBackend):
    """A stdlib venv seeing the image's tuned wheels (torch and friends), only adding on top."""

    mode: ClassVar[EnvMode] = EnvMode.VENV_SYSTEM_SITE

    @classmethod
    def activation_snippet(cls, prefix: Path, *, guardrails: Sequence[Guardrail] = ()) -> str:
        lines = [f'source "{prefix / "bin" / "activate"}"']
        if Guardrail.UNSET_PIP_CONSTRAINT in guardrails:
            # NGC images bake an old PIP_CONSTRAINT into their `ENV`, which fails any newer pin.
            lines.append("unset PIP_CONSTRAINT")
        return "\n".join(lines)

    @classmethod
    def provision_argv(cls, prefix: Path, *, python: str = "python3") -> list[list[str]]:
        return [[python, "-m", "venv", "--system-site-packages", str(prefix)]]
