import shutil
from collections.abc import Sequence
from typing import ClassVar

from ...manifest.schema.container import Container
from .base import ContainerRuntime


class Apptainer(ContainerRuntime):
    """Wraps argv for `apptainer exec`, falling back to its command-compatible `singularity`."""

    binary: ClassVar[str] = "apptainer"
    legacy_binary: ClassVar[str] = "singularity"

    @classmethod
    def command(cls, container: Container, *, prefix_bind: str, argv: Sequence[str]) -> list[str]:
        binds = [*container.binds, prefix_bind]
        return [
            cls.launcher(),
            "exec",
            *(["--nv"] if container.gpus else []),
            *[flag for bind in binds for flag in ("--bind", bind)],
            *(["--pwd", container.workdir] if container.workdir else []),
            *cls.env_flags(container.passthrough),
            container.image,
            *cls.guarded_argv(container, argv),
        ]

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which(cls.binary) is not None or shutil.which(cls.legacy_binary) is not None

    @classmethod
    def launcher(cls) -> str:
        """The binary this host exposes, `apptainer` preferred."""
        if shutil.which(cls.binary) or not shutil.which(cls.legacy_binary):
            return cls.binary
        return cls.legacy_binary
