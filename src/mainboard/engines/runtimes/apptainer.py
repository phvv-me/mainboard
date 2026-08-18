import shutil
from collections.abc import Sequence
from typing import ClassVar

from ...manifest.schema.container import Container
from .base import ContainerRuntime


class Apptainer(ContainerRuntime):
    """Wraps argv for `apptainer exec`, whose binary is sometimes still named `singularity`.

    Apptainer is the maintained successor of Singularity and stays command-line compatible
    with it, so a host that only ships the legacy `singularity` binary is still usable.
    """

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
        """Whether either `apptainer` or its `singularity` alias is on PATH."""
        return shutil.which(cls.binary) is not None or shutil.which(cls.legacy_binary) is not None

    @classmethod
    def launcher(cls) -> str:
        """The binary this host actually exposes, `apptainer` preferred over `singularity`."""
        if shutil.which(cls.binary):
            return cls.binary
        if shutil.which(cls.legacy_binary):
            return cls.legacy_binary
        return cls.binary
