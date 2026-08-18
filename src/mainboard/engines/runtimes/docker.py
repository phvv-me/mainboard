import abc
from collections.abc import Sequence
from typing import ClassVar

from ...manifest.schema.container import Container
from .base import ContainerRuntime


class DockerCompatible(ContainerRuntime):
    """Shared `run --rm` argv assembly for the docker-CLI-compatible runtimes.

    Docker and Podman agree on almost every flag, and the one place they diverge is how
    a GPU gets exposed, so that single decision is left to `gpu_flags` while everything
    else (binds, workdir, env passthrough, guardrails) is built once here.
    """

    @classmethod
    def command(cls, container: Container, *, prefix_bind: str, argv: Sequence[str]) -> list[str]:
        binds = [*container.binds, prefix_bind]
        return [
            cls.binary,
            "run",
            "--rm",
            *(cls.gpu_flags() if container.gpus else []),
            *[flag for bind in binds for flag in ("-v", bind)],
            *(["-w", container.workdir] if container.workdir else []),
            *cls.env_flags(container.passthrough),
            container.image,
            *cls.guarded_argv(container, argv),
        ]

    @classmethod
    @abc.abstractmethod
    def gpu_flags(cls) -> list[str]:
        """The flags that expose every GPU to the container."""


class Docker(DockerCompatible):
    """Wraps argv for `docker run`."""

    binary: ClassVar[str] = "docker"

    @classmethod
    def gpu_flags(cls) -> list[str]:
        return ["--gpus", "all"]
