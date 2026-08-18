import abc
import shutil
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from ...core.errors import MissionError
from ...manifest.schema.container import Container, Guardrail

if TYPE_CHECKING:
    from collections.abc import Sequence


class ContainerRuntime(Registry, abc.ABC):
    """A container engine that wraps a task's argv to run inside a bound base image.

    The environment prefix always lives on a bound host path, never in the image, so a
    runtime's whole job is building the launcher argv around the caller's command, from
    the exec verb and gpu passthrough through binds, guardrail handling, and the image
    reference. Concrete implementations enroll under this root and are looked up by
    `resolve`.
    """

    binary: ClassVar[str] = ""

    @classmethod
    @abc.abstractmethod
    def command(cls, container: Container, *, prefix_bind: str, argv: Sequence[str]) -> list[str]:
        """The full launcher argv that runs `argv` inside `container`.

        container: the base image plus its binds, gpu flag, guardrails and passthrough vars.
        prefix_bind: an extra `source:target` bind carrying the environment prefix, appended
            after `container.binds`.
        argv: the command to run inside the container, guardrail-wrapped as needed.
        """

    @classmethod
    def env_flags(cls, passthrough: Sequence[str]) -> list[str]:
        """`--env` flags carrying each of `passthrough`'s host variables into the container.

        passthrough: host variable names the container should see, from `Container.passthrough`.
        """
        return [flag for variable in passthrough for flag in ("--env", variable)]

    @classmethod
    def guarded_argv(cls, container: Container, argv: Sequence[str]) -> Sequence[str]:
        """`argv`, wrapped in `env -u PIP_CONSTRAINT` when `container` guards against it.

        Apptainer, Docker and Podman each expose flags that add a variable or block a host
        one from being imported, but none of them can unset a variable already baked into
        the image through its Dockerfile `ENV`, which is how the NGC base images ship
        `PIP_CONSTRAINT`. So `UNSET_PIP_CONSTRAINT` is enforced by wrapping the exec'd
        command in a plain `env -u` instead of passing a runtime-specific flag.

        container: carries the guardrails that decide whether the wrap applies.
        argv: the command that will run inside the container.
        """
        if Guardrail.UNSET_PIP_CONSTRAINT not in container.guardrails:
            return argv
        return ["env", "-u", "PIP_CONSTRAINT", *argv]

    @classmethod
    def is_available(cls) -> bool:
        """Whether `binary` is on PATH."""
        return shutil.which(cls.binary) is not None


def resolve(runtime: str) -> type[ContainerRuntime]:
    """The runtime implementation for a manifest's `runtime` key.

    `auto` picks the first runtime available on this host, and any other value names one
    directly. Either miss raises a `MissionError` listing the declared runtime names.

    runtime: `auto`, or an explicit runtime name (`apptainer`, `docker`, `podman`).
    """
    try:
        return (
            ContainerRuntime.first_available()
            if runtime == "auto"
            else ContainerRuntime.find(runtime)
        )
    except LookupError:
        raise MissionError(
            f"no container runtime available for {runtime!r}, "
            f"known runtimes are {ContainerRuntime.names()}"
        ) from None
