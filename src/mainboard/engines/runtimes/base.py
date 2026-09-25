import abc
import shutil
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from ...core.errors import MissionError
from ...manifest.schema.container import Container, Guardrail

if TYPE_CHECKING:
    from collections.abc import Sequence


class ContainerRuntime(Registry, abc.ABC):
    """A container engine that wraps a task's argv to run inside a base image.

    The environment prefix lives on a bound host path, never in the image, so a runtime only
    builds the launcher argv. Implementations enroll here and are looked up by `resolve`.
    """

    binary: ClassVar[str] = ""

    @classmethod
    @abc.abstractmethod
    def command(cls, container: Container, *, prefix_bind: str, argv: Sequence[str]) -> list[str]:
        """The full launcher argv that runs `argv`, guardrail-wrapped, inside `container`.

        prefix_bind: the `source:target` bind carrying the environment prefix, appended after
            `container.binds`.
        """

    @classmethod
    def env_flags(cls, passthrough: Sequence[str]) -> list[str]:
        """`--env` flags carrying each named host variable into the container."""
        return [flag for variable in passthrough for flag in ("--env", variable)]

    @classmethod
    def guarded_argv(cls, container: Container, argv: Sequence[str]) -> Sequence[str]:
        """`argv`, wrapped in `env -u PIP_CONSTRAINT` when `container` guards against it.

        No runtime has a flag that unsets a variable baked into the image's `ENV`, which is how
        the NGC images ship `PIP_CONSTRAINT`, hence the plain `env -u` wrap.
        """
        if Guardrail.UNSET_PIP_CONSTRAINT not in container.guardrails:
            return argv
        return ["env", "-u", "PIP_CONSTRAINT", *argv]

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which(cls.binary) is not None


def resolve(runtime: str) -> type[ContainerRuntime]:
    """The runtime a manifest's `runtime` key names, `auto` taking the first available here."""
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
