import abc
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from ...core.errors import MissionError
from ...manifest.schema.container import EnvMode, Guardrail

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class EnvBackend(Registry, abc.ABC):
    """A way of laying a managed environment onto a bind-mounted host prefix.

    The prefix always lives outside the image, so a backend only has to describe how to
    fill it (`provision_argv`) and how a shell already inside the container reaches it
    (`activation_snippet`). Concrete implementations enroll under this root, keyed to a
    manifest `EnvMode` through the `mode` class attribute, and are looked up by `resolve`.
    """

    mode: ClassVar[EnvMode]

    @staticmethod
    def pins_system_packages(guardrails: Sequence[Guardrail]) -> bool:
        """Whether `guardrails` asks the backend to pin the image's tuned system packages.

        A plain marker an env backend reads while assembling its own provisioning or
        activation commands. `PIN_SYSTEM_PACKAGES` never becomes a container runtime
        flag the way `UNSET_PIP_CONSTRAINT` does, since keeping a resolver from
        shadowing the image's tuned builds is entirely an env-layer concern.
        """
        return Guardrail.PIN_SYSTEM_PACKAGES in guardrails

    @classmethod
    @abc.abstractmethod
    def activation_snippet(cls, prefix: Path, *, guardrails: Sequence[Guardrail] = ()) -> str:
        """Bash that activates the environment already provisioned at `prefix`.

        prefix: the bind-mounted host path the environment was provisioned at.
        guardrails: the container's guardrails, so the snippet can apply the ones that
            only make sense at activation time (clearing an inherited `PIP_CONSTRAINT`).
        """

    @classmethod
    @abc.abstractmethod
    def provision_argv(cls, prefix: Path, *, python: str = "python3") -> list[list[str]]:
        """The commands that create the managed environment at `prefix`.

        Each inner list is one argv to run in order, from inside the container, against
        the bind-mounted host path `prefix`.

        prefix: the bind-mounted host path the environment is built at.
        python: the interpreter to provision with, when the backend invokes one directly.
        """


def resolve(mode: EnvMode) -> type[EnvBackend]:
    """The backend implementation for a manifest's `EnvMode`.

    mode: the manifest's declared environment mode.
    """
    try:
        return EnvBackend.find(mode, attr="mode")
    except LookupError:
        raise MissionError(
            f"no env backend for {mode!r}, known modes are "
            f"{[implementation.mode for implementation in EnvBackend.implementations()]}"
        ) from None
