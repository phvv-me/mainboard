import abc
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from ...core.errors import MissionError
from ...manifest.schema.container import EnvMode, Guardrail

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class EnvBackend(Registry, abc.ABC):
    """A way of laying a managed environment onto a bind-mounted host prefix outside the image.

    Implementations enroll here keyed by their manifest `mode` and are looked up by `resolve`.
    """

    mode: ClassVar[EnvMode]

    @staticmethod
    def pins_system_packages(guardrails: Sequence[Guardrail]) -> bool:
        """Whether to pin the image's tuned system packages.

        Unlike `UNSET_PIP_CONSTRAINT` this never becomes a runtime flag: keeping a resolver from
        shadowing the image's tuned builds is an env-layer concern only.
        """
        return Guardrail.PIN_SYSTEM_PACKAGES in guardrails

    @classmethod
    @abc.abstractmethod
    def activation_snippet(cls, prefix: Path, *, guardrails: Sequence[Guardrail] = ()) -> str:
        """Bash, run inside the container, that activates the environment provisioned at `prefix`.

        guardrails: the container's, for those that apply at activation (an inherited
            `PIP_CONSTRAINT`).
        """

    @classmethod
    @abc.abstractmethod
    def provision_argv(cls, prefix: Path, *, python: str = "python3") -> list[list[str]]:
        """The argvs, run in order inside the container, that create the environment at `prefix`.

        python: the interpreter, when the backend invokes one directly.
        """


def resolve(mode: EnvMode) -> type[EnvBackend]:
    """The backend implementation for a manifest's `EnvMode`."""
    try:
        return EnvBackend.find(mode, attr="mode")
    except LookupError:
        raise MissionError(
            f"no env backend for {mode!r}, known modes are "
            f"{[implementation.mode for implementation in EnvBackend.implementations()]}"
        ) from None
