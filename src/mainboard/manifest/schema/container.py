from enum import StrEnum, auto

from ...core.base import Declared


class Guardrail(StrEnum):
    """Protections applied when layering an env onto a base image.

    `unset_pip_constraint` clears the `PIP_CONSTRAINT` NGC images bake in (unwritable in a SIF).
    """

    UNSET_PIP_CONSTRAINT = auto()


class Container(Declared):
    """A fixed base image the environment layers onto from a bound host prefix.

    runtime: `apptainer`, `docker`, `podman`, or `auto` for the first available on the host.
    binds: the runtime's `source:target` syntax; a bare path binds to itself.
    """

    image: str
    runtime: str = "auto"
    gpus: bool = True
    binds: list[str] = []
    guardrails: list[Guardrail] = [Guardrail.UNSET_PIP_CONSTRAINT]
    workdir: str = ""
    passthrough: list[str] = []
