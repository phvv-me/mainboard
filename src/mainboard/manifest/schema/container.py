from enum import StrEnum, auto

from ...core.base import Declared


class EnvMode(StrEnum):
    """How the managed environment lives inside a container base image."""

    PIXI_PREFIX = auto()
    VENV_SYSTEM_SITE = auto()


class Guardrail(StrEnum):
    """Protections applied when layering an env onto a base image.

    `unset_pip_constraint` clears the `PIP_CONSTRAINT` NGC images bake in (unwritable in a SIF);
    `pin_system_packages` keeps a resolver from shadowing the image's tuned torch builds.
    """

    UNSET_PIP_CONSTRAINT = auto()
    PIN_SYSTEM_PACKAGES = auto()


class Container(Declared):
    """A fixed base image the environment layers onto from a bound host prefix.

    runtime: `apptainer`, `docker`, `podman`, or `auto` for the first available on the host.
    binds: the runtime's `source:target` syntax; a bare path binds to itself.
    """

    image: str
    runtime: str = "auto"
    gpus: bool = True
    binds: list[str] = []
    env_mode: EnvMode = EnvMode.VENV_SYSTEM_SITE
    guardrails: list[Guardrail] = [Guardrail.UNSET_PIP_CONSTRAINT, Guardrail.PIN_SYSTEM_PACKAGES]
    workdir: str = ""
    passthrough: list[str] = []
