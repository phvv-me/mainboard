from enum import StrEnum, auto

from ...core.base import Declared


class EnvMode(StrEnum):
    """How the managed environment lives inside a container base image."""

    PIXI_PREFIX = auto()
    VENV_SYSTEM_SITE = auto()


class Guardrail(StrEnum):
    """Automated protections applied when layering an env onto a base image.

    `unset_pip_constraint` clears the `PIP_CONSTRAINT` NGC images bake in,
    unwritable inside a SIF. `pin_system_packages` stops a resolver from
    shadowing the image's tuned builds (torch and friends) with generic wheels.
    """

    UNSET_PIP_CONSTRAINT = auto()
    PIN_SYSTEM_PACKAGES = auto()


class Container(Declared):
    """A base image the environment layers onto instead of rebuilding.

    The runtime is a registry key (`apptainer`, `docker`, `podman`) or `auto`,
    which picks the first runtime available on the executing host. Binds use
    the runtime's `source:target` syntax; a bare path binds to itself. The
    environment prefix always lives on a bound host path, never in the image,
    so a fixed off-the-shelf image serves every dependency change.
    """

    image: str
    runtime: str = "auto"
    gpus: bool = True
    binds: list[str] = []
    env_mode: EnvMode = EnvMode.VENV_SYSTEM_SITE
    guardrails: list[Guardrail] = [Guardrail.UNSET_PIP_CONSTRAINT, Guardrail.PIN_SYSTEM_PACKAGES]
    workdir: str = ""
    passthrough: list[str] = []
