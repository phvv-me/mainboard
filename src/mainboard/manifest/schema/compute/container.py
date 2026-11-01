from ....core.base import Declared
from ....enums import Guardrail


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
