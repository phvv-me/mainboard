from .base import FrozenModel


class Clock(FrozenModel):
    """Clock reading for one processing-unit domain.

    domain: hardware domain, e.g. `cpu`, `gpu_compute`, `memory`.
    current_mhz: current frequency when sampled.
    max_mhz: maximum or advertised frequency when known.
    source: provider that produced the value.
    supported: whether this platform exposes the reading.
    """

    domain: str
    current_mhz: float | None = None
    max_mhz: float | None = None
    source: str = ""
    supported: bool = True
