from .base import FrozenModel


class Utilization(FrozenModel):
    """GPU core and memory controller utilization.

    gpu_pct: fraction of time the GPU executed at least one kernel (0-100).
    memory_pct: fraction of time the memory controller was active (0-100).
    """

    gpu_pct: int = 0
    memory_pct: int = 0
