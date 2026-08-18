from patos import FrozenModel


class Utilization(FrozenModel):
    """Compute and memory-controller utilization percentages.

    gpu_pct: fraction of time at least one kernel executed.
    memory_pct: fraction of time the memory controller was active.
    """

    gpu_pct: int = 0
    memory_pct: int = 0
