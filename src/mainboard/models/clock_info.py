from .base import FrozenModel


class ClockInfo(FrozenModel):
    """GPU and memory clock frequencies.

    sm_mhz: streaming multiprocessor clock in MHz.
    memory_mhz: memory controller clock in MHz.
    """

    sm_mhz: int = 0
    memory_mhz: int = 0
