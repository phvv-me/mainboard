from enum import StrEnum, auto


class Scheduler(StrEnum):
    """Job scheduler detected on the host's PATH."""

    SLURM = auto()
    PBS = auto()
    PUEUE = auto()
    NONE = auto()
