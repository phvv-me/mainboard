from enum import StrEnum, auto


class Kind(StrEnum):
    """How the supervisor runs a child: kept running, or run on a period."""

    PERIODIC = auto()
    RESIDENT = auto()
