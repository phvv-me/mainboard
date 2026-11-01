from enum import StrEnum, auto


class GateStatus(StrEnum):
    """A gate's judgement on a trial precondition."""

    PASSED = auto()
    BLOCKED = auto()
    FAILED = auto()
