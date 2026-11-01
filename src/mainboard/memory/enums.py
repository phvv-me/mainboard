from enum import StrEnum, auto


class Kind(StrEnum):
    """What an item is."""

    FILE = auto()
    TEXT = auto()
    URI = auto()


class State(StrEnum):
    """Where one keep stands after a call."""

    KEPT = auto()
    KNOWN = auto()
    QUEUED = auto()
    REFUSED = auto()
