from enum import StrEnum, auto


class Kind(StrEnum):
    """What is wrong with a family: a library from outside, one from nowhere, or two releases."""

    OUTSIDE = auto()
    MISSING = auto()
    SPLIT = auto()
