from enum import StrEnum, auto


class Kind(StrEnum):
    """What a build problem is, the column a reader sorts the report by."""

    ERROR = auto()
    REFERENCE = auto()
    CITATION = auto()
    LABEL = auto()
    OVERFULL = auto()
    LIMIT = auto()
