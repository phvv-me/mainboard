from enum import StrEnum, auto


class Fence(StrEnum):
    """How far a cancel got, as `Remote.fence` prints it.

    EXITED: the runner already wrote the exit artifact, which stands untouched.
    FENCED: no allocation took the entry and none can now.
    CLAIMED: an allocation owns it and ends it on the tombstone.
    """

    EXITED = auto()
    FENCED = auto()
    CLAIMED = auto()
