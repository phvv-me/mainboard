from enum import StrEnum, auto


class Probed(StrEnum):
    """Why one axis of a cell reads what it reads, recorded beside the value itself."""

    FOUND = auto()
    ABSENT = auto()
    FAILED = auto()
    UNASKED = auto()


class Admissibility(StrEnum):
    """Whether one row's producing tree can be identified, which is what makes it evidence.

    ADMISSIBLE: the lane belongs to the verified content snapshot.
    DIRTY: historical rejection label, preserved when reading old receipts.
    UNTRACKED: historical rejection label, preserved when reading old receipts.
    UNRECORDED: the row was written before this field existed and can prove none of the above.
    """

    ADMISSIBLE = auto()
    DIRTY = auto()
    UNTRACKED = auto()
    UNRECORDED = auto()


# `Outcome` is fixed: whether the instrument worked, the word `mb job show` branches an exit
# code on. The settled word beside it is the consumer's vocabulary, whose meaning this module does
# not know; each word carries only the letter and terminal markup a progress line prints.
class Outcome(StrEnum):
    """Whether a trial's instrument worked, which is never the consumer's word to choose."""

    PASSED = auto()
    FAILED = auto()


class Stance(StrEnum):
    """What one settled word does to the prediction behind it, declared per word.

    `neither` is the default, so a consumer that never thinks about stance is never recorded as
    having claimed anything, and an inconclusive separation is not rounded into a decisive word.
    """

    CONFIRMS = auto()
    REFUTES = auto()
    NEITHER = auto()
