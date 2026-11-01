from enum import StrEnum, auto


class Outcome(StrEnum):
    """What one verb did to one repository.

    `done` changed something, `current` had nothing to change, `held` was refused by a rule
    that protects work (a diverged branch, a pointer the remote cannot serve, a submodule that
    did not get there first), and `failed` is git refusing on its own terms.
    """

    DONE = auto()
    CURRENT = auto()
    HELD = auto()
    FAILED = auto()

    @property
    def settled(self) -> bool:
        """Whether the repository reached what the verb asked of it."""
        return self in {Outcome.DONE, Outcome.CURRENT}
