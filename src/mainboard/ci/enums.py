from enum import StrEnum, auto

from ..enums import Exit


class Verdict(StrEnum):
    """How one step ended on one leg.

    `ok` passed, `failed` exited nonzero, `timed out` outlived its deadline, `missing` found no
    such program, `unreachable` never reached the host, and `not run` followed a failure.
    """

    OK = auto()
    FAILED = auto()
    TIMED_OUT = "timed out"
    MISSING = auto()
    UNREACHABLE = auto()
    NOT_RUN = "not run"

    @classmethod
    def of(cls, code: int) -> Verdict:
        """The verdict an exit code states, with the shell's words for a deadline and a miss."""
        return {0: cls.OK, Exit.TIMED_OUT: cls.TIMED_OUT, Exit.MISSING: cls.MISSING}.get(
            code, cls.FAILED
        )
