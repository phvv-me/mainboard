from enum import StrEnum, auto

from .ledger import Observation


class Verdict(StrEnum):
    """How one landing ended, as `Observation.verdict` spells it.

    Only `LANDED` delivered a machine that ran the job. `REFUSED` is a create the provider
    declined, which allocated and billed nothing; every other end billed the setup it wasted.
    """

    LANDED = auto()
    SLOW = auto()
    WRONG_CARD = "wrong-card"
    UNREACHABLE = auto()
    REFUSED = auto()
    FAILED = auto()

    @classmethod
    def of(cls, row: Observation) -> Verdict | None:
        """How `row`'s landing ended, None when it says nothing about the machine.

        A row written before verdicts existed (`batch.watch` records one only once the job ran)
        counts as landed; a verdict this release does not know counts as failed, since an unread
        outcome must not read as a success.
        """
        if not row.verdict:
            return cls.LANDED if row.t_running else None
        try:
            return cls(row.verdict)
        except ValueError:
            return cls.FAILED
