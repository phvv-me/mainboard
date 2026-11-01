"""How memory is: what waits in the spool, and what the upstream says when it answers.

`mb memory status` prints it whole; the doctor's `memory` row reads the spool alone, because a
doctor that waited on a server would be a doctor that hung with it.
"""

from datetime import datetime
from typing import Self

from patos import FrozenModel
from pydantic import AwareDatetime

from ..core.constants.units import SECONDS_PER_DAY
from ..core.errors import Unreachable
from ..core.section import Section
from ..enums import Verdict
from ..render.values import Node
from .outbox import Memory, Outbox
from .upstream import Remote, Upstream

_ROW = "memory"
_STATUS = "mb memory status"


class MemoryHealth(FrozenModel):
    """The state of one workspace's memory.

    server: where the upstream answers, empty when it was not asked.
    waiting: keeps in the spool that no pass has delivered.
    oldest: when the first of them was made.
    remote: what the upstream said, None when it was not asked or did not answer.
    reason: why `remote` is None, when it was asked.
    """

    server: str = ""
    waiting: int
    oldest: AwareDatetime | None = None
    remote: Remote | None = None
    reason: str = ""

    @classmethod
    def of(cls, outbox: Outbox, upstream: Upstream | None = None, *, reason: str = "") -> Self:
        """The spool, and the upstream's own word on itself after delivering the spool to it.

        reason: why there is no upstream to ask, said as it is.
        """
        queued = outbox.waiting()
        health = cls(
            waiting=len(queued), oldest=queued[0].since if queued else None, reason=reason
        )
        if upstream is None:
            return health
        try:
            remote = cls._asked(outbox, upstream)
        except Unreachable as fault:
            return health.model_copy(update={"server": upstream.server, "reason": str(fault)})
        return cls.of(outbox).model_copy(update={"server": upstream.server, "remote": remote})

    @staticmethod
    def _asked(outbox: Outbox, upstream: Upstream) -> Remote:
        """The upstream's word on itself, after delivering the spool to it."""
        Memory(outbox, upstream).drain()
        return upstream.status()

    def row(self) -> dict[str, Node]:
        """The status as one record."""
        remote = self.remote
        return {
            "upstream": self.server,
            "login": "" if remote is None else "yes" if remote.login else "no",
            "queued": self.waiting,
            "oldest": self.oldest.astimezone().isoformat(timespec="seconds")
            if self.oldest
            else "",
            "caller": remote.caller if remote else "",
            "finds": remote.finds if remote else "",
            "keeps": remote.keeps if remote else "",
            "processing": remote.processing if remote else "",
            "unanswered": self.reason,
        }

    def section(self, at: datetime) -> Section:
        """The doctor's row: fit while nothing waits, a word on how long the oldest keep has."""
        if not self.waiting or self.oldest is None:
            return Section(section=_ROW, verdict=Verdict.PASS, detail="no keep waits in the spool")
        days = (at - self.oldest).total_seconds() / SECONDS_PER_DAY
        return Section(
            section=_ROW,
            verdict=Verdict.WARN,
            detail=f"{self.waiting} keeps wait in the spool, the oldest {days:.1f} days old",
            fix=_STATUS,
        )
