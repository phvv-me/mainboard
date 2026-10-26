"""Every tracked run rebuilt and resolved, one query per target rather than one per run."""

from typing import TYPE_CHECKING, Final

from ...core.errors import MissionError
from ...core.project import Project
from ...dispatch.backends.base import route
from ...dispatch.jobs import vocabulary
from ...dispatch.jobs.vocabulary import JobState
from ...dispatch.schedulers import HostUnreachable
from ...fleet.runs import Job
from .constants import QUEUED, UNCREATED

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ...dispatch.state import RunRecord
    from ...fleet.board import Board, Run

# How many of an undeclared target's runs are named by handle before the rest are counted.
_NAMED: Final = 4


class Sweep:
    """Every tracked run rebuilt and resolved, one query per target rather than one per run.

    A cache holding a thousand runs on one box costs one query, not a thousand round trips, and
    a run whose verdict is already terminal costs nothing. A target that cannot be resolved (one
    the manifest no longer declares, no declared root, a host that will not answer, a provider
    API that refuses, or one that goes quiet halfway) is recorded once with why; its unanswered
    runs simply have no state here and are left for the next pass. An undeclared target is never
    contacted: its machine is gone, and asking cost every pass an ssh timeout.

    runs: each record's rebuilt run.
    states: each record's current state, absent where its target could not be resolved.
    down: why a target could not be resolved, one entry per target.
    """

    @staticmethod
    def _groups(records: Sequence[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
        """The records a target can be asked about, by target and the kind they were sent as.

        The kind rides in the key because it picks the scheduler and was recorded at dispatch,
        so a run is asked about the way it was submitted even if its host was redeclared. A held
        or uncreated dispatch has no handle any target heard of; `Monitor.held` asks again.
        """
        groups: dict[tuple[str, str], list[RunRecord]] = {}
        for record in records:
            if record.verdict != vocabulary.Word.HELD and not uncreated(record):
                groups.setdefault((record.target, record.kind), []).append(record)
        return groups

    @staticmethod
    def undeclared(target: str, records: Sequence[RunRecord]) -> str:
        """Why `target`'s runs never settle on their own, and the commands that settle them.

        The handles are named with their host, since a rental's are small integers several
        hosts share and a run that already ended is on no listing of live ones: two such runs
        were announced on every sweep for a day with nothing saying which to cancel.
        """
        cancel = f"`{Project().name} job cancel {{}} --on {target}`"
        named = ", ".join(cancel.format(record.handle) for record in records[:_NAMED])
        more = f" and {len(records) - _NAMED} more" if len(records) > _NAMED else ""
        return (
            f"{target} is no longer declared, so it is never asked; "
            f"settle its runs with {named}{more}"
        )

    def __init__(
        self, board: Board, records: Sequence[RunRecord], *, down: Mapping[str, str] | None = None
    ) -> None:
        """records: every run the sweep still owes an outcome for, in the order it reports them.

        down: targets already found unanswering moments ago, each with why, which are not asked
            again: a listing sweeps right after the settling pass, and knocking twice on one
            dead host doubled what it cost (34 s of a `job list`, 2026-09-30).
        """
        self.board = board
        self.runs: dict[RunRecord, Run] = {}
        self.states: dict[RunRecord, JobState] = {}
        self.down: dict[str, str] = dict(down or {})
        for (target, kind), owned in Sweep._groups(records).items():
            if target not in self.down:
                self._resolve(target, kind=kind, records=owned)

    def _resolve(self, target: str, *, kind: str, records: Sequence[RunRecord]) -> None:
        """Settle one target's records, or note why that target could not be asked."""
        if not self.board.declares(target):
            self.down[target] = Sweep.undeclared(target, records)
            return
        try:
            self.settle(kind, records)
        except (HostUnreachable, MissionError, OSError) as fault:
            self.down[target] = str(fault)

    def settle(self, kind: str, records: Sequence[RunRecord]) -> None:
        """Rebuild every run in `records` and resolve the ones still owed a probe.

        Every record is rebuilt, settled ones included, since a target with no declared root
        refuses at the rebuild. A terminal verdict can never change, so it is read from the cache
        without touching the target, which also keeps a finished job the queue already forgot
        from reading back as vanished. The rest cost one query for the whole host, or one call
        per run where a provider, which has no listing, owns them.
        """
        self.runs |= {
            record: self.board.job(record.handle, host=record.target) for record in records
        }
        self.states |= {
            record: JobState.model_validate(record, from_attributes=True)
            for record in records
            if record.verdict in vocabulary.Word.settled()
        }
        pending = [record for record in records if record not in self.states]
        if not pending:
            return
        if route(kind) != QUEUED:
            self.states |= {record: self.runs[record].poll() for record in pending}
            return
        found = Job.states([self.runs[record].handle for record in pending])
        self.states |= {record: found[record.handle] for record in pending}


def uncreated(record: RunRecord) -> str:
    """Why a provider creation with no confirmed handle is only reported, empty for any other."""
    return UNCREATED.get(record.verdict or "", "")
