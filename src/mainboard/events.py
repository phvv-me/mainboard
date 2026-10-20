# THE EVENT STREAM OF THE SETTLING PASS: ONE JSON LINE FOR EVERY JOB THAT MOVED.
#
# A sweep already knows what changed (a job settled ok, failed, was held by a quota or got through
# one), and it told only the process that ran it. A person or an agent who wants to act on those
# changes had to list the jobs again and diff by eye. The sweep now also appends each change to
# `.mainboard/events/jobs.jsonl`, so anything that follows that file (a notifier, a triage script,
# a language model asked about failures only) reads each change exactly once and a quiet batch
# costs it nothing.

from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from .dispatch.shared import now
from .dispatch.state import MonitorReport

if TYPE_CHECKING:
    from .board import Board


class Kind(StrEnum):
    DISPATCHED = auto()
    FAILED = auto()
    HELD = auto()
    OK = auto()


class JobEvent(FrozenModel):
    """One job that moved in one sweep.

    detail: the failure's cause, a held dispatch's refusal, or the path results came home to.
    node: the ledger slug the run serves, which scopes a stream to one experiment.
    """

    at: str
    kind: Kind
    host: str
    handle: str
    name: str = ""
    node: str = ""
    detail: str = ""


class Events:
    """The append-only stream of a workspace, read by byte offset so a follower never rereads."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".mainboard" / "events" / "jobs.jsonl"

    def record(self, board: Board, report: MonitorReport) -> list[JobEvent]:
        """Append what `report` moved, naming each run from the dispatch cache."""
        stamp = now()
        moved = [
            *((Kind.DISPATCHED, run.target, run.handle, run.name) for run in report.resumed),
            *((Kind.HELD, run.target, run.handle, run.reason) for run in report.held),
            *((Kind.OK, job.target, job.handle, job.pulled_path or "") for job in report.finished),
            *((Kind.FAILED, job.target, job.handle, job.reason) for job in report.failed),
        ]
        events = [
            self._named(
                board, JobEvent(at=stamp, kind=kind, host=host, handle=handle, detail=text)
            )
            for kind, host, handle, text in moved
        ]
        if events:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.writelines(event.model_dump_json() + "\n" for event in events)
        return events

    def since(self, cursor: int = 0) -> tuple[list[JobEvent], int]:
        """Every event after byte `cursor`, and the cursor to pass next time."""
        if not self.path.exists():
            return [], 0
        with self.path.open("rb") as stream:
            stream.seek(cursor)
            fresh = stream.read()
        complete = fresh[: fresh.rfind(b"\n") + 1]
        lines = complete.decode("utf-8").splitlines()
        return [JobEvent.model_validate_json(line) for line in lines], cursor + len(complete)

    @staticmethod
    def _named(board: Board, event: JobEvent) -> JobEvent:
        """The event with its run's label and ledger slug, from the cache that dispatched it."""
        try:
            run = board.dispatcher.cache.run(event.handle, event.host)
        except LookupError:
            return event
        return event.model_copy(update={"name": run.label, "node": run.node})
