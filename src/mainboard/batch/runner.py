# The three verbs over one declared batch: measure what must ship, price it, then dispatch it.
# Each verb publishes what it learned as receipts and reads what the last one left, so they
# compose in any order and none holds state in memory the next one needs.

from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..dispatch import vocabulary
from ..dispatch.schedulers import is_quota_refusal
from ..dispatch.shared import Watcher
from ..dispatch.transport import HostUnreachable
from ..dispatch.vocabulary import Request
from .estimate import Estimator
from .receipts import Receipts, Topic, latest, payload, publish
from .spec import Selection
from .transfer import Transfer, TransferSet

if TYPE_CHECKING:
    from pathlib import Path

    from ..board import Board
    from .estimate import BatchEstimate
    from .receipts import Bus
    from .spec import BatchJob, BatchSpec

# Where a batch keeps its own directory under the workspace's generated tree, one per batch id.
_BATCHES = "batches"

# How a batch's jobs are labelled in the run registry, the prefix `labelled_batch` reads back.
_LABEL = "batch:"

# What a dispatch may fail with and become this job's row rather than end the batch: one refusal
# says nothing about the next job, and dying on job two of five wastes job one's dispatch.
_REFUSALS = (MissionError, HostUnreachable, OSError, LookupError, SystemExit)

# What a job the selection left out is recorded as: not refused, just not asked for.
_UNSELECTED = "skipped: not named by --only"

# What a row's `state` column says about the request, one word per thing that can happen to it.
DISPATCHED = "dispatched"


class Dispatched(FrozenModel):
    """What one job's dispatch came to, whether or not a target took it.

    state: `dispatched`, `held`, `refused` or `skipped`, its own column since the three without
        a handle read alike, and a quota postponement must not be filed beside a rejection.
    handle: the scheduler or provider handle, empty when the target did not take it.
    reason: why it was refused, held or skipped, empty when it was accepted.
    """

    job: str
    target: str
    state: str = DISPATCHED
    handle: str = ""
    kind: str = ""
    reason: str = ""


class Batch:
    """One declared batch of jobs, prepared, priced and dispatched as a unit.

    Its identity is its declaration, so the verbs share one receipts stream and nothing else:
    `estimate` prices what `prepare` measured, and `watch` finds every job by id alone. A
    `selection` (all jobs when None) narrows a verb without touching that identity, so waves of one
    plan share one stream and the jobs left behind are recorded as skipped.
    """

    def __init__(
        self,
        board: Board,
        spec: BatchSpec,
        *,
        bus: Bus | None = None,
        selection: Selection | None = None,
    ) -> None:
        self.board = board
        self.spec = spec
        self.dir = directory(board, spec.batch_id)
        self.bus = bus or Receipts(self.dir / "events.ndjson")
        self.selection = selection or Selection()
        self.jobs = self.selection.chosen(spec.jobs)
        self.skipped = tuple(job for job in spec.jobs if not self.selection.holds(job.name))

    @property
    def id(self) -> str:
        return self.spec.batch_id

    def dispatch(self, job: BatchJob, *, watch: Watcher | None = None) -> Dispatched:
        """Send one job to its target, recording its handle, its refusal, or its hold.

        A count-quota refusal only says the queue is full now, so the request is held for the
        durable sweep to ask again rather than dropped from the wave (four of thirteen, miyabi-g's
        njobs-g limit, 2026-09-04).
        """
        bound = self.board.on(job.target)
        try:
            run = bound.submit(
                job.command, name=self.labelling(job.name), watch=watch, **job.submission()
            )
        except _REFUSALS as refusal:
            if is_quota_refusal(str(refusal)):
                return self.held(job, refusal)
            return self.refused(job, refusal)
        kind = run.handle.kind
        publish(
            self.bus,
            self.id,
            Topic.SUBMITTED,
            job=job.name,
            data={
                "handle": run.handle.id,
                "target": job.target,
                "kind": kind,
                "command": job.command,
                **({"node": job.node} if job.node else {}),
            },
        )
        return Dispatched(job=job.name, target=job.target, handle=run.handle.id, kind=kind)

    def estimate(self) -> BatchEstimate:
        """Price every job from what `prepare` measured, measuring the rest. Nothing runs."""
        self.open()
        prepared = latest(self.bus.replay(), Topic.PREPARED)
        transfer = Transfer(self.board)
        measured = [
            TransferSet.model_validate(prepared[job.name].data)
            if job.name in prepared
            else transfer.set_for(job)
            for job in self.jobs
        ]
        table = Estimator(self.board).table(self.id, self.jobs, measured)
        for row in table.jobs:
            publish(self.bus, self.id, Topic.ESTIMATED, job=row.job, data=payload(row))
        return table

    def labelling(self, job: str) -> str:
        """The dispatch label of one job (empty for the whole batch), its run-registry key.

        The label is all that reaches the machine running the job, which must say which job it is.
        Dispatch never parses it; only this and `labelled_batch` spell the `batch:` shape.
        """
        return f"{_LABEL}{self.id}/{job}" if job else f"{_LABEL}{self.id}"

    def open(self) -> None:
        """Announce the batch once, whichever verb touches its receipts first."""
        if self.bus.replay():
            return
        publish(
            self.bus,
            self.id,
            Topic.OPENED,
            data={
                "name": self.spec.name,
                "jobs": [job.name for job in self.spec.jobs],
                "root": str(self.board.root),
            },
        )

    def prepare(self) -> list[TransferSet]:
        """Measure what each job must still put on its target, and publish each measurement.

        Missing data fails a job after the queue wait, and a drifted mirror is unplanned transfer.
        """
        self.open()
        transfer = Transfer(self.board)
        measured = [transfer.set_for(job) for job in self.jobs]
        for prepared in measured:
            publish(self.bus, self.id, Topic.PREPARED, job=prepared.job, data=payload(prepared))
        return measured

    def held(self, job: BatchJob, refusal: BaseException) -> Dispatched:
        """Keep one job whose target had no room in the run registry, durable for the cron sweep
        that resubmits it, with a receipt so this batch's watch shows a row."""
        self.board.dispatcher.hold(
            Request(
                target=job.target,
                command=job.command,
                name=self.labelling(job.name),
                **job.submission(),
            ),
            reason=str(refusal),
        )
        return self.told(job, vocabulary.HELD, Topic.HELD, str(refusal))

    def refused(self, job: BatchJob, refusal: BaseException) -> Dispatched:
        """Record one target's refusal as the receipt the batch keeps in place of a handle."""
        return self.told(job, vocabulary.VANISHED, Topic.REFUSED, str(refusal))

    def run(self, *, watch: Watcher | None = None) -> list[Dispatched]:
        """Dispatch every selected job to its own target, one refusal being only that job's row,
        and record the unselected as skipped.

        watch: announces what each dispatch does on the far side, for a queued host the priming
            of the environment its wave will run in.
        """
        self.open()
        return [
            *(self.dispatch(job, watch=watch) for job in self.jobs),
            *(self.skip(job) for job in self.skipped),
        ]

    def skip(self, job: BatchJob) -> Dispatched:
        """Record one job the selection left out, which would otherwise read as a lost dispatch."""
        return self.told(job, vocabulary.SKIPPED, Topic.SKIPPED, _UNSELECTED)

    def told(self, job: BatchJob, state: str, topic: Topic, reason: str) -> Dispatched:
        """Publish why `job` has no handle and answer its row."""
        publish(
            self.bus, self.id, topic, job=job.name, data={"target": job.target, "reason": reason}
        )
        return Dispatched(job=job.name, target=job.target, state=state, reason=reason)


def labelled_batch(label: str) -> str:
    """The `<batch>/<job>` inside a dispatch `label`, empty when the label names no batch."""
    return label.removeprefix(_LABEL) if label.startswith(_LABEL) else ""


def directory(board: Board, batch_id: str) -> Path:
    """Where `batch_id` keeps its receipts under `board`'s generated tree."""
    return board.root / Project().out_dir / _BATCHES / batch_id
