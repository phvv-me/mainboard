# The three verbs over one declared batch: measure what must ship, price it, then dispatch it.
# Each verb publishes what it learned as receipts and reads what the last one left, so the three
# compose in any order a person actually works in and none of them holds state in memory that the
# next one needs.

from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from ..core.project import Project
from ..dispatch import vocabulary
from ..dispatch.schedulers import is_quota_refusal
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

# What a dispatch is allowed to fail with before it becomes this job's row rather than the end of
# the batch. A target that refuses one job says nothing about the next one, and a batch that dies
# on its second job of five has already spent the first one's dispatch for nothing.
_REFUSALS = (MissionError, HostUnreachable, OSError, LookupError, SystemExit)

# What a job the selection left out is recorded as, the one sentence that keeps it apart from a
# job a target refused: nothing happened to it, it was not asked for.
_UNSELECTED = "skipped: not named by --only"

# What a row's `state` column says about the request, one word per thing that can happen to it.
DISPATCHED = "dispatched"


class Dispatched(FrozenModel):
    """What one job's dispatch came to, whether or not a target took it.

    job: the job's name inside the batch.
    target: the alias it was sent to.
    state: what happened to the request, `dispatched`, `held`, `refused` or `skipped`. Named in
        its own column because the three that carry no handle used to read alike, and a job the
        target's quota merely postponed is the one a reader must not file beside a rejection.
    handle: the scheduler or provider handle, empty when the target did not take it.
    kind: how that target is reached.
    reason: why it was refused or held, empty when it was accepted.
    """

    job: str
    target: str
    state: str = DISPATCHED
    handle: str = ""
    kind: str = ""
    reason: str = ""


class Batch:
    """One declared batch of jobs, prepared, priced and dispatched as a unit.

    The batch's identity is its declaration, so the same spec always addresses the same receipts
    stream and the three verbs write one log between them. That log is the only thing they share:
    `estimate` prices whatever `prepare` measured rather than measuring again, and `watch` finds
    every dispatched job in it by id alone, without the spec that declared them.

    A selection narrows what a verb acts on without touching that identity, so a plan whose nine
    ready jobs go out today and whose four remaining ones go out tomorrow is one batch and one
    stream throughout, and the jobs left behind are recorded as skipped rather than left silent.
    """

    def __init__(
        self,
        board: Board,
        spec: BatchSpec,
        *,
        bus: Bus | None = None,
        selection: Selection | None = None,
    ) -> None:
        """board: the workspace the jobs are dispatched from.

        spec: the declared batch.
        bus: where receipts go, the batch's own NDJSON file when None.
        selection: which of the plan's jobs this run acts on, all of them when None.
        """
        self.board = board
        self.spec = spec
        self.dir = directory(board, spec.batch_id)
        self.bus = bus or Receipts(self.dir / "events.ndjson")
        self.selection = selection or Selection()
        self.jobs = self.selection.chosen(spec.jobs)
        self.skipped = tuple(job for job in spec.jobs if not self.selection.holds(job.name))

    @property
    def id(self) -> str:
        """The batch identity, its spec's own."""
        return self.spec.batch_id

    def dispatch(self, job: BatchJob) -> Dispatched:
        """Send one job to its target, recording its handle, its refusal, or its hold.

        A target that refuses on its own count quota has not rejected the job, it has said the
        queue is full right now, so the request is held at this workstation and the durable sweep
        asks again rather than the job being dropped from the wave (four of thirteen, miyabi-g's
        njobs-g limit, 2026-09-04). Every other refusal is still a refusal.
        """
        bound = self.board.on(job.target)
        try:
            run = bound.submit(job.command, name=self.labelling(job.name), **job.submission())
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
        """Price every job from what this workspace already recorded. Nothing runs.

        Whatever `prepare` measured is read back from the receipts rather than measured again,
        so pricing a batch costs nothing at all once it has been prepared, and a job nobody
        prepared is measured here so the row is never a blank where the bytes should be.
        """
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
        """The dispatch label one job of this batch carries, its key in the run registry.

        The job rides in the label because the label is the only thing that reaches the machine
        running the job, and what that machine publishes about itself has to say which job it
        is. Dispatch keeps a label as free text and never parses it, so this method and
        `labelled_batch` are the only two places the `batch:` shape is spelled out.

        job: the job's name inside the batch, empty for the batch as a whole.
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

        The analysis a dispatch is worth doing before: a job whose data never reached its host
        is a job that fails after the queue wait rather than before it, and a mirror that has
        drifted is minutes of transfer nobody planned for.
        """
        self.open()
        transfer = Transfer(self.board)
        measured = [transfer.set_for(job) for job in self.jobs]
        for prepared in measured:
            publish(self.bus, self.id, Topic.PREPARED, job=prepared.job, data=payload(prepared))
        return measured

    def held(self, job: BatchJob, refusal: BaseException) -> Dispatched:
        """Keep one job whose target had no room, so the next sweep asks for it again.

        The request goes into the run registry, which is what makes the hold durable: the sweep
        that resubmits it runs on a cron and knows nothing about the process that held it. The
        receipt is published beside it so this batch's own watch has a row to show rather than a
        gap where a job of the plan should be.
        """
        told = Dispatched(
            job=job.name, target=job.target, state=vocabulary.HELD, reason=str(refusal)
        )
        self.board.dispatcher.hold(
            Request(
                target=job.target,
                command=job.command,
                name=self.labelling(job.name),
                **job.submission(),
            ),
            reason=told.reason,
        )
        publish(
            self.bus,
            self.id,
            Topic.HELD,
            job=job.name,
            data={"target": job.target, "reason": told.reason},
        )
        return told

    def refused(self, job: BatchJob, refusal: BaseException) -> Dispatched:
        """Record one target's refusal as the receipt the batch keeps in place of a handle."""
        told = Dispatched(
            job=job.name, target=job.target, state=vocabulary.VANISHED, reason=str(refusal)
        )
        publish(
            self.bus,
            self.id,
            Topic.REFUSED,
            job=job.name,
            data={"target": job.target, "reason": told.reason},
        )
        return told

    def run(self) -> list[Dispatched]:
        """Dispatch every job to its own target and publish what each dispatch came to.

        One target refusing is that job's row and the next job still goes, since a batch spread
        over a fleet routinely meets one machine that is asleep, out of quota, or not declared,
        and the other four jobs are still worth running.

        A job the selection left out is recorded as skipped rather than left unmentioned, so
        every reader of this stream knows there is nothing coming for it.
        """
        self.open()
        return [
            *(self.dispatch(job) for job in self.jobs),
            *(self.skip(job) for job in self.skipped),
        ]

    def skip(self, job: BatchJob) -> Dispatched:
        """Record one job the selection left out, so nothing downstream waits for it.

        A watch builds its rows out of these receipts, and a declared job with no line at all
        reads exactly like one whose dispatch was lost. This is the line that says the difference:
        nothing happened to it, it was not asked for.
        """
        told = Dispatched(
            job=job.name, target=job.target, state=vocabulary.SKIPPED, reason=_UNSELECTED
        )
        publish(
            self.bus,
            self.id,
            Topic.SKIPPED,
            job=job.name,
            data={"target": job.target, "reason": told.reason},
        )
        return told


def labelled_batch(label: str) -> str:
    """The `<batch>/<job>` inside a dispatch `label`, empty when the label names no batch."""
    return label.removeprefix(_LABEL) if label.startswith(_LABEL) else ""


def directory(board: Board, batch_id: str) -> Path:
    """Where `batch_id` keeps its receipts under `board`'s generated tree."""
    return board.root / Project().out_dir / _BATCHES / batch_id
