"""What a dispatch is asked for (`Order`) and what it hands back (`Handle`, `Verdict`)."""

from collections.abc import Callable

from patos import FrozenModel

from ...context.plan import ExecutionPlan
from ...runtime.job import ToolCall
from ..jobs.vocabulary import Resources
from ..mirroring.shipment import Shipment
from ..shared import HandleId, Watcher

# A verdict's process exit code: 0 ok, 1 failed, 2 still running, 3 vanished or unknown.
_VERDICT_EXITS = {"ok": 0, "failed": 1, "running": 2}


class Order(FrozenModel):
    """One job to dispatch: what runs, where, under which resources and environment.

    resources: the scheduler request, queue, walltime, memory and GPUs already resolved.
    name: the run's label, which also names the directory its attempts resume from.
    node: the ledger slug this run serves, recorded on the run and its receipts.
    watch: announces the far-side stages long enough to be worth saying.
    root: the mirror's workspace root on an ssh-family host, empty for a rental.
    prefix: the built environment the job activates, addressed by the content of the compiled
        artifact shipped; empty activates the workspace's own environment.
    artifact: the compiled manifest, lock and state the environment was addressed by, shipped so
        the pinned tree carries the very compile the pin was taken over.
    sampler: the call the job makes beside its command; the caller decides what a host watches
        about itself.
    attestation: the call the job makes in the foreground before its command, recording what the
        machine looked like as the work started.
    containerize: builds the container argv around `["bash", "-c", cmd]`; required when the plan
        is containerized.
    """

    plan: ExecutionPlan
    shipment: Shipment
    resources: Resources
    name: str = ""
    node: str = ""
    watch: Watcher | None = None
    root: str = ""
    prefix: str = ""
    artifact: tuple[str, ...] = ()
    sampler: ToolCall | None = None
    attestation: ToolCall | None = None
    containerize: Callable[[list[str]], list[str]] | None = None


class Handle(FrozenModel):
    """A dispatched job, enough to poll, await, or fetch it without re-resolving the host.

    id: the scheduler's handle (PBS or SLURM job id, pueue task id, local run id), always text,
        since pueue's small integers read back as numbers would fail validation inside a poll.
    host: the ssh alias the job runs on, or the declared alias of the provider host rented for it.
    root: the mirror's workspace root on that host, empty for a provider that syncs none. The
        mirror rather than the snapshot, since later log reads, results pulls and post-mortems
        address it and it outlives the snapshot.
    kind: the scheduler (`pbs`/`slurm`/`ssh`/`local`) or provider (`vast`/`hpc-ai`/`modal`) kind
        at submit time, which routes a later probe back to whichever answered for this run.
    fetch_path: the results path recorded at submit time, pulled back by `Job.pull`.
    """

    id: HandleId
    host: str
    root: str
    kind: str
    fetch_path: str | None = None


class Verdict(FrozenModel):
    """A terminal outcome of an awaited job, the value `Job.wait` answers.

    verdict: `ok`, `failed`, `vanished` or `unknown`.
    exit_code: the process exit status, when the scheduler reported one.
    reason: a one-line cause for a non-ok verdict.
    """

    verdict: str
    exit_code: int | None = None
    reason: str = ""

    @property
    def code(self) -> int:
        """The verdict as a process exit code."""
        return _VERDICT_EXITS.get(self.verdict, 3)

    @property
    def ok(self) -> bool:
        """Whether the job finished cleanly."""
        return self.verdict == "ok"
