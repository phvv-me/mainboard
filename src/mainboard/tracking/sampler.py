# The live lane: this machine's own readings, published into a job's receipts while it runs.
#
# A hosted dashboard sees the memory a process used, never the cgroup ceiling the scheduler set,
# which is what an OOM kill fires against. Every sample carries used memory, that cap and the
# fraction between them, the series that says whether a job is about to die. Samples land in the
# job's own NDJSON first and the declared sink second, so this runs unchanged on a laptop, on
# gold and on a compute node with no route out.

from threading import Event as Flag
from threading import Thread
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import psutil

from ..batch.receipts import Topic, publish
from ..core.project import Project
from ..probe.machine import Machine
from ..runtime.job import ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from pydantic import JsonValue

    from ..batch.receipts import Bus, Event

# How long a stop waits for the sampling thread to notice, past which the thread is a daemon and
# the process may leave without it.
_JOIN_S = 5.0

# The file a host keeps the tracking credential in, written by the dispatcher and read by the job.
# Not the workspace `.env`, so staging never overwrites what the host declares, and one known
# path to audit, rotate or delete.
_HOST_ENV = "tracking.json"

# Compute utilization above which an accelerator counts as working, `probe.gating.gpu_busy`'s
# threshold, so one workspace has one idea of busy.
_BUSY_PCT = 10


class Reading(Protocol):
    """A memory figure a sample reads."""

    @property
    def used_gb(self) -> float: ...


class Ceiling(Protocol):
    """The enforced memory cap a sample reads its headroom against."""

    @property
    def capped(self) -> bool: ...

    @property
    def limit_gb(self) -> float: ...


class Busyness(Protocol):
    """A unit's compute and memory-controller utilization."""

    @property
    def gpu_pct(self) -> int: ...

    @property
    def memory_pct(self) -> int: ...


class Accelerator(Protocol):
    """One GPU as a sample reads it."""

    @property
    def memory(self) -> Reading: ...

    @property
    def utilization(self) -> Busyness: ...


class Node(Protocol):
    """The host as a sample reads it, with the cap its jobs really run under."""

    @property
    def cgroup_memory(self) -> Ceiling: ...

    @property
    def memory(self) -> Reading: ...


class Sampled(Protocol):
    """The slice of a machine one sample reads, so a test hands over a stand-in instead."""

    @property
    def gpus(self) -> Sequence[Accelerator]: ...

    @property
    def host(self) -> Node: ...


class Sampler:
    """This machine, read into a job's receipts on a fixed interval, from its own thread.

    Entering takes the first reading at once, so even a job dying in its first minute leaves a
    series. An interval of zero samples nothing, turning the lane off without the caller
    branching. The thread is a daemon and every stop is bounded, since this runs beside somebody's
    training loop and must never be why a job will not exit.
    """

    def __init__(
        self,
        bus: Bus,
        *,
        stream: str,
        job: str,
        interval: float,
        seconds: float = 0.0,
        parent: int = 0,
        machine: Sampled | None = None,
    ) -> None:
        """bus: where samples are published, the job's own receipts.

        interval: seconds between readings, 0 to sample nothing.
        seconds: a hard stop, 0 to sample until the caller stops it.
        parent: a pid whose exit ends sampling, 0 for none; beside a dispatched command it is the
            command's shell, so the sampler never outlives the job as an orphan.
        machine: what is read, this machine when None.
        """
        self.bus = bus
        self.stream = stream
        self.job = job
        self.interval = interval
        self.seconds = seconds
        self.parent = parent
        self.machine = machine or Machine()
        self.stopped = Flag()
        self.opened = monotonic()
        self.thread = Thread(target=self.loop, name=f"mainboard-sample-{job}", daemon=True)

    def __enter__(self) -> Sampler:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.stop()

    @property
    def expired(self) -> bool:
        """Whether this sampler has outlived its budget or the process it was told to follow."""
        if self.seconds and monotonic() - self.opened >= self.seconds:
            return True
        return bool(self.parent) and not psutil.pid_exists(self.parent)

    def attest(self) -> Event:
        """Publish one reading as this job's `job.attested` receipt, saying if the node was idle.

        Taken once, in the foreground, before the work starts. Jobs on one host run concurrently,
        so a contended benchmark otherwise looks as authoritative as a clean one (a 1.29x speedup
        vanished when re-run on a verified-idle host, 2026-08-22). Attesting rather than
        serializing forbids nothing and lets a campaign decide what a busy machine means, and the
        whole reading rides along so a reader judges the conditions, not one word.
        """
        busiest = max((gpu.utilization.gpu_pct for gpu in self.machine.gpus), default=0)
        return publish(
            self.bus,
            self.stream,
            Topic.ATTESTED,
            job=self.job,
            data={**self.reading(), "idle": busiest <= _BUSY_PCT},
        )

    def loop(self) -> None:
        self.sample()
        while not self.stopped.wait(self.interval) and not self.expired:
            self.sample()

    def reading(self) -> dict[str, JsonValue]:
        """One reading of this machine: GPU memory and busyness, host memory against its cap."""
        host, gpus = self.machine.host, self.machine.gpus
        cap = host.cgroup_memory
        used = host.memory.used_gb
        return {
            "gpu_used_gb": round(sum(gpu.memory.used_gb for gpu in gpus), 4),
            "gpu_pct": max((gpu.utilization.gpu_pct for gpu in gpus), default=0),
            "gpu_memory_pct": max((gpu.utilization.memory_pct for gpu in gpus), default=0),
            "host_used_gb": round(used, 4),
            "host_cap_gb": round(cap.limit_gb, 4),
            "host_capped": cap.capped,
            "host_frac": round(used / cap.limit_gb, 4) if cap.limit_gb else 0.0,
        }

    def sample(self) -> Event:
        return publish(self.bus, self.stream, Topic.SAMPLE, job=self.job, data=self.reading())

    def start(self) -> None:
        self.opened = monotonic()
        if self.interval > 0:
            self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=_JOIN_S)


def host_env(root: str) -> str:
    """Where a host under workspace `root` keeps the tracking credential, as a JSON object."""
    return f"{root}/{Project().out_dir}/{_HOST_ENV}"


def attesting(*, root: str, stream: str, job: str) -> ToolCall:
    """The call a dispatched job makes to attest to its own machine before it works.

    The foreground twin of `sampling`: a reading taken before the command describes the
    conditions it was handed, not the command. Its output is discarded (it belongs in the
    receipts, not the log), and a failure never stops the job, since a missing attestation is a
    row saying nothing while a refused dispatch is a run that never happened.

    root: the workspace root on the host, where the staged credential lives.
    """
    return ToolCall(args=("attest", stream, "--job", job), credentials=host_env(root))


def sampling(
    *, root: str, stream: str, job: str, interval: float, seconds: float = 0.0
) -> ToolCall | None:
    """The call a dispatched job makes to sample itself, None for an `interval` of 0.

    The seam carrying the live lane onto another machine: the job's runner starts the host's own
    tool in the job's environment, publishing into that host's receipts and onward, with nothing
    configured on the host. It never outlives its job: the runner hands it its pid to follow and
    stops it when the command ends, it carries the job's wall budget (`seconds`, 0 for none), and
    its output goes nowhere rather than into the job's log.

    root: the workspace root on the host, where the staged credential lives.
    """
    if interval <= 0:
        return None
    budget = ("--seconds", f"{seconds:g}") if seconds else ()
    return ToolCall(
        args=("sample", stream, "--job", job, "--interval", f"{interval:g}", *budget),
        credentials=host_env(root),
    )
