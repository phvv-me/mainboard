# The live lane: this machine's own readings, published into a job's receipts while it runs.
#
# What makes these readings worth shipping is the one number a hosted dashboard never had. A
# service watching a job sees how much memory the process used; it does not see the ceiling the
# scheduler wrote onto the job's cgroup, which is the number an OOM kill actually fires against.
# `mainboard` probes that ceiling, so every sample carries used memory, the enforced cap, and the
# fraction between them, which is the series that says whether a job is about to die.
#
# The samples go to the same bus every other receipt goes to, so they land in the job's own
# NDJSON first and reach whatever the workspace declared second. That is what lets this run
# unchanged on a laptop, on gold, and on a compute node with no route out.

import shlex
from threading import Event as Flag
from threading import Thread
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import psutil

from ..batch.receipts import Topic, publish
from ..core.project import Project
from ..probe.machine import Machine

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from pydantic import JsonValue

    from ..batch.receipts import Bus, Event

# How long a stop waits for the sampling thread to notice, past which the thread is a daemon and
# the process may leave without it.
_JOIN_S = 5.0

# The file a host keeps the one tracking credential in, written by whoever dispatches and read
# by the job itself. Its own file rather than the workspace `.env`, so staging it can never
# overwrite what that host already declares, and one known path to audit, rotate or delete.
_HOST_ENV = "tracking.env"


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

    Entering starts the thread and takes the first reading at once, so even a job that dies in
    its first minute leaves a series. Leaving stops it. An interval of zero samples nothing at
    all, which is how a workspace turns the lane off without the caller branching on it.

    The thread is a daemon and every stop is bounded, because this is code that runs beside
    somebody's training loop and must never be the reason a job will not exit.
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

        stream: the receipts stream the samples belong to.
        job: the job inside that stream these readings describe.
        interval: seconds between readings, 0 to sample nothing.
        seconds: a hard stop, 0 to sample until the caller stops it.
        parent: a process to outlive rather than outlast, 0 for none. A sampler started beside a
            dispatched command watches that command's shell, so it ends when the job does
            instead of surviving it as an orphan.
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

    def loop(self) -> None:
        """Sample now, then every interval, until stopped, expired, or orphaned."""
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
        """Publish one reading as this job's next `job.sample` receipt."""
        return publish(self.bus, self.stream, Topic.SAMPLE, job=self.job, data=self.reading())

    def start(self) -> None:
        """Begin sampling, unless the declared interval asked for no samples at all."""
        self.opened = monotonic()
        if self.interval > 0:
            self.thread.start()

    def stop(self) -> None:
        """Stop sampling and wait a bounded moment for the thread to notice."""
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=_JOIN_S)


def host_env(root: str) -> str:
    """Where a host keeps the tracking credential a dispatched job reads.

    root: the workspace root on that host.
    """
    return f"{root}/{Project().out_dir}/{_HOST_ENV}"


def sampling_line(
    *, root: str, stream: str, job: str, interval: float, seconds: float = 0.0
) -> str:
    """The shell line a dispatched job runs so it samples itself, empty when it should not.

    This is the seam that carries the live lane onto a machine that is not this one. A job
    script starts the tool the host already has, in the environment the script already
    activated, and the sampler publishes into that host's own receipts and onward to whatever
    the workspace declared. Nothing about it is configured on the host.

    Three things keep it from outliving its job. It follows the job script's own process, so it
    ends when the job ends however the job ends; it carries the same wall budget the job was
    given; and its output goes nowhere, since a sampler must never write into a captured log the
    job's own output belongs in.

    root: the workspace root on the host, where the staged credential lives.
    stream: the receipts stream the samples belong to.
    job: the job inside that stream.
    interval: seconds between readings, 0 for a job that samples nothing.
    seconds: the job's own wall budget as a hard stop, 0 for none.
    """
    if interval <= 0:
        return ""
    sample = shlex.join(
        [
            Project().name,
            "sample",
            stream,
            "--job",
            job,
            "--interval",
            f"{interval:g}",
            *(("--seconds", f"{seconds:g}") if seconds else ()),
        ]
    )
    staged = shlex.quote(host_env(root))
    return (
        f"( set -a; . {staged} 2>/dev/null; set +a; "
        f'exec {sample} --parent "$$" ) >/dev/null 2>&1 &'
    )
