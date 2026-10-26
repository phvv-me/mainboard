"""A dispatched run as an object: `Job` on an ssh-family host, `ProviderJob` on a rental."""

import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from ..dispatch.backends.base import Delivery, LogSource, ProviderBackend, Rentable
from ..dispatch.constants import POLL_SECONDS
from ..dispatch.dispatcher import Handle, Verdict
from ..dispatch.hosts.rentals import identity
from ..dispatch.jobs import vocabulary
from ..dispatch.schedulers import HostUnreachable, failure_reason, read_log, registry
from ..dispatch.transport.core import SshTransport
from ..dispatch.transport.targets import home_of, placed
from ..dispatch.wrapping import connection
from ..log import logger

if TYPE_CHECKING:
    from ..dispatch.jobs.vocabulary import JobState
    from ..dispatch.schedulers import Scheduler
    from .board.workspace import Workspace


class Job:
    """One dispatched run, addressed as an object instead of handle flags."""

    @property
    def scheduler(self) -> Scheduler:
        """The backend selected on the kind this run was dispatched under, not today's profile.

        Every other probe follows the recorded kind too (`Dispatcher.state`, `Dispatcher.states`,
        the sweep's grouping); a host whose declared kind changed under a live job would
        otherwise have that job killed through a scheduler that never took it.
        """
        return registry.schedulers.select(self.handle.kind, default="ssh")

    @staticmethod
    def states(handles: Sequence[Handle]) -> dict[str, JobState]:
        """Every handle's state, keyed by id, over one connection to the host they share.

        The batched `poll`: a cache months old holds a thousand runs on one box, and asking it
        once rather than a thousand times is what lets a sweep finish. One connection and one
        scheduler answer for all, so every handle must share the first one's host, root and
        kind. A handle the listing misses (a SLURM `squeue` spanning only live jobs, a PBS server
        that purged its history) is asked alone over the same connection, where its real ending
        is. `HostUnreachable` surfaces as from `poll`: it is about the host, not any one job.
        """
        if not handles:
            return {}
        shared = handles[0]
        scheduler = registry.schedulers.select(shared.kind, default="ssh")
        ids = list(dict.fromkeys(handle.id for handle in handles))
        with connection(shared.host, attempts=1) as remote:
            listed = scheduler.states(remote, shared.root, ids)
            return {
                job_id: listed.get(job_id) or scheduler.state(remote, shared.root, handle=job_id)
                for job_id in ids
            }

    def __init__(self, board: Workspace, handle: Handle) -> None:
        """board: the host-bound board that submitted this job."""
        self.board = board
        self.handle = handle

    def _verdict(self, state: JobState) -> Verdict:
        """Persist a terminal state and project it onto a `Verdict`.

        Only a job not ending `ok` has its log read for a reason, so a clean run pays no second
        round trip.
        """
        cache = self.board.dispatcher.cache
        with suppress(LookupError):
            run = cache.run(self.handle.id, target=self.handle.host)
            cache.resolve(run, state.state, state.exit_code, state.verdict)
        if state.verdict == "ok":
            return Verdict(verdict="ok", exit_code=state.exit_code)
        with connection(self.handle.host) as remote:
            log = read_log(remote, self.handle.root, handle=self.handle.id)
        return Verdict(
            verdict=state.verdict,
            exit_code=state.exit_code,
            reason=failure_reason(log, state.exit_code),
        )

    def kill(self) -> None:
        """Cancel the job on its scheduler."""
        with connection(self.handle.host) as remote:
            self.scheduler.cancel(remote, self.handle.root, handle=self.handle.id)

    def logs(self) -> str:
        """The job's captured log so far, merged stdout and stderr."""
        with connection(self.handle.host) as remote:
            return self.scheduler.logs(remote, self.handle.root, handle=self.handle.id)

    def poll(self) -> JobState:
        """The job's state now, raising `HostUnreachable` when its host will not answer.

        A durable sweep has to say which host went quiet rather than quietly try again; `state`
        is the same probe with the blip absorbed.
        """
        with connection(self.handle.host, attempts=1) as remote:
            return self.scheduler.state(remote, self.handle.root, handle=self.handle.id)

    def pull(self) -> None:
        """Bring the job's recorded results path back to this machine."""
        if not self.handle.fetch_path:
            raise LookupError(f"handle {self.handle.id!r} has no fetch path to pull")
        self.board.dispatcher.fetch_path(
            self.handle.host, root=self.handle.root, path=self.handle.fetch_path
        )

    def release(self) -> None:
        """Nothing to let go of: a queue stops charging when the job ends, so no kill is sent.

        The verb exists because a provider run bills until cancelled, and a sweep settling
        either kind says the same thing to both.
        """

    def state(self) -> JobState | None:
        """One non-blocking probe, None when the host could not be reached on this tick.

        None is a reason to look again rather than a verdict; `wait` is the blocking loop.
        """
        try:
            return self.poll()
        except HostUnreachable as down:
            logger.warning("{} unreachable, retrying: {}", self.handle.id, down)
            return None

    def transcript(self) -> str | None:
        """The tolerant twin of `logs`: None when the host will not answer.

        A settle tells that from an empty log, so it waits rather than records a log it never read.

        A settle wants the output if it can have it and must never fail the sweep over a host
        that went quiet between the probe and the read.
        """
        try:
            return self.logs()
        except (HostUnreachable, MissionError, OSError) as quiet:
            logger.warning("no transcript for {}: {}", self.handle.id, quiet)
            return None

    def wait(self, *, interval: float = POLL_SECONDS) -> Verdict:
        """Poll every `interval` seconds until the job is terminal, and answer its verdict.

        A probe the host did not answer is a blip, retried on the next poll.
        """
        while (state := self.state()) is None or state.verdict == "running":
            time.sleep(interval)
        return self._verdict(state)


class ProviderJob:
    """One provider-dispatched run, the transport-free twin of `Job`.

    Only the lifecycle is on every backend. Logs and artifact delivery are capabilities, asked
    for by contract first and refused with the backend's own advice when it has none.
    """

    def __init__(self, board: Workspace, backend: ProviderBackend, handle: Handle) -> None:
        """board: the workspace bound to the provider host.

        handle: the dispatch handle carrying the provider's opaque run id.
        """
        self.board = board
        self.backend = backend
        self.handle = handle

    def _collected(self, backend: Rentable, *, path: str) -> None:
        """Collect `path` from the rented machine over ssh, its `~` root placed under its home."""
        profile = self.board.plan().profile
        key = identity(profile.vars.get("ssh-key", "")).private
        endpoint = backend.endpoint(self.handle.id, key=key)
        policy = SshTransport(endpoint=endpoint)
        root = self.handle.root or profile.root
        if root.startswith("~"):
            with connection(endpoint.destination, policy) as remote:
                root = placed(root, home=home_of(remote))
        self.board.dispatcher.fetch_path(endpoint.destination, root=root, path=path, ssh=policy)

    def kill(self) -> None:
        """Cancel the run on the provider."""
        self.backend.cancel(self.handle.id)

    def logs(self) -> str:
        """The run's captured log so far, refusing when this provider keeps none."""
        if not isinstance(self.backend, LogSource):
            raise MissionError(self.backend.refusal(LogSource, handle=self.handle.id))
        return self.backend.logs(self.handle.id)

    def poll(self) -> JobState:
        """The run's state now, as the provider reports it."""
        return self.backend.state(self.handle.id)

    def pull(self) -> None:
        """Bring the handle's recorded results path back, refusing when this provider cannot.

        The same no-argument verb as `Job.pull`, so one sweep pulls either kind of run.
        """
        path = self.handle.fetch_path
        if not path:
            raise LookupError(f"handle {self.handle.id!r} has no fetch path to pull")
        if isinstance(self.backend, Delivery):
            self.backend.deliver(self.handle.id, path=path)
        elif isinstance(self.backend, Rentable):
            self._collected(self.backend, path=path)
        else:
            raise MissionError(self.backend.refusal(Delivery, handle=self.handle.id, path=path))

    def release(self) -> None:
        """End the rental, the only thing that stops a provider charging for it.

        A finished command does not end a provider run. Vast holds the instance and restarts the
        exited container until cancelled (thirteen re-runs in five minutes, verified live
        2026-08-19), and an HPC-AI instance runs until terminated.
        """
        self.backend.cancel(self.handle.id)

    def transcript(self) -> str | None:
        """The run's captured output, None when the provider keeps one it would not hand over.

        It is empty when this provider keeps none.

        A rented machine's disk dies with the rental, so logs and artifacts must both be read
        before release, and a log that failed to transfer is no empty log (B03).
        """
        if not isinstance(self.backend, LogSource):
            return ""
        try:
            return self.backend.logs(self.handle.id)
        except (MissionError, OSError) as quiet:
            logger.warning("no transcript for {}: {}", self.handle.id, quiet)
            return None

    def wait(
        self, *, interval: float = 15.0, poll: Callable[[float], None] = time.sleep
    ) -> Verdict:
        """Poll the computational verdict; the durable monitor owns evidence and release.

        interval: seconds between provider state polls.
        poll: the sleeper between polls, injectable for tests.
        """
        while True:
            state = self.backend.state(self.handle.id)
            if state.verdict in vocabulary.Word.settled():
                return Verdict(verdict=state.verdict, exit_code=state.exit_code)
            poll(interval)


# A dispatched run from either world. Both answer `poll`, `pull` and `release` alike, so one
# durable sweep settles a queued job and a rented instance without asking which it holds.
type Run = Job | ProviderJob
