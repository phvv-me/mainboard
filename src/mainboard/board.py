import os
import platform
import shlex
import time
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import TYPE_CHECKING, NoReturn, cast

from plumbum import ProcessExecutionError
from plumbum import local as localhost

from .batch.estimate import Estimator, JobEstimate
from .batch.receipts import Receipts, Topic, publish
from .batch.runner import Batch, directory
from .batch.spec import BatchJob, Selection
from .batch.transfer import TransferSet
from .batch.watch import Watch
from .compute import Survey
from .context.admission import admit
from .context.expressions import evaluate
from .context.resolver import Resolver
from .core.errors import MissionError
from .core.project import Project
from .core.shell import foreground
from .deps import Dependencies
from .dispatch import vocabulary
from .dispatch.backends.base import (
    Credentials,
    Delivery,
    LogSource,
    ProviderBackend,
    Rentable,
    route,
)
from .dispatch.commandline import joined, vetted
from .dispatch.dispatcher import Dispatcher, Handle, Verdict
from .dispatch.jobs.spec import walltime_seconds
from .dispatch.landing import Landing, renter
from .dispatch.onboard import HostSetup, Onboarding, facts_command, read_facts
from .dispatch.rentals import identity
from .dispatch.schedulers import HostUnreachable, pick, registry
from .dispatch.shared import logger
from .dispatch.shipment import Shipment
from .dispatch.snapshots import Snapshots
from .dispatch.targets import find_root
from .dispatch.transport import SshTransport
from .dispatch.vocabulary import Request, Resources
from .dispatch.wrapping import connection, missing, wrap
from .doctor import Doctor
from .engines.compile.backend import PIXI_VERSION
from .engines.compile.pixi_manifest import self_installed
from .engines.compile.prefixes import MANIFEST, Prefixes, digest_of, prefix_path
from .engines.compile.provisioner import Provisioner, environment_shard, task_line
from .engines.compile.state import SyncState
from .engines.runtimes import resolve
from .experiments.fleet import Fleet
from .experiments.identity import run_id
from .jobs.closure import Closure
from .jobs.target import Target
from .manifest.loading import load
from .monitor import Monitor
from .nodes import evidence_of
from .probe.snapshot import HostFacts
from .scaffold import Scaffold
from .tracking import (
    Sampler,
    attesting_line,
    credential,
    host_env,
    is_batched,
    mirrored,
    sampling_line,
    streamed,
)
from .verdicts import Verdicts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .batch.receipts import Bus
    from .batch.spec import BatchSpec
    from .context.plan import ExecutionPlan
    from .dispatch.schedulers import Scheduler
    from .dispatch.shared import Watcher
    from .dispatch.vocabulary import JobState
    from .manifest.schema.root import Manifest

# `route`'s answer for the schedulers reached over ssh, the family whose hosts run the work
# themselves rather than renting an instance to run it on.
_SSH_FAMILY = "ssh-family"


class Job:
    """One dispatched run, addressed as an object instead of handle flags."""

    def __init__(self, board: Board, handle: Handle) -> None:
        """board: the host-bound board that submitted this job.

        handle: the dispatch handle identifying it on the scheduler.
        """
        self.board = board
        self.handle = handle

    @property
    def scheduler(self) -> Scheduler:
        """The backend that answers for this run, selected on the kind it was dispatched under.

        The recorded kind rather than whatever the host's profile says today, which is the rule
        every other probe already follows (`Dispatcher.state`, `Dispatcher.states`, and the
        sweep's own grouping). A host whose declared kind changed under a live job would
        otherwise have that job killed through a scheduler that never took it.
        """
        return registry.SCHEDULERS.select(self.handle.kind, default="ssh")

    def kill(self) -> None:
        """Cancel the job on its scheduler."""
        with connection(self.handle.host) as remote:
            self.scheduler.cancel(remote, self.handle.root, handle=self.handle.id)

    def logs(self) -> str:
        """The job's captured log so far, merged stdout and stderr."""
        with connection(self.handle.host) as remote:
            return self.scheduler.logs(remote, self.handle.root, handle=self.handle.id)

    def transcript(self) -> str:
        """The run's captured output, empty when this backend keeps none or will not answer.

        The tolerant twin of `logs`, for a settle that wants the output if it can have it and
        must never fail the sweep over a host that went quiet between the probe and the read.
        """
        try:
            return self.logs()
        except (HostUnreachable, MissionError, OSError, ProcessExecutionError) as quiet:
            logger.warning("no transcript for %s: %s", self.handle.id, quiet)
            return ""

    def poll(self) -> JobState:
        """The job's state now, raising `HostUnreachable` when its host will not answer.

        The unabsorbed read a durable sweep wants, since a sweep has to say which host went
        quiet rather than quietly try again, and `state` is this same probe with the blip
        absorbed for a caller polling on its own cadence.
        """
        return self.board.dispatcher.state(self.handle)

    def pull(self) -> None:
        """Bring the job's recorded results path back to this machine."""
        self.board.dispatcher.fetch(self.handle)

    def release(self) -> None:
        """Let go of whatever a settled job still holds, which for a scheduler is nothing.

        A queue stops charging when the job ends, so a finished pueue or PBS job needs no kill
        and never gets one. The verb exists because a provider-backed run keeps billing until it
        is cancelled, and a sweep settling either kind says the same thing to both.
        """

    def state(self) -> JobState | None:
        """One non-blocking probe of the job's current scheduler state.

        None when the host could not be reached on this tick, which is a reason to look again
        rather than a verdict; `wait` is the same probe under a blocking loop.
        """
        return self.board.dispatcher.probe(self.handle)

    def wait(self, *, interval: float | None = None) -> Verdict:
        """Block until the job is terminal and return its verdict."""
        extra = {"interval": interval} if interval is not None else {}
        return self.board.dispatcher.await_many([self.handle], **extra)[self.handle]


class ProviderJob:
    """One provider-dispatched run, the transport-free twin of `Job`.

    Only the lifecycle is guaranteed here, since only the lifecycle is on every backend. Logs and
    artifact delivery are capabilities, so each is asked for by contract first and refuses with
    the backend's own advice when that backend never had one.
    """

    def __init__(self, board: Board, backend: ProviderBackend, handle: Handle) -> None:
        """board: the workspace bound to the provider host.

        backend: the provider backend instance that submitted this run.

        handle: the dispatch handle carrying the provider's opaque run id.
        """
        self.board = board
        self.backend = backend
        self.handle = handle

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

    def transcript(self) -> str:
        """The run's captured output, empty when this provider keeps none or will not answer.

        A rented machine's disk dies with the rental, so logs and artifacts must both be read
        before release. A provider without logs answers an empty string rather than a refusal.
        """
        if not isinstance(self.backend, LogSource):
            return ""
        try:
            return self.backend.logs(self.handle.id)
        except (MissionError, OSError) as quiet:
            logger.warning("no transcript for %s: %s", self.handle.id, quiet)
            return ""

    def pull(self) -> None:
        """Bring the run's recorded results path back, refusing when this provider cannot.

        The same no-argument verb the scheduler side carries, reading the path off the handle
        the dispatch recorded, so one sweep pulls either kind of run without asking which it has.
        """
        path = self.handle.fetch_path
        if not path:
            raise LookupError(f"handle {self.handle.id!r} has no fetch path to pull")
        if isinstance(self.backend, Delivery):
            self.backend.deliver(self.handle.id, path=path)
        elif isinstance(self.backend, Rentable):
            profile = self.board.plan().profile
            key = identity(profile.vars.get("ssh-key", "")).private
            endpoint = self.backend.endpoint(self.handle.id, key=key)
            policy = SshTransport(endpoint=endpoint)
            root = self.handle.root or profile.root
            if not root:
                with connection(endpoint.destination, policy) as remote:
                    root = find_root(remote)
            self.board.dispatcher.fetch_path(
                endpoint.destination, root=root, path=path, ssh=policy
            )
        else:
            raise MissionError(self.backend.refusal(Delivery, handle=self.handle.id, path=path))

    def release(self) -> None:
        """End the rental, which is the only thing that stops a provider charging for it.

        A finished command does not end a provider run. Vast holds the instance at its intended
        status and restarts the exited container until someone cancels (thirteen re-runs in five
        minutes, verified live 2026-08-19), and an HPC-AI instance keeps running until it is
        terminated, so a settled verdict has to be followed by this or the meter never stops.
        """
        self.backend.cancel(self.handle.id)

    def wait(
        self, *, interval: float = 15.0, poll: Callable[[float], None] = time.sleep
    ) -> Verdict:
        """Poll the computational verdict; the durable monitor owns evidence and release.

        interval: seconds between provider state polls.
        poll: the sleeper between polls, injectable for tests.
        """
        while True:
            state = self.backend.state(self.handle.id)
            if state.verdict in vocabulary.TERMINAL:
                return Verdict(verdict=state.verdict, exit_code=state.exit_code)
            poll(interval)


# A dispatched run, whichever of the two worlds took it. Both shapes answer `poll`, `pull` and
# `release` the same way, which is what lets one durable sweep settle a queued job and a rented
# instance without asking which it is holding.
type Run = Job | ProviderJob


class Board:
    """The one addressable interface: a workspace, pivoted onto a host by `on`.

    `Board()` finds the manifest like git finds a repository. The unbound
    board is this machine; `board.on("gold")` is the same board bound to a
    declared host, where `run`, `submit`, and `facts` keep the same shapes
    while the profile decides scheduler, environment, container, and queue
    policy. The composed subsystems stay public for anything the facade does
    not carry.
    """

    def __init__(self, root: Path | None = None, *, host: str = "local") -> None:
        """root: the workspace root, discovered upward from the cwd when None.

        host: the host alias this board is bound to, `local` for here.
        """
        self.project = Project()
        self.root = root or self.project.find_root(Path.cwd())
        self.host = host
        self.shared: dict[str, object] = {}
        self.guard = RLock()

    @property
    def dispatcher(self) -> Dispatcher:
        """The dispatch core, rooted at this workspace and shared across every host pivot.

        Rooted rather than left to the working directory, so a command typed in a subdirectory
        reads the same run registry, stages into the same jobs directory and mirrors the same
        tree as one typed at the root.
        """
        return self.once("dispatcher", lambda: Dispatcher(root=self.root))

    @property
    def local(self) -> bool:
        """Whether this board is bound to the current machine."""
        return self.host == "local"

    @property
    def manifest(self) -> Manifest:
        """The loaded workspace manifest, shared across every `on` pivot."""
        return self.once("manifest", lambda: load(self.root / self.project.manifest))

    @property
    def resolver(self) -> Resolver:
        """The plan resolver over this workspace's manifest."""
        return self.once("resolver", lambda: Resolver(self.manifest))

    def announce(self, label: str, run: Run, *, command: str, host: str, node: str = "") -> None:
        """Open this run's own receipts stream, so a dispatch outside a batch is tracked too.

        A batch publishes its own submissions and is skipped here, so no line is written twice.

        label: the run's dispatch label, which says both where it belongs and who publishes it.
        run: the dispatched run, for the handle its stream is keyed on.
        command: what the job runs, recorded as this run's config.
        host: the target it was dispatched to.
        node: the ledger slug the run serves, carried on the line only when one was declared.
        """
        if is_batched(label) or not self.manifest.tracking.on:
            return
        stream, job = streamed(label, handle=run.handle.id)
        publish(
            self.receipts(stream),
            stream,
            Topic.SUBMITTED,
            job=job,
            data={
                "handle": run.handle.id,
                "target": host,
                "kind": run.handle.kind,
                "command": command,
                **({"node": node} if node else {}),
            },
        )

    def attest(self, stream: str, *, job: str) -> None:
        """Publish one attestation of this machine into `stream`'s receipts and return.

        The synchronous, once-only twin of `samples`. It reads the machine it is called on, so a
        dispatched job runs it on the node that will do the work rather than on the one that
        dispatched it, which is the only reading that describes the measurement's conditions.

        stream: the receipts stream the attestation belongs to.
        job: the job inside that stream this reading describes.
        """
        Sampler(self.receipts(stream), stream=stream, job=job, interval=0.0).attest()

    def attesting(self, tracked: tuple[str, str], *, root: str) -> str:
        """The line this job's script runs to attest to its own machine, empty when none does.

        A sibling of `sampling`, gated on the same declaration, since both are the workspace's
        tracking lane reaching a host and neither is worth staging on a workspace that tracks
        nothing. Unlike the sampler this one carries no interval, because an attestation happens
        exactly once and its whole value is that it happens before the work.

        tracked: the stream and job the attestation belongs to.
        root: the workspace root on the host.
        """
        if not self.manifest.tracking.on:
            return ""
        stream, job = tracked
        return attesting_line(root=root, stream=stream, job=job)

    def batch(self, spec: BatchSpec, *, selection: Selection | None = None) -> Batch:
        """The declared batch over this workspace, ready to prepare, price and dispatch.

        Host-independent like `monitor`, since a batch names a target per job and fans across
        the fleet rather than running on whichever host a board happens to be bound to.

        spec: the declared batch.
        selection: which of the plan's jobs to act on, all of them when None. The batch keeps its
            identity and its receipts stream either way, so a plan sent out in waves is one batch.
        """
        return Batch(self, spec, bus=self.receipts(spec.batch_id), selection=selection)

    def compute(self) -> Survey:
        """The survey of every compute path this workspace can reach, this machine included.

        Host-independent like `monitor`, since one pass covers the whole fleet at once; a board
        bound to a host hands back the same whole-workspace survey an unbound one does.
        """
        return Survey(self)

    def containerizer(
        self, plan: ExecutionPlan, root: str
    ) -> Callable[[list[str]], list[str]] | None:
        """The container argv builder for `plan`, None when the plan is bare."""
        if not plan.containerized or plan.container is None:
            return None
        runtime_name = plan.container.runtime
        if runtime_name == "auto" and not self.local:
            modules = plan.profile.modules
            runtime_name = (
                "apptainer" if "apptainer" in modules or "singularity" in modules else "docker"
            )
        runtime = resolve(runtime_name)()
        container = plan.container
        return lambda argv: runtime.command(container, prefix_bind=plan.prefix(root), argv=argv)

    def deps(self) -> Dependencies:
        """The manifest's declared requirements, editable and re-solvable from here.

        Host-independent like `monitor` and `compute`, since a dependency belongs to the
        workspace rather than to whichever machine happens to install it.
        """
        return Dependencies(self)

    def doctor(self, env: str = "") -> Doctor:
        """One verdict over this workspace and one resolved environment.

        env: the environment name, the bound host profile's own when empty.
        """
        return Doctor(self, env=env)

    def expectation(
        self,
        command: str,
        *,
        queue: str = "",
        walltime: str = "",
        mem_gb: int = 0,
        gpus: int = 0,
        gpu_name: str = "",
        max_usd: float = 0.0,
        attempt: int = 1,
    ) -> JobEstimate:
        """What one submit on this host is expected to cost, admitted and priced before dispatch.

        The same resource resolution `submit` runs, then the queue policy check a dispatch
        would enforce anyway, so a request the policy refuses dies here in one sentence rather
        than after an ssh round trip. The price is the estimator's, a provider's metered rate
        for a rented host and zero for hardware this workspace owns, with the declared walltime
        standing in for the runtime the way a batch spec's `runtime_s` does. Nothing connects,
        nothing rents, nothing dispatches.

        command: the command the submit would run.
        """
        plan = self.plan()
        resources = self.resources(
            queue=queue,
            walltime=walltime,
            mem_gb=mem_gb,
            gpus=gpus,
            gpu_name=gpu_name,
            max_usd=max_usd,
            attempt=attempt,
            plan=plan,
        )
        admit(
            plan.profile,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
        )
        job = BatchJob(
            name=self.host,
            target=self.host,
            command=command,
            runtime_s=walltime_seconds(resources.walltime) if resources.walltime else 0.0,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=resources.mem_gb or 0,
            gpus=resources.gpus,
            gpu_name=resources.gpu_name,
            max_usd=resources.max_usd,
        )
        return Estimator(self).row(job, TransferSet(job=self.host, target=self.host))

    def facts(self) -> HostFacts:
        """The host's probed hardware facts as the versioned wire snapshot.

        A remote host answers with its own installed tool, the one `install` puts there, so the
        probe never depends on this workspace's mainboard being importable by whatever
        interpreter the host happens to ship.
        """
        if self.local:
            return HostFacts.collected()
        line = wrap(self.plan(container="none"), self.remote_root(), command=facts_command())
        with connection(self.host) as remote:
            reply = remote["bash"]["-lc", line]()
        return read_facts(str(reply))

    @property
    def floor(self) -> str:
        """The version this workspace declares for the tool itself, empty when it declares none.

        A workspace that vendors the tool's source has the source and needs no version. One that
        consumes it from an index says which one it needs in the same place it says everything
        else it depends on, so a host with no vendored source installs exactly that.
        """
        declared = self.manifest.requirement(self.project.name)
        return declared.version if declared is not None else ""

    def fleet(self) -> Fleet:
        """The many-jobs surface for simultaneous studies over this board."""
        return Fleet(self)

    def install(
        self,
        env: str = "",
        *,
        resolve: bool = False,
        profile: str = "",
        watch: Watcher | None = None,
        sync_only: bool = False,
    ) -> HostSetup:
        """Install an environment for this board's host, in place here or by onboarding over ssh.

        An unbound board installs on this machine. A board bound to a host alias runs the whole
        onboarding there instead, mirroring the workspace, installing the tool from that mirror,
        provisioning the environment with the host's own tool, and probing what it became.

        Which environment that is comes from the same resolver every other verb uses, so an
        empty `env` means the host profile's declared choice rather than a hardcoded `default`.
        Setting a host up therefore installs what the manifest already says that host runs, and
        naming an environment stays the override it always was.

        `resolve` means the same thing on both sides: this workspace may solve. A host is sent
        the artifact this workspace already solved and installs from it, so onboarding never
        puts a host's own compiler in the lock's dependency path.

        env: the environment name, the host profile's own when empty.
        resolve: allow a fresh dependency solve, refused otherwise when the lock cannot vouch
            for what is on disk. For a host it means solving there instead of installing the
            shipped artifact.
        profile: the declared host profile describing this machine, so the generated activation
            carries that host's module stack; this board's own host when empty.
        watch: announces each onboarding stage as it begins.
        sync_only: re-mirror and re-provision an already onboarded host without reinstalling
            the tool or re-probing its hardware, neither of which changed when only the
            manifest moved; refused on this machine, which has no onboarding to skip parts of.
        """
        if sync_only and self.local:
            raise MissionError(
                "--sync-only onboards a remote host faster; this machine has no onboarding to "
                "shortcut, run `install` instead"
            )
        plan = self.resolver.plan(profile or self.host, env=env, container="none")
        provisioner = Provisioner(self.root, self.manifest)
        if not self.local:
            if not resolve:
                # The host will refuse a lock this manifest did not solve; ask here first,
                # before the mirror and the remote install spend minutes reaching that answer.
                provisioner.compiler_for(plan.env).vouch()
            return Onboarding(
                self.dispatcher,
                plan,
                root=plan.profile.root,
                artifact=provisioner.artifact_for(plan.env),
                resolve=resolve,
                watch=watch,
                digest=provisioner.compiler_for(plan.env).digest(),
                floor=self.floor,
            ).run(sync_only=sync_only)
        provisioner.provision(plan.env, resolve=resolve)
        return HostSetup(
            host=self.host,
            root=str(self.root),
            env=plan.env,
            activate=self.activation(provisioner, plan),
            installer="in-place",
            tool=version(self.project.name),
        )

    def activation(self, provisioner: Provisioner, plan: ExecutionPlan) -> str:
        """Write the shell script a bare shell activates this environment from, where one runs.

        `activate.sh` is bash by construction, and nothing on Windows sources it: writing one
        there hands the reader a script their own shell cannot run, with a PATH built for a
        different world. A Windows workspace activates through the activation pixi cached when
        it was provisioned, so this says so by writing nothing and naming nothing.

        provisioner: the provisioner that has just installed the environment.
        plan: the resolved execution context, whose profile carries this host's module stack.
        """
        if platform.system() == "Windows":
            return ""
        return str(provisioner.activate(plan.env, modules=plan.profile.modules))

    def interact(
        self,
        *command: str,
        env: str = "",
        queue: str = "",
        walltime: str = "",
        keep: bool = False,
        replace: Callable[[str, list[str]], NoReturn] = os.execvp,
    ) -> NoReturn:
        """Hand this terminal a session on the bound host, inside its mirrored workspace.

        The counterpart of `shell` for a machine that is not this one, and the verb that ends
        the habit of ssh'ing in by hand and retyping the `cd` and the queue flags. This process
        is replaced by the ssh rather than wrapping it, so the session owns the terminal and
        every signal reaching it, and leaving the session lands back where the user started.

        Each scheduler decides what a session is on its own host, since the answer genuinely
        differs. An ssh box is already the machine the work runs on, so its own tool takes the
        terminal, while a queued cluster must be asked for an allocation first and hands the
        terminal to a compute node. The staging around either is the `cd`, `PATH` and modules
        every other remote command gets, and nothing more, because whatever answers on the far
        side owns the activation.

        A kept session runs inside a tmux session on the far side, named for this workspace
        and host, so the terminal can drop and the allocation stays up on the cluster; asking
        again with `keep` reattaches to it instead of asking the scheduler for another node.

        command: a command to run instead of handing over the terminal, its own flags included.
        env: an environment name overriding the profile's choice.
        queue: the queue the allocation targets, the profile's own when empty.
        walltime: the session's wall-clock limit, the profile's own when empty.
        keep: hold the session in tmux on the far side and reattach to one already held.
        replace: the process-replacing exec, injectable so a test can read the argv it built.
        """
        if self.local:
            raise MissionError(
                f"an interactive session needs a host. Run `{self.project.name} shell` for "
                "this machine."
            )
        plan = self.plan(env=env, container="none")
        if route(plan.profile.kind) != _SSH_FAMILY:
            raise MissionError(
                f"host {self.host!r} rents instances through {plan.profile.kind!r} and hands "
                f"out no terminal. Run `{self.project.name} submit --on {self.host}` instead."
            )
        defaults = plan.profile.defaults
        resources = Resources(
            queue=queue or defaults.interact_queue or defaults.queue,
            walltime=walltime or defaults.walltime,
            gpus=defaults.gpus,
            account=plan.profile.account,
        )
        admit(
            plan.profile,
            queue=resources.queue or "",
            walltime=resources.walltime or "",
            mem_gb=0,
        )
        session = pick(plan.profile).interactive(
            env=plan.env, command=command, resources=resources
        )
        staged = wrap(plan, self.remote_root(), command=session, activate=False)
        if keep:
            # `new-session -A` attaches to the named session when it exists and only otherwise
            # starts one, so the same verb both opens and returns to a held allocation.
            held = f"{self.project.name}-{self.host}"
            staged = f"tmux new-session -A -s {shlex.quote(held)} {shlex.quote(staged)}"
        # A bounded transport is what a poll wants and the opposite of what a session wants, so
        # the user's own ssh config owns this one connection. `-t` forces the pty the far side
        # needs, and the staged line is quoted whole because ssh joins its argv back into one
        # string for the remote login shell to parse.
        replace("ssh", ["ssh", "-t", self.host, f"bash -lc {shlex.quote(staged)}"])

    def job(self, handle: str | int, *, host: str = "") -> Run:
        """The dispatched run `handle`, rebuilt from the dispatch cache as whichever kind it is.

        A fresh process addresses an already-running job the same way the process that
        submitted it did, without reassembling a `Handle` from the run registry and the host
        profile by hand. The kind the cache recorded decides which world it comes back from, a
        scheduler job bound to its host's workspace or a provider run bound to its backend, so a
        rental outlives the process that started it exactly as a queued job does.

        handle: the scheduler handle or provider run id the job was dispatched under.
        host: the alias to disambiguate a handle recorded on several hosts.
        """
        record = self.dispatcher.cache.run(str(handle), host or None)
        if record.verdict in {vocabulary.PREPARED, vocabulary.SUBMITTING}:
            action = (
                "no create attempted; cancel if abandoned"
                if record.verdict == vocabulary.PREPARED
                else "inspect the provider by this label before retrying or cancelling"
            )
            raise MissionError(
                f"creation {record.creation} has no confirmed provider handle; {action}"
            )
        destination = route(record.kind)
        if destination != "ssh-family":
            return ProviderJob(
                self.on(record.target),
                destination(),
                Handle(
                    id=record.handle,
                    host=record.target,
                    root="",
                    kind=record.kind,
                    fetch_path=record.fetch_path,
                ),
            )
        bound = self.on(record.target)
        return Job(
            bound,
            Handle(
                id=record.handle,
                host=record.target,
                root=bound.remote_root(),
                kind=record.kind,
                fetch_path=record.fetch_path,
            ),
        )

    def line(self, command: str, *, env: str = "", container: str = "") -> str:
        """The staged shell line this board's host would run `command` through.

        The one place the staging is assembled, cd, PATH, modules, then the environment or the
        container, so a caller that wants the command's output rather than its exit code runs
        the very line `run` runs instead of restaging it a second way.

        command: the shell command, or a declared task name and its arguments.
        env: an environment name overriding the profile's choice.
        container: a container override, `none` forcing bare.
        """
        plan = self.plan(env=env, container=container)
        root = str(self.root) if self.local else self.remote_root()
        return wrap(
            plan,
            root,
            command=task_line(self.manifest, command, env=plan.env),
            containerize=self.containerizer(plan, root),
        )

    def monitor(self) -> Monitor:
        """The durable sweep over every job this workspace's dispatch cache still owes an outcome.

        Host-independent, since one pass covers every target at once; a board bound to a host
        hands back the same whole-workspace sweep an unbound one does.
        """
        return Monitor(self)

    def on(self, host: str) -> Board:
        """This workspace bound to `host`, sharing the loaded manifest and caches.

        host: a declared host alias, or any ssh-config alias for defaults.
        """
        bound = Board.__new__(Board)
        bound.project = self.project
        bound.root = self.root
        bound.host = host
        bound.shared = self.shared
        bound.guard = self.guard
        return bound

    def once[Built](self, key: str, build: Callable[[], Built]) -> Built:
        """The one `key` this workspace shares, built on first ask and never a second time.

        Under a lock, because the first ask routinely comes from a worker thread. A doctor
        report asks four questions at once and a survey probes a whole fleet in a pool, so two
        threads reaching an unbuilt subsystem together would each build one, and a second
        dispatch cache is a second SQLite connection owned by whichever thread happened to win.
        The lock is reentrant since one build reads another, a resolver needing the manifest.
        A build that raises is not remembered, so a manifest that will not parse is re-read and
        re-refused rather than answered from a half-filled cache. Emptying a slot is how a
        caller that swaps one shared value (a test rewriting the manifest) makes the values
        derived from it be built again.

        key: what is being shared.
        build: makes it, called once at most.
        """
        with self.guard:
            built = self.shared.get(key) or build()
            self.shared[key] = built
            return cast("Built", built)

    def plan(self, *, env: str = "", container: str = "") -> ExecutionPlan:
        """The resolved execution plan for this board's host."""
        return self.resolver.plan(self.host, env=env, container=container)

    def receipts(self, stream: str) -> Bus:
        """Where one stream's events go: this workspace's own file, plus whatever it declared.

        The composition root for tracking, here rather than inside any one flow, so a batch, a
        plain submit and a study all mirror the same way and none of them has to know that a
        reporting service exists. A workspace whose `[tracking]` table says `off` gets the file
        alone and every caller is unchanged.

        stream: the receipts stream, a batch id, a study id, or one run's own name.
        """
        under = directory(self, stream)
        return mirrored(
            Receipts(under / "events.ndjson"),
            self.manifest.tracking,
            stream=stream,
            directory=under,
            workspace=self.manifest.workspace.name,
        )

    def remote_root(self) -> str:
        """The declared workspace root on the bound host, refusing when absent."""
        root = self.plan().profile.root
        if not root:
            raise MissionError(
                f"host {self.host!r} declares no root; set [hosts.{self.host}] root"
            )
        return root

    def rented(
        self,
        backend: ProviderBackend,
        plan: ExecutionPlan,
        *,
        shipment: Shipment,
        resources: Resources,
        name: str = "",
        node: str = "",
        watch: Watcher | None = None,
    ) -> Handle:
        """Dispatch `shipment` and retain the rental before any workspace provisioning.

        A rental is set up the way a declared host is, so it is shipped the same artifact
        `install` ships gold: this workstation solved the lock and the machine installs frozen
        against it. The lock is asked to vouch for the manifest here, before the rental opens,
        since a refusal a minute later is a refusal that has already cost money. A plan that
        brings its own container skips all of it, since a prebuilt image already holds everything
        its command needs.

        backend: the provider backend this dispatch resolved to.
        plan: the resolved execution context for the provider host.
        shipment: what the job runs and ships. A job spelled by file needs a workspace to ship
            its closure into, so a plan whose image is the whole environment refuses it.
        resources: the resolved request, whose spend cap and walltime bound the rental.
        name: the label retained with the allocated handle.
        node: the research node served by the dispatch.
        watch: announces each landing stage as it begins, since a landing is minutes of mirror,
            install and provisioning that would otherwise stand silent on a metered box.
        """
        renting = renter(backend, plan)
        if renting is None:
            if shipment.sealed:
                raise MissionError(
                    f"{shipment.spelling} is a job spelled by file, and host {plan.host!r} "
                    "runs a prebuilt image that ships no workspace for its closure; run it as "
                    "a command inside that image, or on a host that mirrors the workspace"
                )
            allocation = self.dispatcher.allocating(
                plan, shipment, resources, name=name, node=node, evidence="pending"
            )
            try:
                handle = backend.submit(plan, shipment.command, resources, allocation=allocation)
            finally:
                allocation.interrupted()
            return Handle(
                id=handle,
                host=plan.host,
                root="",
                kind=plan.profile.kind,
                fetch_path=shipment.fetch or None,
            )
        provisioner = Provisioner(self.root, self.manifest)
        provisioner.compiler_for(plan.env).vouch()
        return Landing(
            self.dispatcher,
            renting,
            plan,
            resources=resources,
            artifact=provisioner.artifact_for(plan.env),
            watch=watch,
            floor=self.floor,
        ).land(shipment, name=name, node=node)

    def dispatch(self, asked: Request) -> Run:
        """Make the dispatch `asked` describes, whichever host it names.

        The one way a held request is asked for again, so a retry made by the durable sweep is
        the same dispatch the batch made and not a second spelling of it. Every default is
        resolved here rather than remembered from the first attempt, which is what makes a
        request held overnight land under whatever the manifest says in the morning.

        asked: the dispatch as it was originally requested.
        """
        return self.on(asked.target).submit(
            asked.command,
            name=asked.name,
            queue=asked.queue,
            walltime=asked.walltime,
            mem_gb=asked.mem_gb,
            gpus=asked.gpus,
            gpu_name=asked.gpu_name,
            max_usd=asked.max_usd,
            nodes=asked.nodes,
            attempt=asked.attempt,
            fetch=asked.fetch,
            node=asked.node,
            needs=asked.needs,
            env=asked.env,
            container=asked.container,
        )

    def provide(self, env: str = "", source: str = "", expect: str = "") -> Path:
        """Build the immutable environment a dispatched job activates, once, and name it.

        The verb a host runs for itself. A dispatch pins the digest of the compiled artifact it
        ships into the snapshot the job runs from, and this is what turns that digest into a
        built environment: one directory per lock, never written to again, so a wave queued
        against one lock keeps it however many times the workspace re-solves while it waits.

        Called again for an environment that is already built, it answers where it is and
        touches nothing, which is what lets every job of a wave call it and one of them build.

        env: the environment to build, the host profile's own when empty.
        source: the directory holding the compiled artifact to build from, workspace-relative
            or absolute; this workspace's own generated environment when empty.
        expect: the digest the dispatch pinned, refused when this machine reads the artifact as
            a different environment; unchecked when empty, which is what a build nobody
            dispatched still wants.
        """
        plan = self.plan(env=env, container="none")
        provisioner = Provisioner(self.root, self.manifest)
        where = self.dispatcher.local(source) if source else provisioner.environment_dir(plan.env)
        if expect:
            self.__pinned(where, expect, provisioner, modules=plan.profile.modules)
        prefixes = Prefixes(self.root, self.manifest, plan.env)
        built = prefixes.materialize(where, modules=plan.profile.modules)
        # Building is also the moment to let go of what nothing names any more, since this is
        # the machine that holds both the prefixes and the trees that point at them.
        dropped = prefixes.prune(live=prefixes.referenced(Path(Snapshots(str(self.root)).base)))
        if dropped:
            logger.info(
                "dropped %d unreferenced environment(s): %s", len(dropped), ", ".join(dropped)
            )
        return built

    def __pinned(
        self,
        where: Path,
        expect: str,
        provisioner: Provisioner,
        *,
        modules: Mapping[str, str],
    ) -> None:
        """Refuse to build when this machine reads the shipped artifact as another environment.

        Both sides must agree on the generated files, selected second-stage declarations, and
        ordered host modules. The refusal also names both Pixi versions, since lock rewrites
        were another source of identity drift. Building anyway would put an environment at a
        path no queued job will ever activate.

        where: the compiled artifact this build would read.
        expect: the digest the dispatch pinned.
        provisioner: this workspace's compile stack, which knows the pixi running here.
        """
        arrived = digest_of(where, modules=modules)
        if arrived == expect:
            return
        state = SyncState.load(where)
        solved = state.solved_by or "an unrecorded pixi"
        raise MissionError(
            f"{where} describes environment {arrived}, but the dispatch pinned {expect}. "
            "Check the selected second-stage runtime and ordered host modules as well as the "
            "generated files. That "
            f"artifact was compiled for {state.compiled_at or 'an unrecorded root'} from "
            f"manifest {state.compiled_from[:12] or 'nothing'}, so {self.__behind(state)}. It "
            f"was solved by pixi {solved} while this machine runs pixi "
            f"{provisioner.solver_version() or 'none'}: a pixi that is not the one the fleet is "
            f"pinned to ({PIXI_VERSION}) rewrites the lock while provisioning and moves the "
            f"address with it too. Run `{self.project.name} setup {self.host}` from the "
            "dispatching workspace, which ships this machine both the pinned pixi and the "
            "compile the dispatch addressed."
        )

    def __behind(self, state: SyncState) -> str:
        """Which side of a refused prime is holding the older compile, said in one clause.

        A dispatch ships the artifact it pinned, so the two can only disagree when something
        here wrote over it, and the root recorded beside it is what tells the two apart: the
        dispatching workspace's root means the shipped compile arrived and this machine then
        recompiled the mirror on top of it, and this machine's own root means the ship never
        landed and what stands here is a local compile of whatever the mirror last held.

        state: the blessing recorded beside the artifact that was read.
        """
        if state.compiled_at and Path(state.compiled_at) != self.root:
            return (
                "the dispatching workspace's own compile is what landed and something on this "
                "machine has recompiled over it since"
            )
        return (
            "this is a compile made on this machine rather than the one the dispatch shipped, "
            "which is a mirror left behind by a manifest edit"
        )

    def imports(self, plan: ExecutionPlan) -> tuple[str, ...]:
        """The workspace-relative directories a job imports this workspace's own packages from.

        A prefix is addressed by content, so one serves every tree whose manifest and lock agree,
        and its editable installs therefore point at the machine's own workspace root: the
        mirror. A sync landing between two waves then moves that source under jobs already
        queued or running, which is the one thing about a dispatched job a shared prefix cannot
        freeze. Anchoring the prefix at the snapshot instead only trades it for a worse fault,
        since snapshots are pruned a few deep while prefixes stand.

        So the job freezes it rather than the prefix: the pinned tree's own import roots go on
        `PYTHONPATH`, ahead of everything the environment adds, and what the prefix keeps of the
        editable install is the dependency metadata, which is all it was needed for here.

        An editable install puts one directory on `sys.path`: `src/` when the package keeps its
        code there and the package directory itself when it does not. That is read off this
        workspace, which is the tree every mirror and snapshot is a copy of.

        A path dependency that lives outside the root is compiled at `.mainboard/vendor/<dist>`
        (see `engines.compile.vendor`), which is inside it, so a vendored house package reaches
        this roster with the workspace's own packages and needs nothing said about it here.

        plan: the resolved execution context whose environment is being dispatched.
        """
        where = Provisioner(self.root, self.manifest).environment_dir(plan.env)
        try:
            compiled = (where / MANIFEST).read_text(encoding="utf-8")
        except OSError:
            return ()
        packages = self_installed(compiled, generated_dir=environment_shard(plan.env))
        return tuple(
            f"{package}/src".lstrip("/") if (self.root / package / "src").is_dir() else package
            for package in packages
        )

    def addressed(self, plan: ExecutionPlan, root: str) -> str:
        """Where on `root`'s host the immutable environment this dispatch pins is built.

        Content-addressed from this workspace's own compiled artifact, which is the same bytes
        the mirror ships and the snapshot hardlinks, so the digest the dispatch pins and the one
        the host arrives at when it builds are the same number reached independently.

        A workspace with nothing compiled has no environment to address and answers with
        nothing, which leaves the dispatch reaching the mirror's own the way it always did.

        plan: the resolved execution context whose environment is being addressed.
        root: the workspace root on the host.
        """
        if plan.containerized:
            return ""
        where = Provisioner(self.root, self.manifest).environment_dir(plan.env)
        try:
            digest = digest_of(where, modules=plan.profile.modules)
        except MissionError as unbuilt:
            logger.warning("dispatching without an addressed environment: %s", unbuilt)
            return ""
        return prefix_path(root, plan.env, digest)

    def resources(
        self,
        *,
        queue: str = "",
        walltime: str = "",
        mem_gb: int = 0,
        gpus: int = 0,
        gpu_name: str = "",
        max_usd: float = 0.0,
        nodes: int = 1,
        attempt: int = 1,
        plan: ExecutionPlan | None = None,
    ) -> Resources:
        """The resolved resource request for this host, profile defaults filling what is unset.

        The one resolution `submit` and `expectation` share, so what a submit is priced at is
        what it actually asks for. Expression-valued defaults are evaluated against `attempt`,
        so a retry escalates instead of dying to the same ceiling twice.

        plan: an already-resolved execution plan, this board's own when None.
        """
        resolved = plan or self.plan()
        defaults = resolved.profile.defaults
        memory = mem_gb or (evaluate(defaults.mem_gb, attempt=attempt) if defaults.mem_gb else 0)
        return Resources(
            queue=queue or defaults.queue,
            walltime=walltime or defaults.walltime,
            mem_gb=memory,
            gpus=gpus or defaults.gpus,
            gpu_name=gpu_name or defaults.gpu_name,
            max_usd=max_usd or defaults.max_usd,
            nodes=nodes,
            account=resolved.profile.account,
        )

    def run(self, command: Sequence[str], *, env: str = "", container: str = "") -> int:
        """Run `command` through the host's activated plan, returning its exit code.

        Locally the wrapped line executes in place; remotely it rides one ssh
        connection. Either way the same staging applies, cd, PATH, modules,
        then the environment or the container. A command naming a declared task
        is resolved by pixi inside the generated workspace instead of by the
        shell, which is what makes `run test` and `run -- pytest -q` the same verb.

        A job spelled `path/to/file.py::name` runs through the same runner a dispatched one
        does, with its closure's import roots and its provenance exported the same way, so the
        receipts it writes here are the receipts it would write on a node.

        command: exact command argv, or a declared task name and its arguments, or a job.
        env: an environment name overriding the profile's choice.
        container: a container override, `none` forcing bare.
        """
        plan = self.plan(env=env, container=container)
        target = Target.spelled(command, self.root)
        if target is not None:
            shipment = self.sealed(target, plan)
            listing = self.dispatcher.stage_listing(shipment)
            command = shipment.locally(self.root, closure=listing)
        if self.local and not plan.containerized:
            return Provisioner(self.root, self.manifest).run(command, plan.env)
        line = self.line(joined(command), env=env, container=container)
        if self.local:
            return foreground(localhost["bash"]["-lc", line])
        with connection(self.host) as remote:
            return foreground(remote["bash"]["-lc", line])

    def samples(
        self,
        stream: str,
        *,
        job: str,
        interval: float = 0.0,
        seconds: float = 0.0,
        parent: int = 0,
    ) -> Sampler:
        """This machine read into `stream`'s receipts for as long as the block runs.

        The in-process half of the live lane, for code that wants its own machine on the same
        run its receipts are on. A dispatched job gets the same thing without asking, since its
        script starts this through the CLI.

        stream: the receipts stream the samples belong to.
        job: the job inside that stream these readings describe.
        interval: seconds between readings, the manifest's own when 0.
        seconds: a hard stop, 0 to sample until the caller stops it.
        parent: a process to end with, 0 for none.
        """
        return Sampler(
            self.receipts(stream),
            stream=stream,
            job=job,
            interval=interval or self.manifest.tracking.interval,
            seconds=seconds,
            parent=parent,
        )

    def sampling(self, tracked: tuple[str, str], *, root: str, resources: Resources) -> str:
        """The line this job's script runs so it samples itself, empty when nothing samples it.

        Staging the credential is part of building the line rather than a step beside it,
        because the two are the same decision: a host is asked to ship its own series, so it is
        given the one variable that lets it, and a host that is asked for nothing is told
        nothing. A machine with no credential here still samples, into a queued offline run.

        tracked: the stream and job the samples belong to.
        root: the workspace root on the host.
        resources: the resolved request, whose walltime bounds the sampler the way it bounds
            the job.
        """
        declared = self.manifest.tracking
        if not declared.on or declared.interval <= 0:
            return ""
        stream, job = tracked
        return sampling_line(
            root=root,
            stream=stream,
            job=job,
            interval=declared.interval,
            seconds=walltime_seconds(resources.walltime) if resources.walltime else 0.0,
        )

    def scaffold(self) -> Scaffold:
        """The project generator, rendering the workspace's own templates through copier."""
        return Scaffold(self)

    def serve(self, name: str) -> int:
        """Run a declared engine's command through its container, returning its exit code.

        The same staging `run` gives any command, over one this workspace already named:
        `[engines.<name>]`'s command, inside the container it declares. No image is built here,
        only the launcher `run` already knows how to build for any container, so the container's
        own image must already exist.

        name: the `[engines.<name>]` table to serve.
        """
        try:
            engine = self.manifest.engines[name]
        except KeyError:
            raise MissionError(
                f"no engine {name!r}; declared engines are {sorted(self.manifest.engines)}"
            ) from None
        return self.run(engine.command, env=engine.env, container=engine.container)

    def shell(
        self,
        env: str = "",
        *,
        replace: Callable[[str, list[str], Mapping[str, str]], NoReturn] = os.execve,
    ) -> NoReturn:
        """Hand this terminal to an interactive shell inside the workspace environment.

        pixi already owns interactive activation, so the shell is `pixi shell` pointed at the
        generated workspace rather than a second activation written here. This process is
        replaced instead of wrapped, so the shell owns the terminal and every signal reaching
        it, and leaving the shell lands back where the user started rather than in a parent
        this tool left waiting. An environment nothing provisioned is refused the way a wrapped
        command is, naming the one command that fixes it, since a shell on whatever interpreter
        the machine happens to ship is exactly what the staging exists to prevent.

        The shell enters frozen, so opening one reads the lock and never rewrites it. Left to
        itself pixi treats entering an environment as a reason to bring the lock up to date
        with the manifest, which turns the everyday way into a workspace into an implicit solve
        nobody asked for, and this tool has one deliberate door for that, `install --resolve`.

        Replacing a process drops the environment a spawned child would have inherited, so the
        workspace's declared floors are handed over explicitly. Without that, a host that cannot
        present the virtual package fails on `shell` alone while every other verb works.

        env: the environment name, the host profile's own when empty.
        replace: the process-replacing exec, injectable so a test can read what it was handed.
        """
        if not self.local:
            raise MissionError(
                f"a shell runs on this machine only. Run "
                f"`{self.project.name} interact --on {self.host}` for a session there."
            )
        plan = self.plan(env=env, container="none")
        pixi = Provisioner(self.root, self.manifest).pixi_for(plan.env)
        if not pixi.ready(plan.env):
            raise MissionError(missing(plan, plan.prefix(str(self.root))))
        binary = str(pixi.executable)
        argv = [binary, "shell", *pixi.scope(), "--frozen", "-e", plan.env]
        replace(binary, argv, os.environ | pixi.overrides)

    def stage(self, root: str) -> None:
        """Put the one credential this host's jobs need where the job script will read it.

        A dispatched job ships its own live series, which needs the key on the machine running
        the job rather than on the one that dispatched it. Exactly one variable is written, into
        its own file with no group or world permission, so the host gets what the lane needs and
        nothing else this workspace holds. It is passed over stdin rather than as an argument,
        so it never appears in a process listing, and it is never logged. A machine holding no
        credential stages nothing, and its jobs queue offline for a later `wandb sync`.

        Staged once per dispatch by `submit` rather than by each line that needs it, since
        both the sampler and the attestation read the same file and two ssh round trips writing
        the same bytes buy nothing.

        root: the workspace root on the host.
        """
        variable = credential(self.manifest.tracking)
        Credentials().load()
        secret = os.environ.get(variable, "") if variable else ""
        if not secret or self.local:
            return
        path = host_env(root)
        written = f"umask 077; mkdir -p {shlex.quote(str(PurePosixPath(path).parent))}; "
        written += f"cat > {shlex.quote(path)}"
        with connection(self.host) as remote:
            (remote["bash"]["-c", written] << f"{variable}={shlex.quote(secret)}\n")()

    def submit(
        self,
        command: str,
        *,
        name: str = "",
        queue: str = "",
        walltime: str = "",
        mem_gb: int = 0,
        gpus: int = 0,
        gpu_name: str = "",
        max_usd: float = 0.0,
        nodes: int = 1,
        attempt: int = 1,
        fetch: str | None = None,
        node: str = "",
        needs: Sequence[str] = (),
        env: str = "",
        container: str = "",
        watch: Watcher | None = None,
    ) -> Run:
        """Dispatch `command` as a job on this host and return it as a run.

        A command line ships the mirror; a job spelled `path/to/file.py::name` ships its closure
        and runs through the runner. Both are one `Shipment`, read once, so everything below
        agrees about what runs and what tree it is.

        Unset resources fall back to the host profile's declared defaults,
        with expression-valued defaults evaluated against `attempt` so a
        retry escalates instead of dying to the same ceiling twice.

        A provider host is dispatched through its backend rather than over ssh, and the run it
        hands back is recorded in the same dispatch cache a queued job lands in. That record is
        what lets the durable sweep settle the run and end the rental, so a provider job nobody
        stays to watch stops costing money when its command does.

        Every dispatch is tracked, which is why a run that named itself nothing is named here.
        A stream needs one key that outlives this process, since the sweep that settles the run
        may be a cron on another day, and the run registry already keeps exactly one such field.

        command: the command the generated job runs.
        gpu_name: the GPU type a provider backend rents, ignored by the ssh family.
        max_usd: the spend cap a provider backend refuses to submit without.
        attempt: the 1-based try number, feeding the default expressions.
        fetch: a results path recorded for `Job.pull`, the job file's own declaration when
            unset, then the node's own evidence directory when the run serves one.
        node: the ledger slug this run serves, carried into its record and receipts.
        needs: data paths a job reads on the host, joining the ones its file declares.
        watch: announces the stages that happen on the far side and take long enough to be
            worth saying: every step of a rental's landing, and the priming of a queued host's
            environment.
        """
        # Before the plan, before the resources, and before any transport: a command a shell
        # cannot run costs a scheduler round trip on owned hardware and a whole rental on a
        # metered one, since a provider bills from boot and never learns the command never ran.
        command = vetted(command)
        plan = self.plan(env=env, container=container)
        shipment = self.shipment(command, plan, needs=needs)
        fetch = self.results(fetch or shipment.fetch, node=node) or None
        shipment = shipment.model_copy(update={"fetch": fetch or ""})
        resources = self.resources(
            queue=queue,
            walltime=walltime,
            mem_gb=mem_gb,
            gpus=gpus,
            gpu_name=gpu_name,
            max_usd=max_usd,
            nodes=nodes,
            attempt=attempt,
            plan=plan,
        )
        # A run that arrived without a name is minted one, content-addressed over the target,
        # the command and this instant, so its receipts stream has a durable key and
        # `mainboard jobs` reads better for it too.
        fingerprint = run_id({"host": plan.host, "command": shipment.spelling, "at": time.time()})
        label = name or f"{plan.host}-{fingerprint[:8]}"
        tracked = streamed(label, handle="")
        destination = route(plan.profile.kind)
        if destination != "ssh-family":
            backend = destination()
            run: Run = ProviderJob(
                self.on(plan.host),
                backend,
                self.rented(
                    backend,
                    plan,
                    shipment=shipment,
                    resources=resources,
                    name=label,
                    node=node,
                    watch=watch,
                ),
            )
        else:
            root = self.remote_root()
            self.stage(root)
            provisioner = Provisioner(self.root, self.manifest)
            # Before the address is taken and before the mirror leaves, so the pin, the shipped
            # artifact and the manifest this command was invoked under are one thing.
            provisioner.recompiled(plan.env)
            run = Job(
                self,
                self.dispatcher.run(
                    plan,
                    shipment,
                    root=root,
                    resources=resources,
                    name=label,
                    node=node,
                    fetch=fetch,
                    containerize=self.containerizer(plan, root),
                    sampler=self.sampling(tracked, root=root, resources=resources),
                    attestation=self.attesting(tracked, root=root),
                    watch=watch,
                    prefix=self.addressed(plan, root),
                    artifact=provisioner.artifact_for(plan.env),
                ),
            )
        self.announce(label, run, command=shipment.spelling, host=plan.host, node=node)
        return run

    def shipment(
        self, command: str, plan: ExecutionPlan, *, needs: Sequence[str] = ()
    ) -> Shipment:
        """What `command` runs and ships: a job's closure when it spells one, else the mirror.

        command: the vetted command line, a job spelled `path/to/file.py::name [args]` or not.
        plan: the resolved execution context whose environment names the import roots.
        needs: data paths declared at dispatch time, which only a job can take.
        """
        target = Target.spelled(shlex.split(command), self.root)
        if target is not None:
            return self.sealed(target, plan, needs=needs)
        if needs:
            raise MissionError(
                f"--needs belongs to a job spelled by file; {command!r} is a command and "
                "reaches the mirror as it is"
            )
        return Shipment.of_command(
            command, source=self.dispatcher.source(command), imports=self.imports(plan)
        )

    def sealed(
        self, target: Target, plan: ExecutionPlan, *, needs: Sequence[str] = ()
    ) -> Shipment:
        """The shipment of one job: its closure over this workspace's import roots, sealed.

        target: the job as spelled.
        plan: the resolved execution context whose environment names the import roots, and
            whose compiled prefix is where a distribution's installed shape is read from: this
            workspace's own copy of the environment, the lock the host installs frozen from.
        needs: data paths declared at dispatch time, joining the ones the job file declares.
        """
        closure = Closure.of(
            target,
            root=self.root,
            distributions=self.imports(plan),
            environment=Path(plan.prefix(str(self.root))),
            needs=needs,
        )
        return Shipment.of_closure(closure, root=self.root)

    def results(self, fetch: str | None, *, node: str = "", command: str = "") -> str:
        """What this dispatch pulls back: `fetch` when it names one, else `node`'s own evidence.

        A run that named the ledger node it serves has already said where its receipts go, since
        a node is a directory and its evidence is the directory inside it. Asking the caller to
        repeat that as a `--fetch` is how a whole GH200 wave's receipts stayed on the cluster
        (2026-09-05), so the node answers for itself and an explicit path still wins.

        fetch: the results path the caller declared, None or empty for none.
        node: the ledger slug this run serves, empty when it serves none.
        """
        if not fetch and command:
            target = Target.spelled(shlex.split(command), self.root)
            if target is not None:
                fetch = target.declaration(self.root).fetch
        return fetch or evidence_of(self.root, node)

    def verdicts(self) -> Verdicts:
        """The receipts-derived outcomes of this workspace's runs, the anti-fabrication read.

        Host-independent like `monitor`, since receipts belong to the workspace rather than to
        whichever host a board happens to be bound to.
        """
        return Verdicts(self)

    def watch(self, batch_id: str) -> Watch:
        """The live view over an already-dispatched batch, found by id alone.

        No spec, since everything a live view needs is durable: the batch's receipts name its
        handles and the dispatch cache says what became of them. A process that dispatched
        nothing can therefore take over watching a batch another one started.

        batch_id: the batch to watch.
        """
        return Watch(self, batch_id, bus=self.receipts(batch_id))
