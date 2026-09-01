import os
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
from .batch.spec import BatchJob
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
from .dispatch.backends.base import Credentials, Delivery, LogSource, ProviderBackend, route
from .dispatch.commandline import joined, vetted
from .dispatch.dispatcher import Dispatcher, Handle, Verdict
from .dispatch.jobs.spec import walltime_seconds
from .dispatch.onboard import HostSetup, Onboarding, facts_command, read_facts
from .dispatch.schedulers import HostUnreachable, pick, registry
from .dispatch.shared import logger
from .dispatch.vocabulary import Resources
from .dispatch.wrapping import connection, missing, wrap
from .doctor import Doctor
from .engines.compile.provisioner import Provisioner, task_line
from .engines.runtimes import resolve
from .experiments.fleet import Fleet
from .experiments.identity import run_id
from .manifest.loading import load
from .monitor import Monitor
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
    from .dispatch.onboard import Watcher
    from .dispatch.schedulers import Scheduler
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

    def __init__(self, backend: ProviderBackend, handle: Handle) -> None:
        """backend: the provider backend instance that submitted this run.

        handle: the dispatch handle carrying the provider's opaque run id.
        """
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

        A rented machine's disk dies with the rental, so this is the only channel a provider run
        has for anything it produced, which is why it must be read before the release that
        destroys the instance and why a provider without one is an empty string rather than a
        refusal.
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
        if not isinstance(self.backend, Delivery):
            raise MissionError(self.backend.refusal(Delivery, handle=self.handle.id, path=path))
        self.backend.deliver(self.handle.id, path=path)

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
        """Poll the provider until the run is terminal, end the rental, and return its verdict.

        The rental is released before the verdict is handed back, since a caller that blocked on
        this run is the last thing standing between a finished command and an instance that
        bills until someone notices.

        interval: seconds between provider state polls.
        poll: the sleeper between polls, injectable for tests.
        """
        while True:
            state = self.backend.state(self.handle.id)
            if state.verdict in vocabulary.TERMINAL:
                self.release()
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

    def batch(self, spec: BatchSpec) -> Batch:
        """The declared batch over this workspace, ready to prepare, price and dispatch.

        Host-independent like `monitor`, since a batch names a target per job and fans across
        the fleet rather than running on whichever host a board happens to be bound to.

        spec: the declared batch.
        """
        return Batch(self, spec, bus=self.receipts(spec.batch_id))

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

    def doctor(self) -> Doctor:
        """One verdict over this workspace, composed from the probes each subsystem owns."""
        return Doctor(self)

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
            return Onboarding(
                self.dispatcher,
                plan,
                root=plan.profile.root,
                artifact=provisioner.artifact,
                resolve=resolve,
                watch=watch,
                digest=provisioner.compiler.digest(),
            ).run(sync_only=sync_only)
        provisioner.provision(plan.env, resolve=resolve)
        activate = provisioner.activate(plan.env, modules=plan.profile.modules)
        return HostSetup(
            host=self.host,
            root=str(self.root),
            env=plan.env,
            activate=str(activate),
            installer="in-place",
            tool=version(self.project.name),
        )

    def interact(
        self,
        *command: str,
        env: str = "",
        queue: str = "",
        walltime: str = "",
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

        command: a command to run instead of handing over the terminal, its own flags included.
        env: an environment name overriding the profile's choice.
        queue: the queue the allocation targets, the profile's own when empty.
        walltime: the session's wall-clock limit, the profile's own when empty.
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
        destination = route(record.kind)
        if destination != "ssh-family":
            return ProviderJob(
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

        command: exact command argv, or a declared task name and its arguments.
        env: an environment name overriding the profile's choice.
        container: a container override, `none` forcing bare.
        """
        plan = self.plan(env=env, container=container)
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
        pixi = Provisioner(self.root, self.manifest).pixi
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
        env: str = "",
        container: str = "",
    ) -> Run:
        """Dispatch `command` as a job on this host and return it as a run.

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
        fetch: a results path recorded for `Job.pull`.
        node: the ledger slug this run serves, carried into its record and receipts.
        """
        # Before the plan, before the resources, and before any transport: a command a shell
        # cannot run costs a scheduler round trip on owned hardware and a whole rental on a
        # metered one, since a provider bills from boot and never learns the command never ran.
        command = vetted(command)
        plan = self.plan(env=env, container=container)
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
        fingerprint = run_id({"host": plan.host, "command": command, "at": time.time()})
        label = name or f"{plan.host}-{fingerprint[:8]}"
        tracked = streamed(label, handle="")
        destination = route(plan.profile.kind)
        if destination != "ssh-family":
            backend = destination()
            run: Run = ProviderJob(
                backend,
                self.dispatcher.track(
                    backend.submit(plan, command, resources),
                    host=plan.host,
                    kind=plan.profile.kind,
                    command=command,
                    name=label,
                    node=node,
                    fetch=fetch,
                ),
            )
        else:
            root = self.remote_root()
            self.stage(root)
            run = Job(
                self,
                self.dispatcher.run(
                    plan,
                    command,
                    root=root,
                    resources=resources,
                    name=label,
                    node=node,
                    fetch=fetch,
                    containerize=self.containerizer(plan, root),
                    sampler=self.sampling(tracked, root=root, resources=resources),
                    attestation=self.attesting(tracked, root=root),
                ),
            )
        self.announce(label, run, command=command, host=plan.host, node=node)
        return run

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
