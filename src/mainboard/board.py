import json
import os
import platform
import shlex
import time
from copy import copy
from functools import partial
from importlib.metadata import version
from pathlib import Path
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
from .dispatch.landing import Landing, renter
from .dispatch.onboard import (
    HostSetup,
    Onboarding,
    facts_command,
    gpus_command,
    read_facts,
)
from .dispatch.rentals import identity
from .dispatch.schedulers import HostUnreachable, pick, registry
from .dispatch.shared import logger
from .dispatch.shells import dialect_for, is_windows, open_shell
from .dispatch.shipment import Shipment
from .dispatch.snapshots import Snapshots
from .dispatch.targets import home_of, placed, rooted
from .dispatch.targets import resolve as resolved_profile
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
from .fitness import Fitness
from .git import Tree
from .jobs.call import Fresh
from .jobs.closure import Closure
from .jobs.target import Target
from .manifest.loading import load
from .manuscript import Manuscript
from .monitor import Monitor
from .nodes import evidence_of
from .probe.occupancy import Occupancy
from .probe.snapshot import HostFacts
from .runtime.activation import Runtime
from .runtime.job import walltime_seconds
from .scaffold import Scaffold
from .tracking import (
    Sampler,
    attesting,
    credential,
    host_env,
    is_batched,
    mirrored,
    sampling,
    streamed,
)
from .verdicts import Verdicts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .batch.receipts import Bus
    from .batch.spec import BatchSpec
    from .context.plan import ExecutionPlan
    from .core.section import Section
    from .dispatch.schedulers import Scheduler
    from .dispatch.shared import Watcher
    from .dispatch.vocabulary import JobState
    from .manifest.schema.root import Manifest
    from .probe.system import System
    from .runtime.job import ToolCall

# How long one manuscript build may take. A first build downloads the engine's bundle, so the
# ceiling is minutes, and a TeX run looping on a bad macro still ends inside it.
_PAPER_SECONDS = 900.0


class Job:
    """One dispatched run, addressed as an object instead of handle flags."""

    def __init__(self, board: Board, handle: Handle) -> None:
        """board: the host-bound board that submitted this job."""
        self.board = board
        self.handle = handle

    @property
    def scheduler(self) -> Scheduler:
        """The backend selected on the kind this run was dispatched under, not today's profile.

        Every other probe follows the recorded kind too (`Dispatcher.state`, `Dispatcher.states`,
        the sweep's grouping); a host whose declared kind changed under a live job would
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
        """The tolerant twin of `logs`: empty when this backend keeps none or will not answer.

        A settle wants the output if it can have it and must never fail the sweep over a host
        that went quiet between the probe and the read.
        """
        try:
            return self.logs()
        except (HostUnreachable, MissionError, OSError, ProcessExecutionError) as quiet:
            logger.warning("no transcript for %s: %s", self.handle.id, quiet)
            return ""

    def poll(self) -> JobState:
        """The job's state now, raising `HostUnreachable` when its host will not answer.

        A durable sweep has to say which host went quiet rather than quietly try again; `state`
        is the same probe with the blip absorbed.
        """
        return self.board.dispatcher.state(self.handle)

    def pull(self) -> None:
        """Bring the job's recorded results path back to this machine."""
        self.board.dispatcher.fetch(self.handle)

    def release(self) -> None:
        """Nothing to let go of: a queue stops charging when the job ends, so no kill is sent.

        The verb exists because a provider run bills until cancelled, and a sweep settling
        either kind says the same thing to both.
        """

    def state(self) -> JobState | None:
        """One non-blocking probe, None when the host could not be reached on this tick.

        None is a reason to look again rather than a verdict; `wait` is the blocking loop.
        """
        return self.board.dispatcher.probe(self.handle)

    def wait(self, *, interval: float | None = None) -> Verdict:
        """Block until the job is terminal and return its verdict."""
        extra = {"interval": interval} if interval is not None else {}
        return self.board.dispatcher.await_many([self.handle], **extra)[self.handle]


class ProviderJob:
    """One provider-dispatched run, the transport-free twin of `Job`.

    Only the lifecycle is on every backend. Logs and artifact delivery are capabilities, asked
    for by contract first and refused with the backend's own advice when it has none.
    """

    def __init__(self, board: Board, backend: ProviderBackend, handle: Handle) -> None:
        """board: the workspace bound to the provider host.

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
        before release.
        """
        if not isinstance(self.backend, LogSource):
            return ""
        try:
            return self.backend.logs(self.handle.id)
        except (MissionError, OSError) as quiet:
            logger.warning("no transcript for %s: %s", self.handle.id, quiet)
            return ""

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
            profile = self.board.plan().profile
            key = identity(profile.vars.get("ssh-key", "")).private
            endpoint = self.backend.endpoint(self.handle.id, key=key)
            policy = SshTransport(endpoint=endpoint)
            root = self.handle.root or profile.root
            if root.startswith("~"):
                with connection(endpoint.destination, policy) as remote:
                    root = placed(root, home=home_of(remote))
            self.board.dispatcher.fetch_path(
                endpoint.destination, root=root, path=path, ssh=policy
            )
        else:
            raise MissionError(self.backend.refusal(Delivery, handle=self.handle.id, path=path))

    def release(self) -> None:
        """End the rental, the only thing that stops a provider charging for it.

        A finished command does not end a provider run. Vast holds the instance and restarts the
        exited container until cancelled (thirteen re-runs in five minutes, verified live
        2026-08-19), and an HPC-AI instance runs until terminated.
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


# A dispatched run from either world. Both answer `poll`, `pull` and `release` alike, so one
# durable sweep settles a queued job and a rented instance without asking which it holds.
type Run = Job | ProviderJob


class Board:
    """The one addressable interface: a workspace, pivoted onto a host by `on`.

    `Board()` finds the manifest like git finds a repository. The unbound board is this machine;
    `board.on("gold")` is the same board bound to a declared host, where `run`, `submit` and
    `facts` keep their shapes while the profile decides scheduler, environment, container and
    queue policy. The composed subsystems stay public for anything the facade does not carry.
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

        Rooted rather than left to the cwd, so a command typed in a subdirectory reads the same
        run registry, stages into the same jobs directory and mirrors the same tree.
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

        A batch publishes its own submissions and is skipped, so no line is written twice.

        label: the run's dispatch label, which says both where it belongs and who publishes it.
        command: what the job runs, recorded as this run's config.
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
        """Publish one attestation of this machine into `stream`'s receipts.

        The synchronous, once-only twin of `samples`. It reads the machine it runs on, so a
        dispatched job attests the node doing the work, the only reading that describes the
        measurement's conditions.
        """
        Sampler(self.receipts(stream), stream=stream, job=job, interval=0.0).attest()

    def attesting(self, tracked: tuple[str, str], *, root: str) -> ToolCall | None:
        """The call this job makes to attest its own machine, None when tracking is off.

        Gated like `sampling`, but with no interval: an attestation happens exactly once and its
        whole value is that it happens before the work.

        tracked: the stream and job the attestation belongs to.
        root: the workspace root on the host.
        """
        if not self.manifest.tracking.on:
            return None
        stream, job = tracked
        return attesting(root=root, stream=stream, job=job)

    def batch(self, spec: BatchSpec, *, selection: Selection | None = None) -> Batch:
        """The declared batch over this workspace, ready to prepare, price and dispatch.

        Host-independent, since a batch names a target per job and fans across the fleet.

        selection: which of the plan's jobs to act on, all when None. The batch keeps its
            identity and receipts stream either way, so a plan sent out in waves is one batch.
        """
        return Batch(self, spec, bus=self.receipts(spec.batch_id), selection=selection)

    def compute(self) -> Survey:
        """The host-independent survey of every compute path this workspace reaches, here too."""
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
        """The manifest's declared requirements, editable and re-solvable; host-independent."""
        return Dependencies(self)

    def doctor(self, env: str = "") -> Doctor:
        """One verdict over this workspace and one resolved environment.

        env: the environment name, the bound host profile's own when empty.
        """
        return Doctor(self, env=env)

    def git(self) -> Tree:
        """This workspace's checkout: the root repository and every submodule, as one tree."""
        return Tree(self.root, self.manifest.git)

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

        The resource resolution `submit` runs, then the queue policy check, so a refused request
        dies here in one sentence rather than after an ssh round trip. The price is the
        estimator's (a provider's metered rate, zero on owned hardware), with the declared
        walltime standing in for the runtime as a batch spec's `runtime_s` does. Nothing
        connects, rents or dispatches.
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

        A remote host answers with the tool `install` put there, so the probe never depends on
        this workspace's mainboard importing under whatever interpreter the host ships.
        """
        if self.local:
            return HostFacts.collected(self.root)
        with open_shell(self.plan(container="none"), self.remote_root()) as shell:
            return read_facts(shell.run(facts_command(), activate=True))

    def findings(self, system: System) -> list[Section]:
        """What this host's software census means for this workspace, one judged row each.

        The judge `compute`, `setup` and `center verify` share, so a driver below the CUDA floor
        is the same row whichever verb found it.
        """
        return Fitness(self.root, self.manifest).judge(system, host=self.host)

    def occupancy(self) -> Occupancy:
        """Who holds each card of this host right now, local or through the host's own tool.

        A scheduler host answers for its card-less login node, not its allocations; ask `jobs`
        for what runs there.
        """
        if self.local:
            return Occupancy.collected()
        with open_shell(self.plan(container="none"), self.remote_root()) as shell:
            text = shell.run(gpus_command(), activate=True)
        line = next((line for line in reversed(text.splitlines()) if line.startswith("{")), "")
        if not line:
            raise MissionError(f"no occupancy in the probe output: {text.strip()[-240:]}")
        return Occupancy.model_validate_json(line)

    @property
    def floor(self) -> str:
        """The tool version this workspace declares, empty when it declares none.

        A workspace vendoring the tool's source needs no version; one consuming it from an index
        names it like any other dependency, and a host with no vendored source installs that.
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

        A board bound to a host runs the whole onboarding there: mirror the workspace, install
        the tool from the mirror, provision with the host's own tool, probe what it became. An
        empty `env` means the profile's declared choice rather than `default`, so setting a host
        up installs what the manifest says it runs. A host installs from the artifact this
        workspace already solved, so its own compiler never enters the lock's dependency path.

        env: the environment name, the host profile's own when empty.
        resolve: allow a fresh dependency solve, refused otherwise when the lock cannot vouch
            for what is on disk. For a host it means solving there instead of installing the
            shipped artifact.
        profile: the declared host profile describing this machine, so the generated activation
            carries that host's module stack; this board's own host when empty.
        watch: announces each onboarding stage as it begins.
        sync_only: re-mirror and re-provision an onboarded host without reinstalling the tool or
            re-probing its hardware, neither of which changed when only the manifest moved;
            refused on this machine, which has no onboarding to skip parts of.
        """
        if sync_only and self.local:
            raise MissionError(
                "--sync-only onboards a remote host faster; this machine has no onboarding to "
                "shortcut, run `install` instead"
            )
        plan = self.resolver.plan(profile or self.host, env=env, container="none")
        provisioner = Provisioner(self.root, self.manifest)
        if not self.local:
            compiler = provisioner.compiler_for(plan.env)
            if not resolve:
                # The host will refuse a lock this manifest did not solve; ask here first,
                # before the mirror and the remote install spend minutes reaching that answer.
                compiler.vouch()
            return Onboarding(
                self.dispatcher,
                plan,
                artifact=provisioner.artifact_for(plan.env),
                resolve=resolve,
                watch=watch,
                digest=compiler.digest(),
                floor=self.floor,
            ).run(sync_only=sync_only)
        provisioner.provision(plan.env, resolve=resolve)
        # A platform this machine cannot run has no prefix to activate here; its lock ships with
        # `setup`. `activate.sh` is bash, which nothing on Windows sources: a Windows workspace
        # activates through the activation pixi cached at provisioning, so none is named.
        activate = (
            str(provisioner.activate(plan.env, modules=plan.profile.modules))
            if provisioner.runs_here(plan.env) and platform.system() != "Windows"
            else ""
        )
        return HostSetup(
            host=self.host,
            root=str(self.root),
            env=plan.env,
            activate=activate,
            installer="in-place",
            tool=version(self.project.name),
        )

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

        The counterpart of `shell` for another machine. This process is replaced by the ssh, so
        the session owns the terminal and its signals and leaving it lands where the user began.
        Each scheduler decides what a session is: an ssh box hands the terminal to its own tool,
        a queued cluster first allocates a compute node. The staging is only the `cd`, `PATH`
        and modules every remote command gets; the far side owns activation. A kept session
        runs in a tmux session named for this workspace and host, so the terminal can drop while
        the allocation stays up, and `keep` again reattaches instead of allocating another node.

        command: a command to run instead of handing over the terminal, its own flags included.
        keep: hold the session in tmux on the far side and reattach to one already held.
        replace: the process-replacing exec, injectable so a test can read the argv it built.
        """
        if self.local:
            raise MissionError(
                f"an interactive session needs a host. Run `{self.project.name} shell` for "
                "this machine."
            )
        plan = self.plan(env=env, container="none")
        if route(plan.profile.kind) != "ssh-family":
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
        dialect = dialect_for(plan.profile)
        staged = dialect.stage(plan, self.remote_root(), command=session, activate=False)
        if keep:
            # `new-session -A` attaches to the named session when it exists and only otherwise
            # starts one, so the same verb both opens and returns to a held allocation.
            held = f"{self.project.name}-{self.host}"
            staged = f"tmux new-session -A -s {shlex.quote(held)} {shlex.quote(staged)}"
        # A bounded transport suits a poll, not a session, so the user's ssh config owns this one.
        replace("ssh", dialect.session(self.host, staged))

    def job(self, handle: str | int, *, host: str = "") -> Run:
        """The dispatched run `handle`, rebuilt from the dispatch cache as whichever kind it is.

        A fresh process addresses a running job as the submitting one did, with no `Handle`
        reassembled by hand. The recorded kind decides the world it returns from, so a rental
        outlives the process that started it exactly as a queued job does.

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
        bound = self.on(record.target)
        rebuilt = partial(
            Handle,
            id=record.handle,
            host=record.target,
            kind=record.kind,
            fetch_path=record.fetch_path,
        )
        destination = route(record.kind)
        if destination == "ssh-family":
            return Job(bound, rebuilt(root=bound.remote_root()))
        return ProviderJob(bound, destination(), rebuilt(root=""))

    def line(self, command: str, *, env: str = "", container: str = "") -> str:
        """The staged shell line this board's host would run `command` through.

        The one place the staging (cd, PATH, modules, then environment or container) is
        assembled, so a caller wanting the output rather than the exit code runs the very line
        `run` runs.

        command: the shell command, or a declared task name and its arguments.
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
        """The host-independent durable sweep over every job the dispatch cache owes an outcome."""
        return Monitor(self)

    def on(self, host: str) -> Board:
        """This workspace bound to `host`, sharing the loaded manifest and caches.

        host: a declared host alias, or any ssh-config alias for defaults.
        """
        bound = copy(self)
        bound.host = host
        return bound

    def once[Built](self, key: str, build: Callable[[], Built]) -> Built:
        """The one `key` this workspace shares, built on first ask and never a second time.

        Locked because the first ask routinely comes from a worker thread (a doctor asks four
        questions at once, a survey probes a fleet in a pool), and two threads building a
        dispatcher would open a second SQLite connection owned by whichever thread won. The lock
        is reentrant since one build reads another. A build that raises is not remembered, so an
        unparsable manifest is re-read and re-refused. Emptying a slot makes the values derived
        from it be rebuilt (a test rewriting the manifest).
        """
        with self.guard:
            built = self.shared.get(key) or build()
            self.shared[key] = built
            return cast("Built", built)

    def paper(self, name: str) -> Manuscript:
        """The declared manuscript `name` (`[papers.<name>]`), built through this environment."""
        try:
            declared = self.manifest.papers[name]
        except KeyError:
            raise MissionError(
                f"no paper {name!r}; declared papers are {sorted(self.manifest.papers)}"
            ) from None
        provisioner = Provisioner(self.root, self.manifest)
        return Manuscript(
            name,
            declared,
            root=self.root,
            run=partial(provisioner.capture, env=self.plan().env, timeout=_PAPER_SECONDS),
        )

    def plan(self, *, env: str = "", container: str = "") -> ExecutionPlan:
        """The resolved execution plan for this board's host.

        A bound host's profile is completed from what its setup probed, so a platform the
        manifest never spelled out is still the one every later command stages for, and a `~`
        root is the one path under that host's home.
        """
        plan = self.resolver.plan(self.host, env=env, container=container)
        if self.local:
            return plan
        try:
            recorded = self.dispatcher.cache.host(self.host)
        except LookupError:
            return plan
        if recorded.capabilities is None:
            return plan
        return plan.model_copy(
            update={"profile": resolved_profile(plan.profile, recorded.capabilities)}
        )

    def receipts(self, stream: str) -> Bus:
        """Where one stream's events go: this workspace's own file, plus whatever it declared.

        The composition root for tracking, so a batch, a plain submit and a study all mirror the
        same way and none knows a reporting service exists. `[tracking]` set `off` gets the file
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
        """The workspace root on the bound host, refusing one its setup never placed."""
        return rooted(self.plan().profile, host=self.host)

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

        A rental is shipped the artifact `install` ships gold: this workstation solved the lock
        and the machine installs frozen against it. The lock vouches here, before the rental
        opens, since a refusal a minute later has already cost money. A plan bringing its own
        container skips all of it, since a prebuilt image already holds everything.

        shipment: what the job runs and ships. A job spelled by file needs a workspace to ship
            its closure into, so a plan whose image is the whole environment refuses it.
        resources: the resolved request, whose spend cap and walltime bound the rental.
        name: the label retained with the allocated handle.
        node: the research node served by the dispatch.
        watch: announces each landing stage, since a landing is minutes of mirror, install and
            provisioning that would otherwise stand silent on a metered box.
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

        The one way a held request is asked for again, so the durable sweep's retry is the
        batch's dispatch and not a second spelling of it. Defaults resolve now rather than being
        remembered, so a request held overnight lands under the morning's manifest.
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

        The verb a host runs for itself. A dispatch pins the digest of the artifact it ships
        into the job's snapshot, and this turns that digest into one directory per lock, never
        written again, so a wave queued against one lock keeps it however often the workspace
        re-solves. Called for an environment already built it answers where it is and touches
        nothing, so every job of a wave can call it and one builds.

        env: the environment to build, the host profile's own when empty.
        source: the directory holding the compiled artifact to build from, workspace-relative
            or absolute; this workspace's own generated environment when empty.
        expect: the digest the dispatch pinned, refused when this machine reads the artifact as
            a different environment; unchecked when empty, as a build nobody dispatched wants.
        """
        plan = self.plan(env=env, container="none")
        provisioner = Provisioner(self.root, self.manifest)
        where = self.dispatcher.local(source) if source else provisioner.environment_dir(plan.env)
        if expect:
            self.__pinned(where, expect, provisioner, modules=plan.profile.modules)
        prefixes = Prefixes(self.root, self.manifest, plan.env)
        built = prefixes.materialize(where, modules=plan.profile.modules)
        # This machine holds both the prefixes and the trees that point at them, so building is
        # the moment to let go of what nothing names any more.
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

        Both sides must agree on the generated files, selected second-stage declarations and
        ordered host modules; building anyway would put an environment at a path no queued job
        will activate. The refusal names both pixis, since lock rewrites also drift identity.
        """
        arrived = digest_of(where, modules=modules)
        if arrived == expect:
            return
        state = SyncState.load(where)
        # A dispatch ships the artifact it pinned, so a mismatch means something here wrote over
        # it. The recorded root tells which: the dispatching workspace's means the ship landed
        # and was recompiled over, this machine's means it never landed and the mirror's stale
        # compile stands.
        behind = (
            "the dispatching workspace's own compile is what landed and something on this "
            "machine has recompiled over it since"
            if state.compiled_at and Path(state.compiled_at) != self.root
            else "this is a compile made on this machine rather than the one the dispatch "
            "shipped, which is a mirror left behind by a manifest edit"
        )
        solved = state.solved_by or "an unrecorded pixi"
        raise MissionError(
            f"{where} describes environment {arrived}, but the dispatch pinned {expect}. "
            "Check the selected second-stage runtime and ordered host modules as well as the "
            "generated files. That "
            f"artifact was compiled for {state.compiled_at or 'an unrecorded root'} from "
            f"manifest {state.compiled_from[:12] or 'nothing'}, so {behind}. It "
            f"was solved by pixi {solved} while this machine runs pixi "
            f"{provisioner.solver_version() or 'none'}: a pixi that is not the one the fleet is "
            f"pinned to ({PIXI_VERSION}) rewrites the lock while provisioning and moves the "
            f"address with it too. Run `{self.project.name} setup {self.host}` from the "
            "dispatching workspace, which ships this machine both the pinned pixi and the "
            "compile the dispatch addressed."
        )

    def imports(self, plan: ExecutionPlan) -> tuple[str, ...]:
        """The workspace-relative directories a job imports this workspace's own packages from.

        A content-addressed prefix serves every tree whose manifest and lock agree, so its
        editable installs point at the machine's workspace root, the mirror, and a sync between
        two waves moves source under queued jobs. Anchoring the prefix at the snapshot trades
        that for worse, since snapshots are pruned a few deep while prefixes stand. So the job
        puts the pinned tree's import roots on `PYTHONPATH` ahead of the environment, and the
        prefix keeps only the editable install's dependency metadata.

        An editable install puts `src/` on `sys.path` when the package keeps its code there and
        the package directory otherwise, read off this workspace, which every mirror and
        snapshot copies. An out-of-root path dependency is compiled at `.mainboard/vendor/<dist>`
        (see `engines.compile.vendor`), inside the root, so it arrives here with the rest.
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

        Content-addressed from this workspace's compiled artifact, the bytes the mirror ships
        and the snapshot hardlinks, so the pinned digest and the one the host reaches when it
        builds agree independently. With nothing compiled it answers empty and the dispatch
        reaches the mirror's own environment.

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

        Shared by `submit` and `expectation`, so a submit is priced at what it asks for.
        Expression-valued defaults are evaluated against `attempt`, so a retry escalates
        instead of dying to the same ceiling twice.

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

        Local commands execute in place; remote ones use one ssh connection, a batch cluster's
        login endpoint included. A declared task name is resolved by pixi inside the generated
        workspace rather than by the shell, which makes `run test` and `run pytest -q` one verb.
        A job file target runs here only, through the runner and closure format submitted jobs
        use; remotely it needs `submit` for allocation and source transfer, while collection and
        help stay local and allocate nothing.

        command: exact command argv, or a declared task name and its arguments, or a job.
        container: a container override, `none` forcing bare.
        """
        plan = self.plan(env=env, container=container)
        target = Target.spelled(command, self.root)
        exported: dict[str, str] | None = None
        if target is not None:
            if not self.local:
                raise MissionError(
                    "remote file targets require submission for source transfer and allocation; "
                    f"use mainboard submit --on {plan.host} -- {target.spelling}, then "
                    "mainboard wait or mainboard monitor. Run collection and help locally."
                )
            shipment = self.sealed(target, plan)
            listing = self.dispatcher.stage_listing(shipment)
            if platform.system() == "Windows" and not plan.containerized:
                exported = shipment.local_exports(self.root, closure=listing)
                command = shlex.split(shipment.command)
            else:
                command = shipment.locally(self.root, closure=listing)
        if self.local and not plan.containerized:
            return Provisioner(self.root, self.manifest).run(command, plan.env, exports=exported)
        if not self.local and is_windows(plan.profile):
            with open_shell(plan, self.remote_root()) as shell:
                return shell.foreground(task_line(self.manifest, joined(command), env=plan.env))
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

        The in-process half of the live lane; a dispatched job gets the same through the CLI.

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

    def sampling(
        self, tracked: tuple[str, str], *, root: str, resources: Resources
    ) -> ToolCall | None:
        """The call this job makes so it samples itself, None when nothing samples it.

        A host asked to ship its own series gets the credential staged by `stage`; one without a
        credential still samples, into a queued offline run.

        tracked: the stream and job the samples belong to.
        root: the workspace root on the host.
        resources: the resolved request, whose walltime bounds the sampler as it bounds the job.
        """
        declared = self.manifest.tracking
        if not declared.on:
            return None
        stream, job = tracked
        return sampling(
            root=root,
            stream=stream,
            job=job,
            interval=declared.interval,
            seconds=walltime_seconds(resources.walltime) if resources.walltime else 0.0,
        )

    def scaffold(self) -> Scaffold:
        """The project generator, rendering the workspace's own templates through copier."""
        return Scaffold(self)

    def shell(
        self,
        env: str = "",
        *,
        replace: Callable[[str, list[str], Mapping[str, str]], NoReturn] = os.execve,
    ) -> NoReturn:
        """Hand this terminal to an interactive `pixi shell` inside the workspace environment.

        pixi owns interactive activation, so there is no second activation here. This process
        is replaced rather than wrapped, so the shell owns the terminal and its signals and
        leaving it lands where the user began. An unprovisioned environment is refused naming
        the fix, since a shell on the machine's own interpreter is what staging prevents. The
        shell enters `--frozen`: pixi would otherwise treat entering as a reason to re-solve the
        lock, and `install --resolve` is the one deliberate door for that. Exec drops what a
        spawned child would inherit, so the declared floors and the runtime step's changes are
        passed explicitly; without the floors a host lacking the virtual package fails on
        `shell` alone.

        env: the environment name, the host profile's own when empty.
        replace: the process-replacing exec, injectable so a test can read what it was handed.
        """
        if not self.local:
            raise MissionError(
                f"a shell runs on this machine only. Run "
                f"`{self.project.name} shell --on {self.host}` for a session there."
            )
        plan = self.plan(env=env, container="none")
        pixi = Provisioner(self.root, self.manifest).pixi_for(plan.env)
        if not pixi.ready(plan.env):
            raise MissionError(missing(plan, plan.prefix(str(self.root))))
        binary = str(pixi.executable)
        argv = [binary, "shell", *pixi.scope(), "--frozen", "-e", plan.env]
        environ = os.environ | pixi.overrides
        replace(binary, argv, environ | Runtime(pixi.env_prefix(plan.env)).changes(environ))

    def stage(self, root: str) -> None:
        """Put the one credential this host's jobs need where the job's runner will read it.

        A dispatched job ships its own live series, so the key must be on the machine running
        it. Exactly one variable is written, to its own file with no group or world permission,
        passed over stdin so it never shows in a process listing, and never logged. A machine
        holding no credential stages nothing and its jobs queue offline for a later `wandb
        sync`. `submit` stages once per dispatch, since the sampler and the attestation read
        the same file.

        root: the workspace root on the host.
        """
        variable = credential(self.manifest.tracking)
        Credentials().load()
        secret = os.environ.get(variable, "") if variable else ""
        if not secret or self.local:
            return
        with open_shell(self.plan(container="none"), root) as shell:
            shell.write(host_env(root), json.dumps({variable: secret}))

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
        agrees on what runs from which tree. Unset resources come from `resources`.

        A provider host is dispatched through its backend and recorded in the same dispatch
        cache. Monitor must run to collect evidence and request release, so install its
        periodic pass for unattended jobs; provider outages can delay release, and a command
        exit or local timeout alone does not stop billing.

        Every dispatch is tracked, so an unnamed run is named here: its stream needs a key that
        outlives this process, since the settling sweep may be a cron on another day, and the
        run registry keeps exactly one such field.

        gpu_name: the GPU type a provider backend rents, ignored by the ssh family.
        max_usd: the spend cap a provider backend refuses to submit without.
        attempt: the 1-based try number, feeding the default expressions.
        fetch: a results path recorded for `Job.pull`, the job file's own declaration when
            unset, then the node's own evidence directory when the run serves one.
        node: the ledger slug this run serves, carried into its record and receipts.
        needs: data paths a job reads on the host, joining the ones its file declares.
        watch: announces the far-side stages long enough to be worth saying: every step of a
            rental's landing, and the priming of a queued host's environment.
        """
        # Before any plan or transport: a command a shell cannot run costs a scheduler round
        # trip on owned hardware and a whole rental on a metered one, billed from boot.
        command = vetted(command)
        plan = self.plan(env=env, container=container)
        shipment = self.shipment(command, plan, needs=needs)
        shipment.admit(self.root)
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
        # Content-addressed over the target, the command and this instant, so the receipts
        # stream has a durable key and `mainboard jobs` reads better for it too.
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
            # Before the address is taken and the mirror leaves, so the pin, the shipped artifact
            # and the manifest this command was invoked under are one thing.
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
            command,
            source=self.dispatcher.source(command, paths=plan.profile.sync.include),
            imports=self.imports(plan),
        )

    def sealed(
        self, target: Target, plan: ExecutionPlan, *, needs: Sequence[str] = ()
    ) -> Shipment:
        """The shipment of one job: its closure over this workspace's import roots, sealed.

        plan: the resolved execution context whose environment names the import roots, and
            whose compiled prefix (this workspace's copy of the lock the host installs frozen
            from) is where a distribution's installed shape is read.
        needs: data paths declared at dispatch time, joining the ones the job file declares.
        """
        # Reject malformed runner arguments before building or dispatching a snapshot.
        Fresh.parsed(target.args)
        closure = Closure.of(
            target,
            root=self.root,
            distributions=self.imports(plan),
            environment=Path(plan.prefix(str(self.root))),
            needs=needs,
        )
        return Shipment.of_closure(closure, root=self.root)

    def results(self, fetch: str | None, *, node: str = "", command: str = "") -> str:
        """What this dispatch pulls back: `fetch`, the job's declared path, or `node`'s evidence.

        A node is a directory and its evidence the directory inside it, so it answers for
        itself: making callers repeat it as `--fetch` is how a whole GH200 wave's receipts
        stayed on the cluster (2026-09-05). An explicit path still wins.
        """
        if not fetch and command:
            target = Target.spelled(shlex.split(command), self.root)
            if target is not None:
                fetch = target.declaration(self.root).fetch
        return fetch or evidence_of(self.root, node)

    def verdicts(self) -> Verdicts:
        """The receipts-derived outcomes of this workspace's runs, the anti-fabrication read."""
        return Verdicts(self)

    def watch(self, batch_id: str) -> Watch:
        """The live view over an already-dispatched batch, found by id alone.

        Everything it needs is durable (the receipts name the handles, the dispatch cache says
        what became of them), so a process that dispatched nothing can watch another's batch.
        """
        return Watch(self, batch_id, bus=self.receipts(batch_id))
