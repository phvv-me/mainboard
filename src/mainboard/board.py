import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

from plumbum import FG, ProcessExecutionError
from plumbum import local as localhost

from .context.expressions import evaluate
from .context.resolver import Resolver
from .core.errors import MissionError
from .core.project import Project
from .dispatch.backends.base import ProviderBackend, route
from .dispatch.dispatcher import Dispatcher, Handle, Verdict
from .dispatch.onboard import HostSetup, Onboarding, facts_command, read_facts
from .dispatch.schedulers import pick
from .dispatch.schedulers.base import Resources
from .dispatch.wrapping import connection, missing, wrap
from .engines.compile.provisioner import Provisioner, task_line
from .engines.runtimes import resolve
from .experiments.fleet import Fleet
from .manifest.loading import load
from .monitor import Monitor
from .probe.snapshot import HostFacts

if TYPE_CHECKING:
    from collections.abc import Callable

    from plumbum.commands.base import BaseCommand

    from .context.plan import ExecutionPlan
    from .dispatch.onboard import Watcher
    from .dispatch.schedulers.base import JobState
    from .manifest.schema.root import Manifest


class Job:
    """One dispatched run, addressed as an object instead of handle flags."""

    def __init__(self, board: Board, handle: Handle) -> None:
        """board: the host-bound board that submitted this job.

        handle: the dispatch handle identifying it on the scheduler.
        """
        self.board = board
        self.handle = handle

    def kill(self) -> None:
        """Cancel the job on its scheduler."""
        with connection(self.handle.host) as remote:
            pick(self.board.plan().profile).cancel(remote, self.handle.root, handle=self.handle.id)

    def logs(self) -> str:
        """The job's captured log so far, merged stdout and stderr."""
        with connection(self.handle.host) as remote:
            scheduler = pick(self.board.plan().profile)
            return scheduler.logs(remote, self.handle.root, handle=self.handle.id)

    def pull(self) -> None:
        """Bring the job's recorded results path back to this machine."""
        self.board.dispatcher.fetch(self.handle)

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
    """One provider-dispatched run, the transport-free twin of `Job`."""

    def __init__(self, backend: ProviderBackend, handle: str) -> None:
        """backend: the provider backend instance that submitted this run.

        handle: the provider's opaque handle id.
        """
        self.backend = backend
        self.handle = handle

    def kill(self) -> None:
        """Cancel the run on the provider."""
        self.backend.cancel(self.handle)

    def logs(self) -> str:
        """The run's captured log so far."""
        return self.backend.logs(self.handle)

    def pull(self, path: str) -> None:
        """Bring the run's output at `path` back to this machine."""
        self.backend.deliver(self.handle, path=path)

    def wait(
        self, *, interval: float = 15.0, poll: Callable[[float], None] = time.sleep
    ) -> Verdict:
        """Poll the provider until the run is terminal and return its verdict.

        interval: seconds between provider state polls.
        poll: the sleeper between polls, injectable for tests.
        """
        while True:
            state = self.backend.state(self.handle)
            if state.verdict in {"ok", "failed", "vanished", "unknown"}:
                return Verdict(verdict=state.verdict, exit_code=state.exit_code)
            poll(interval)


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

    @property
    def dispatcher(self) -> Dispatcher:
        """The dispatch core, shared across every host this board pivots onto."""
        built = self.shared.setdefault("dispatcher", None) or Dispatcher()
        self.shared["dispatcher"] = built
        return cast("Dispatcher", built)

    @property
    def local(self) -> bool:
        """Whether this board is bound to the current machine."""
        return self.host == "local"

    @property
    def manifest(self) -> Manifest:
        """The loaded workspace manifest, shared across every `on` pivot."""
        loaded = self.shared.setdefault("manifest", None) or load(
            self.root / self.project.manifest
        )
        self.shared["manifest"] = loaded
        return cast("Manifest", loaded)

    @property
    def resolver(self) -> Resolver:
        """The plan resolver over this workspace's manifest."""
        built = self.shared.setdefault("resolver", None) or Resolver(self.manifest)
        self.shared["resolver"] = built
        return cast("Resolver", built)

    def containerizer(
        self, plan: ExecutionPlan, root: str
    ) -> Callable[[list[str]], list[str]] | None:
        """The container argv builder for `plan`, None when the plan is bare."""
        if not plan.containerized or plan.container is None:
            return None
        runtime = resolve(plan.container.runtime)()
        container = plan.container
        return lambda argv: runtime.command(container, prefix_bind=plan.prefix(root), argv=argv)

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

    def job(self, handle: str | int, *, host: str = "") -> Job:
        """The dispatched job `handle`, rebuilt from the dispatch cache.

        A fresh process addresses an already-running job the same way the process that
        submitted it did, without reassembling a `Handle` from the run registry and the host
        profile by hand.

        handle: the scheduler handle the job was dispatched under.
        host: the alias to disambiguate a handle recorded on several hosts.
        """
        record = self.dispatcher.cache.run(str(handle), host or None)
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

    def install(
        self,
        env: str = "",
        *,
        resolve: bool = False,
        profile: str = "",
        watch: Watcher | None = None,
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
        """
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
            ).run()
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
        return bound

    def plan(self, *, env: str = "", container: str = "") -> ExecutionPlan:
        """The resolved execution plan for this board's host."""
        return self.resolver.plan(self.host, env=env, container=container)

    def remote_root(self) -> str:
        """The declared workspace root on the bound host, refusing when absent."""
        root = self.plan().profile.root
        if not root:
            raise MissionError(
                f"host {self.host!r} declares no root; set [hosts.{self.host}] root"
            )
        return root

    def run(self, command: str, *, env: str = "", container: str = "") -> int:
        """Run `command` through the host's activated plan, returning its exit code.

        Locally the wrapped line executes in place; remotely it rides one ssh
        connection. Either way the same staging applies, cd, PATH, modules,
        then the environment or the container. A command naming a declared task
        is resolved by pixi inside the generated workspace instead of by the
        shell, which is what makes `run test` and `run -- pytest -q` the same verb.

        command: the shell command to run, or a declared task name and its arguments.
        env: an environment name overriding the profile's choice.
        container: a container override, `none` forcing bare.
        """
        plan = self.plan(env=env, container=container)
        root = str(self.root) if self.local else self.remote_root()
        line = wrap(
            plan,
            root,
            command=task_line(self.manifest, command, env=plan.env),
            containerize=self.containerizer(plan, root),
        )
        if self.local:
            return _streamed(localhost["bash"]["-lc", line])
        with connection(self.host) as remote:
            return _streamed(remote["bash"]["-lc", line])

    def shell(
        self, env: str = "", *, replace: Callable[[str, list[str]], NoReturn] = os.execv
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

        env: the environment name, the host profile's own when empty.
        replace: the process-replacing exec, injectable so a test can read the argv it built.
        """
        if not self.local:
            raise MissionError(
                f"a shell runs on this machine only. Ssh to {self.host} and run "
                f"`{self.project.name} shell` there."
            )
        plan = self.plan(env=env, container="none")
        pixi = Provisioner(self.root, self.manifest).pixi
        if not pixi.ready(plan.env):
            raise MissionError(missing(plan, plan.prefix(str(self.root))))
        binary = str(pixi.command.executable)
        replace(binary, [binary, "shell", *pixi.scope(), "--frozen", "-e", plan.env])

    def submit(
        self,
        command: str,
        *,
        name: str = "",
        queue: str = "",
        walltime: str = "",
        mem_gb: int = 0,
        gpus: int = 0,
        nodes: int = 1,
        attempt: int = 1,
        fetch: str | None = None,
        env: str = "",
        container: str = "",
    ) -> Job:
        """Dispatch `command` as a job on this host and return it as a `Job`.

        Unset resources fall back to the host profile's declared defaults,
        with expression-valued defaults evaluated against `attempt` so a
        retry escalates instead of dying to the same ceiling twice.

        command: the command the generated job runs.
        attempt: the 1-based try number, feeding the default expressions.
        fetch: a results path recorded for `Job.pull`.
        """
        plan = self.plan(env=env, container=container)
        defaults = plan.profile.defaults
        memory = mem_gb or (evaluate(defaults.mem_gb, attempt=attempt) if defaults.mem_gb else 0)
        resources = Resources(
            queue=queue or defaults.queue,
            walltime=walltime or defaults.walltime,
            mem_gb=memory,
            gpus=gpus or defaults.gpus,
            nodes=nodes,
            account=plan.profile.account,
        )
        destination = route(plan.profile)
        if destination != "ssh-family":
            return ProviderJob(destination(), destination().submit(plan, command, resources))
        root = self.remote_root()
        handle = self.dispatcher.run(
            plan,
            command,
            root=root,
            resources=resources,
            name=name,
            fetch=fetch,
            containerize=self.containerizer(plan, root),
        )
        return Job(self, handle)


def _streamed(command: BaseCommand) -> int:
    """Run a bound command with inherited stdio, returning its exit code."""
    try:
        command & FG
    except ProcessExecutionError as error:
        return int(error.retcode or 1)
    else:
        return 0
