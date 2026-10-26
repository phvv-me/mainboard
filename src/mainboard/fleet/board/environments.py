import os
import shlex
from enum import StrEnum, auto
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn

from ...context.admission import admit
from ...core.errors import MissionError
from ...core.execution.shell import become, interactive_shell
from ...core.project import Project
from ...dispatch.backends.base import route
from ...dispatch.hosts.onboard import (
    HostSetup,
    Onboarding,
)
from ...dispatch.jobs.vocabulary import Allocation, Resources
from ...dispatch.mirroring.shipment import Shipment
from ...dispatch.mirroring.snapshots import Snapshots, queued_here
from ...dispatch.schedulers import pick
from ...dispatch.transport.shells import Dialect, open_shell
from ...dispatch.wrapping import missing, wrap
from ...engines.compile import backend as backend_module
from ...engines.compile.constants import MANIFEST
from ...engines.compile.pixi.paths import self_installed
from ...engines.compile.pixi.policy import LockPolicy
from ...engines.compile.prefixes import Prefixes, digest_of, prefix_path
from ...engines.compile.provisioner import Provisioner, environment_shard, task_line
from ...engines.compile.state import SyncState
from ...engines.pack import Formats, Packed, pack
from ...engines.runtimes import resolve
from ...jobs.call import Fresh
from ...jobs.closure import Closure
from ...jobs.targets.target import Target
from ...log import logger
from .workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ...context.plan import ExecutionPlan
    from ...dispatch.shared import Watcher


_SSH_FAMILY: Final = "ssh-family"

# An interactive session as the host's profile shapes it, one node, not held.
_ALLOCATION = Allocation()


class Onboard(StrEnum):
    """How much of a host's onboarding runs: all of it, all of it with the workspace's dotfiles,
    or a sync (re-mirror and re-provision, the tool and the hardware probe left as they are,
    the dotfiles as the setup chose)."""

    MINIMAL = auto()
    DOTFILES = auto()
    SYNC = auto()


class Environments(Workspace):
    """The workspace's environments: installed, provided, packed and entered."""

    @staticmethod
    def _session(plan: ExecutionPlan, allocation: Allocation) -> Resources:
        """What an interactive session on `plan`'s host asks for, admitted by its queue policy."""
        defaults = plan.profile.defaults
        resources = Resources(
            queue=allocation.queue or defaults.interact_queue or defaults.queue,
            walltime=allocation.walltime or defaults.walltime,
            gpus=defaults.gpus,
            nodes=allocation.nodes,
            account=plan.profile.account,
        )
        admit(
            plan.profile, queue=resources.queue or "", walltime=resources.walltime or "", mem_gb=0
        )
        return resources

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
            f"pinned to ({backend_module.PixiEngine.VERSION}) rewrites the lock while "
            "provisioning and moves the address with it too. Run "
            f"`{self.project.name} host setup {self.host}` from the dispatching workspace, which "
            "ships this machine both the pinned pixi and the compile the dispatch addressed."
        )

    def __release(self, prefixes: Prefixes, *, keep: str) -> None:
        """Let go of the source snapshots and environments nothing names any more.

        This machine holds both the prefixes and the trees that point at them, so building is
        the moment, and one inventory of what is still named serves both.

        keep: the digest just built, never dropped.
        """
        snapshots = Snapshots(str(self.root), out=self.project.out_dir(self.root))
        inventory = snapshots.inventory(busy=queued_here())
        if gone := snapshots.prune(inventory):
            logger.info("dropped {} unused source snapshot(s)", len(gone))
        if dropped := prefixes.prune(live={keep}, inventory=inventory):
            logger.info(
                "dropped {} unreferenced environment(s): {}", len(dropped), ", ".join(dropped)
            )

    def _addressed(self, plan: ExecutionPlan, root: str) -> str:
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
            logger.warning("dispatching without an addressed environment: {}", unbuilt)
            return ""
        return prefix_path(root, environment=plan.env, digest=digest, out=Project().out_dir())

    def _containerizer(
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

    def _dotfiles(self) -> bool:
        """Whether this host was set up `--dotfiles`, which a sync keeps; False if never set up."""
        try:
            return self.dispatcher.cache.hosts.recorded(self.host).dotfiles
        except LookupError:
            return False

    def _kept(self, staged: str, *, nodes: int) -> str:
        """`staged` inside the tmux session this workspace keeps on its host.

        `new-session -A` attaches to the named session when it exists and only otherwise starts
        one, so the same verb both opens and returns to a held allocation; named by the package,
        so a session held under an older release is still the one found.
        """
        held = f"{self.project.package}-{self.host}" + (f"-n{nodes}" if nodes > 1 else "")
        return f"tmux new-session -A -s {shlex.quote(held)} {shlex.quote(staged)}"

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
        snapshot copies. An out-of-root path dependency is compiled at `<state>/vendor/<dist>`
        (see `engines.compile.vendor`), inside the root, so it arrives here with the rest.
        """
        where = Provisioner(self.root, self.manifest).environment_dir(plan.env)
        try:
            compiled = (where / MANIFEST).read_text(encoding="utf-8")
        except OSError:
            return ()
        packages = self_installed(compiled, generated_dir=environment_shard(plan.env, self.root))
        return tuple(
            f"{package}/src".lstrip("/") if (self.root / package / "src").is_dir() else package
            for package in packages
        )

    def install(
        self,
        env: str = "",
        *,
        policy: LockPolicy = LockPolicy.UPDATE,
        install: bool = True,
        profile: str = "",
    ) -> HostSetup:
        """Install an environment on this machine, in place.

        An empty `env` means the profile's declared choice rather than `default`, so a machine
        installs what the manifest says it runs.

        env: the environment name, the host profile's own when empty.
        policy: what may happen to the lock first, as pixi's flags say it.
        install: install what the lock pins; off, this machine's lock alone moves.
        profile: the declared host profile describing this machine, so the generated activation
            carries that host's module stack; this board's own host when empty.
        """
        if not self.local:
            raise MissionError(
                f"install builds this machine's environment; `{self.project.name} host setup "
                f"{self.host}` onboards that host"
            )
        plan = self.resolver.plan(profile or self.host, env=env, container="none")
        provisioner = Provisioner(self.root, self.manifest)
        provisioner.provision(plan.env, policy=policy, install=install)
        # A platform this machine cannot run has no prefix to activate here; its lock ships with
        # `setup`.
        activate = (
            str(provisioner.activate(plan.env, modules=plan.profile.modules))
            if install and provisioner.runs_here(plan.env)
            else ""
        )
        return HostSetup(
            host=self.host,
            root=str(self.root),
            env=plan.env,
            activate=activate,
            installer="in-place",
            tool=version(self.project.package),
        )

    def interact(
        self,
        *command: str,
        env: str = "",
        allocation: Allocation = _ALLOCATION,
        replace: Callable[[str, list[str]], NoReturn] = become,
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
                f"out no terminal. Run `{self.project.name} job submit --on {self.host}` instead."
            )
        try:
            self.dispatcher.cache.hosts.recorded(self.host)
        except LookupError as unset:
            raise MissionError(str(unset).strip("'\"")) from None
        session = pick(plan.profile).interactive(
            env=plan.env, command=command, resources=Environments._session(plan, allocation)
        )
        dialect = Dialect()
        staged = dialect.stage(plan, self.remote_root(), command=session, activate=False)
        if allocation.keep:
            staged = self._kept(staged, nodes=allocation.nodes)
        # A bounded transport suits a poll, not a session, so the user's ssh config owns this one.
        argv = dialect.session(self.host, line=staged)
        replace(argv[0], argv)

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
            containerize=self._containerizer(plan, root),
        )

    def pack(self, env: str, formats: Formats) -> Packed:
        """`env` built into artifacts on this host, by its own tool (`mb pack`).

        The package mirrors are close there, and nothing crosses from this machine but the request.
        """
        if self.local:
            return pack(self.root, env, formats)
        line = shlex.join([self.project.name, "pack", env, *formats.flags, "--json"])
        with open_shell(self.plan(container="none"), self.remote_root()) as shell:
            said = shell.run(line, activate=True)
        answer = next((row for row in reversed(said.splitlines()) if row.startswith("{")), "")
        if not answer:
            raise MissionError(f"{self.host} answered no pack: {said.strip()[-240:]}")
        return Packed.model_validate_json(answer)

    def provide(self, env: str = "", *, source: str = "", expect: str = "") -> Path:
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
        self.__release(prefixes, keep=built.name)
        return built

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

    def setup(
        self,
        env: str = "",
        *,
        policy: LockPolicy = LockPolicy.UPDATE,
        watch: Watcher | None = None,
        onboard: Onboard = Onboard.MINIMAL,
    ) -> HostSetup:
        """Onboard this board's host over ssh.

        Mirror the workspace, install the tool from the mirror, provision with the host's own
        tool, probe what it became. A host installs from the artifact this workspace already
        solved, so its own compiler never enters the lock's dependency path.

        env: the environment name, the host profile's own when empty.
        policy: what may happen to the lock first; a host is passed the same policy, `locked`
            being how it refuses a lock this manifest did not solve.
        watch: announces each onboarding stage as it begins.
        onboard: how much of the onboarding runs, and whether the workspace's dotfiles apply.
        """
        if self.local:
            raise MissionError(
                f"setup and sync onboard a remote host; `{self.project.name} install` builds "
                "this machine's environment"
            )
        plan = self.resolver.plan(self.host, env=env, container="none")
        provisioner = Provisioner(self.root, self.manifest)
        compiler = provisioner.compiler_for(plan.env)
        if policy is LockPolicy.LOCKED:
            # The host will refuse a lock this manifest did not solve; ask here first, before
            # the mirror and the remote install spend minutes reaching that answer.
            compiler.vouch()
        sync = onboard is Onboard.SYNC
        dotfiles = self._dotfiles() if sync else onboard is Onboard.DOTFILES
        return Onboarding(
            self.dispatcher,
            plan,
            artifact=provisioner.artifact_for(plan.env),
            policy=policy,
            watch=watch,
            digest=compiler.digest(),
            floor=self.floor,
            dotfiles=self.manifest.workspace.dotfiles if dotfiles else "",
        ).run(sync_only=sync)

    def shell(
        self,
        env: str = "",
        *,
        policy: LockPolicy = LockPolicy.UPDATE,
        install: bool = True,
        replace: Callable[[str, list[str], Mapping[str, str]], NoReturn] = become,
    ) -> NoReturn:
        """Hand this terminal to the user's own shell with the workspace environment entered.

        As `pixi shell` does, the lock is first brought to `policy` and the environment installed
        when it is not, then its activation applied and this process becomes the shell, so it
        owns the terminal and leaving it lands where the user began. The environment's
        `activate.sh` (modules, pixi, second-stage tools) is sourced by `sh`, which then becomes
        the shell. With `install` off, an unprovisioned environment is refused naming the fix.

        env: the environment name, the host profile's own when empty.
        policy: what may happen to the lock first, as pixi's flags say it.
        install: install the environment, or bring it in line with its lock, first.
        replace: the process-replacing exec, injectable so a test can read what it was handed.
        """
        if not self.local:
            raise MissionError(
                f"a shell runs on this machine only. Run "
                f"`{self.project.name} shell --on {self.host}` for a session there."
            )
        plan = self.plan(env=env, container="none")
        provisioner = Provisioner(self.root, self.manifest)
        provisioner.refreshed(plan.env, policy=policy, install=install)
        pixi = provisioner.pixi_for(plan.env)
        if not pixi.ready(plan.env):
            raise MissionError(missing(plan, plan.prefix(str(self.root))))
        script = self.root / self.project.activation(plan.env, self.root)
        if not script.is_file():
            if not install:
                tool = self.project.name
                raise MissionError(f"{plan.env!r} has no activation; run `{tool} install`")
            script = provisioner.activate(plan.env, modules=plan.profile.modules)
        entering = ["/bin/sh", "-c", '. "$1" && shift && exec "$@"', "mb-shell", str(script)]
        replace(entering[0], [*entering, *interactive_shell()], os.environ | pixi.overrides)
