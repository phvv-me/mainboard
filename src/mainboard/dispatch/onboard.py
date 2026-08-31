# Onboarding a host: mirror the workspace, put the tool on the machine, provision the manifest's
# environment there, and read the host back through the activation that install just wrote. The
# successor to the shell script the previous generation shipped, expressed over the transports
# dispatch already owns rather than a second, parallel way to reach a host.

import shlex
from typing import TYPE_CHECKING, Protocol

from patos import FrozenModel, Resolution, Strategy, StrategyError

from ..core.errors import MissionError
from ..core.project import Project
from ..probe.snapshot import HostFacts
from .schedulers.base import failure_reason
from .shared import logger
from .targets import Facts, find_root, probe_capabilities
from .wrapping import activation, connection, wrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context.plan import ExecutionPlan
    from .dispatcher import Dispatcher
    from .transport import Machine

# The tool this workspace answers to, so nothing below spells the name of the binary it installs.
_TOOL = Project().name

# Where the tool's own source sits inside a synced workspace. Installing from it (rather than
# from a published build) is what keeps a host's tool and the manifest it compiles from ever
# drifting apart, the failure the previous generation's setup script existed to prevent.
_SOURCE = f"packages/{_TOOL}"

# uv's official installer, used only when a host has neither uv nor pip to install the tool with.
_UV_INSTALLER = "curl -LsSf https://astral.sh/uv/install.sh | sh"


def facts_command() -> str:
    """The command a machine answers with its own hardware snapshot as JSON."""
    return f"{_TOOL} facts --json"


class Watcher(Protocol):
    """Announces the stage an onboarding has reached, so a long run never stands silent."""

    def __call__(self, stage: str) -> None:
        """stage: what the onboarding is doing now, as one human phrase."""


class HostSetup(FrozenModel):
    """What one machine became after an install: where its workspace is and how it activates.

    The record `Board.install` hands back and the dispatch cache keeps per host, so a later
    command reads what a host is without touching it, and a fallback install route stays a
    stated fact rather than a silent degradation.

    host: the alias onboarded, `local` for an install on this machine.
    root: the workspace root on that machine.
    env: the environment provisioned there.
    activate: the activation script the workspace now carries.
    installer: the install route that won, `in-place` when the tool was already running here.
    rejected: the routes passed over, each with the reason it was not usable.
    tool: the tool version the machine reports once installed.
    capabilities: the host as the bootstrap probe found it, None for an in-place install.
    hardware: the host's hardware snapshot, read back through the new activation.
    onboarded_at: ISO-8601 time the install finished.
    synced_at: ISO-8601 time the workspace was last mirrored here, empty until one lands after
        the onboarding that first mirrored it.
    """

    host: str
    root: str
    env: str = "default"
    activate: str = ""
    installer: str = ""
    rejected: tuple[tuple[str, str], ...] = ()
    tool: str = ""
    capabilities: Facts | None = None
    hardware: HostFacts | None = None
    onboarded_at: str = ""
    synced_at: str = ""

    @property
    def mirrored_at(self) -> str:
        """When this host's copy of the workspace was last brought up to date.

        The latest mirror when one has been recorded since, else the onboarding that first put
        the workspace there. This is the watermark a transfer set measures a delta against, so a
        host nobody has mirrored since being set up still has an honest answer.
        """
        return max(self.synced_at, self.onboarded_at)


class RemoteShell:
    """A host's shell staged by an execution plan, the one way onboarding runs a remote command.

    Two footings: a bare command gets `cd`, the per-user install dirs on `PATH` and the host's
    modules, all an unprovisioned machine can offer, while an activated one additionally sources
    the environment, which is what proves the environment the install just built actually runs.

    remote: the open connection commands ride.
    plan: the resolved execution context staging them.
    root: the workspace root on the host.
    """

    def __init__(self, remote: Machine, plan: ExecutionPlan, root: str) -> None:
        self.remote = remote
        self.plan = plan
        self.root = root

    def ok(self, command: str) -> bool:
        """Whether `command` exits zero on the host, its output discarded."""
        retcode, _, _ = self.__execute(command, activate=False)
        return retcode == 0

    def run(self, command: str, *, activate: bool = False) -> str:
        """`command`'s stdout on the host, raising a `MissionError` naming why it failed.

        command: the command to run in the workspace.
        activate: run it through the plan's activation rather than the bare staging.
        """
        retcode, out, err = self.__execute(command, activate=activate)
        if retcode:
            reason = failure_reason(err or out, retcode)
            raise MissionError(f"`{command}` failed on {self.plan.host!r}: {reason}")
        return str(out)

    def __execute(self, command: str, *, activate: bool) -> tuple[int, str, str]:
        line = wrap(self.plan, self.root, command=command, activate=activate)
        retcode, out, err = self.remote["bash"][["-lc", line]].run(retcode=None)
        return int(retcode), str(out), str(err)


class Installer:
    """One route to putting the tool on a host, probed before the cascade commits to it.

    probe: the shell test deciding whether this route applies on the host.
    command: the shell line that installs the tool once the route wins.
    """

    def __init__(self, shell: RemoteShell, *, probe: str, command: str) -> None:
        self.shell = shell
        self.probe = probe
        self.command = command

    def available(self) -> bool:
        """Whether the host has what this route needs, the cascade's rejection test."""
        return self.shell.ok(self.probe)

    def install(self) -> str:
        """Run the route's install line on the host and return its output."""
        return self.shell.run(self.command)


def installers(shell: RemoteShell, source: str = _SOURCE) -> Strategy[Installer]:
    """The ordered install routes for `shell`'s host, best first.

    uv installs the tool as its own isolated tool environment, which is why it leads: it needs
    no interpreter on the host new enough to run the tool itself. Where uv is absent but the
    host can fetch it, bootstrapping uv beats falling back to a user-site pip install, which is
    the last route and the only one bound to whatever `python3` the host happens to ship. Every
    route installs from the synced source, so a host can never run a tool older than the
    manifest it is about to compile.

    shell: the host shell each route probes and installs through.
    source: the tool's source directory inside the synced workspace.
    """
    strategy: Strategy[Installer] = Strategy(f"{_TOOL} installer")
    present = f"[ -d {shlex.quote(source)} ]"
    strategy.register(
        "uv",
        Installer(
            shell,
            probe=f"command -v uv && {present}",
            command=f"uv tool install --force --editable {shlex.quote(source)}",
        ),
    )
    strategy.register(
        "uv-bootstrap",
        Installer(
            shell,
            probe=f"command -v curl && {present}",
            command=f"{_UV_INSTALLER} && uv tool install --force --editable {shlex.quote(source)}",
        ),
    )
    strategy.register(
        "pip",
        Installer(
            shell,
            probe=f"python3 -m pip --version && {present}",
            command="python3 -m pip install --user --break-system-packages "
            f"--force-reinstall --editable {shlex.quote(source)}",
        ),
    )
    return strategy


def read_facts(text: str) -> HostFacts:
    """The `HostFacts` inside `text`, read from its first `{` so shell chatter above is ignored.

    text: a remote command's captured output ending in the facts JSON.
    """
    start = text.find("{")
    if start < 0:
        raise MissionError(
            f"no host facts in the probe output: {text.strip()[-240:] or '(empty)'}"
        )
    return HostFacts.model_validate_json(text[start:])


class Onboarding:
    """Brings one host from bare ssh access to a workspace that runs jobs.

    The steps are the ones a person would take by hand and in the same order: probe what the
    host is, mirror the workspace onto it, install the tool from that mirror, have the tool
    compile and install the manifest's environment there, then read the host back through the
    activation it now carries. Each step runs over the transports dispatch already owns, so
    onboarding stays one behavior of the dispatch subsystem rather than a second way in.

    The environment provisioned is the plan's own, which is the host profile's declared choice
    unless the caller overrode it. A host that declares `env = "serving"` is therefore set up
    with serving without anyone repeating the name, and the environment the onboarding installs
    can never drift from the one the plan's later commands activate.

    The workstation solves, the host installs. `artifact` rides the mirror through the denylist
    that otherwise keeps the generated directory local, and the host then installs from that
    lock rather than solving again. Solving on the host means reading dependency metadata,
    reading metadata means building source distributions, and that puts the host's own compiler
    in the lock's dependency path, where one machine's toolchain decides whether an unrelated
    platform's requirement can be read at all. `resolve` is the escape hatch for the rare host
    that genuinely must solve for itself.

    dispatcher: the dispatch core whose mirror and state cache the onboarding uses.
    plan: the resolved execution context for the host, container-free by construction.
    root: the workspace root on the host, discovered on the host when empty.
    artifact: the compiled manifest, lock and state that ship with the mirror so the host can
        install frozen; empty leaves the host to solve.
    resolve: let the host run its own dependency solve instead of installing from the artifact.
    watch: announces each stage as it begins.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        plan: ExecutionPlan,
        *,
        root: str = "",
        artifact: Sequence[str] = (),
        resolve: bool = False,
        watch: Watcher | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.plan = plan
        self.root = root
        self.artifact = tuple(artifact)
        self.resolve = resolve
        self.watch = watch or _announce

    @property
    def env(self) -> str:
        """The environment provisioned, the plan's own."""
        return self.plan.env

    def bootstrap(self, shell: RemoteShell) -> Resolution[Installer]:
        """Install the tool through the first route the host supports, keeping the rejections.

        A host that supports no route at all fails here, naming every route and why it was
        refused, rather than failing later inside a provisioning step that assumed the tool.

        shell: the host shell the routes probe and install through.
        """
        routes = installers(shell)
        try:
            resolution = routes.cascade()
        except StrategyError as refused:
            raise MissionError(
                f"cannot install {_TOOL} on {self.plan.host!r}: {refused}"
            ) from None
        routes.select(resolution.winner).install()
        return resolution

    def provision(self, shell: RemoteShell, *, host: str, root: str) -> None:
        """Have the host's own tool compile the synced manifest and install `env` from it.

        The host is told which declared profile describes it, so the activation script it
        generates carries that host's module stack rather than this machine's.

        shell: the host shell the install runs through.
        host: the alias whose declared profile the host provisions itself as.
        root: the workspace root on the host.
        """
        resolve = " --resolve" if self.resolve else ""
        shell.run(
            f"{_TOOL} install {shlex.quote(self.env)}{resolve} --profile {shlex.quote(host)}"
        )
        script = activation(root, env=self.env)
        if not shell.ok(f"test -f {shlex.quote(script)}"):
            raise MissionError(
                f"{host!r} has no {script} after installing {self.env!r}; "
                "the environment was not provisioned"
            )

    def run(self, *, sync_only: bool = False) -> HostSetup:
        """Onboard the host and return (and record) what it became.

        The mirror carries the compiled artifact alongside the sources, so the install step
        below has a lock this workspace already solved and never asks the host to solve one.

        sync_only: skip the bootstrap and the hardware probe, re-mirroring and re-provisioning
            an already onboarded host instead of onboarding it from nothing; see `_sync`.
        """
        host = self.plan.host
        if sync_only:
            return self._sync(host)
        with connection(host) as remote:
            self.watch(f"probing {host}")
            capabilities = probe_capabilities(remote, host)
            root = self.root or find_root(remote)
            shell = RemoteShell(remote, self.plan, root)
            self.watch(f"mirroring the workspace to {host}:{root}")
            self.dispatcher.rsync_up(
                self.plan, root, required=[self.artifact] if self.artifact else []
            )
            self.watch(f"installing {_TOOL} on {host}")
            winner = self.bootstrap(shell)
            self.watch(f"provisioning {self.env} on {host}")
            self.provision(shell, host=host, root=root)
            self.watch(f"reading {host} back through its activation")
            hardware = read_facts(shell.run(facts_command(), activate=True))
            setup = HostSetup(
                host=host,
                root=root,
                env=self.env,
                activate=activation(root, env=self.env),
                installer=winner.winner,
                rejected=winner.rejected,
                tool=shell.run(f"{_TOOL} --version").strip(),
                capabilities=capabilities,
                hardware=hardware,
            )
        recorded = self.dispatcher.cache.save_host(setup)
        logger.info("onboarded %s at %s through %s", host, root, recorded.installer)
        return recorded

    def _sync(self, host: str) -> HostSetup:
        """Re-mirror and re-provision `host`, its bootstrap and hardware probe skipped.

        The fast path back to a host whose environment has drifted from a manifest that moved
        since it was set up: neither the tool nor the hardware changed, only the workspace and
        the environment compiled from it, so nothing here reinstalls or re-probes either.
        Refuses when the host has never been onboarded, since there is nothing yet to sync.

        THE RECORD IS RE-READ AFTER THE MIRROR, NOT BEFORE. `rsync_up` stamps `synced_at` on
        its own, mid-block, and building the saved record from a copy taken before that would
        overwrite the very stamp it just wrote.

        host: the alias to sync, already recorded from a prior `run()`.
        """
        recorded = self.dispatcher.cache.host(host)
        root = self.root or recorded.root
        with connection(host) as remote:
            shell = RemoteShell(remote, self.plan, root)
            self.watch(f"mirroring the workspace to {host}:{root}")
            self.dispatcher.rsync_up(
                self.plan, root, required=[self.artifact] if self.artifact else []
            )
            self.watch(f"provisioning {self.env} on {host}")
            self.provision(shell, host=host, root=root)
        fresh = self.dispatcher.cache.host(host)
        updated = self.dispatcher.cache.save_host(
            fresh.model_copy(
                update={"root": root, "env": self.env, "activate": activation(root, env=self.env)}
            )
        )
        logger.info("synced %s at %s", host, root)
        return updated


def _announce(stage: str) -> None:
    """The default `Watcher`, logging each stage for a caller that renders no progress."""
    logger.info("%s", stage)
