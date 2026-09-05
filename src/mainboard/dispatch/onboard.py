# Onboarding a host: mirror the workspace, put the tool on the machine, provision the manifest's
# environment there, and read the host back through the activation that install just wrote. The
# successor to the shell script the previous generation shipped, expressed over the transports
# dispatch already owns rather than a second, parallel way to reach a host.
#
# How the tool gets onto the machine depends on the workspace rather than on the machine. A
# workspace that vendors the tool's own source installs from that source, which is what keeps a
# host from ever running a tool older than the manifest it is about to compile. A workspace that
# consumes the tool from an index has no source to ship, so it installs the version it declares
# from the index, or keeps the one the machine already runs when that already satisfies it.
# Assuming the first shape is what refused to onboard a standalone workspace at all, blaming a
# host whose uv, pip and mainboard were all present for a directory nobody had shipped it.

import shlex
from typing import TYPE_CHECKING

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from patos import FrozenModel, Resolution, Strategy, StrategyError

from ..core.errors import MissionError
from ..core.project import Project
from ..engines.compile.backend import PIXI_VERSION, POSIX_INSTALLER
from ..probe.snapshot import HostFacts
from .schedulers.base import failure_reason
from .schedulers.pueue import Pueue
from .schedulers.registry import pick
from .shared import Watcher, announce, logger
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
    pixi: the pixi version the machine runs, which every setup and sync brings to the fleet's
        one pinned version, so a host that quietly moved shows here instead of in a dead wave.
    capabilities: the host as the bootstrap probe found it, None for an in-place install.
    hardware: the host's hardware snapshot, read back through the new activation.
    onboarded_at: ISO-8601 time the install finished.
    synced_at: ISO-8601 time the workspace was last mirrored here, empty until one lands after
        the onboarding that first mirrored it.
    digest: the manifest digest this host was last provisioned from, empty for a host onboarded
        before this field existed. `doctor` compares it against the manifest as it reads now to
        say whether a host has drifted out of sync since it was set up.
    """

    host: str
    root: str
    env: str = "default"
    activate: str = ""
    installer: str = ""
    rejected: tuple[tuple[str, str], ...] = ()
    tool: str = ""
    pixi: str = ""
    capabilities: Facts | None = None
    hardware: HostFacts | None = None
    onboarded_at: str = ""
    synced_at: str = ""
    digest: str = ""

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


def specifier(floor: str) -> str:
    """The declared version as a requirement operator carries it, empty when nothing is declared.

    A manifest writes a version the way its own resolver spells one, so `">=0.4.8"` arrives with
    its operator and `"0.4.8"` arrives without: an unadorned version means that exact one.

    floor: the version the workspace declares for the tool, `*` or empty for any.
    """
    if floor in {"", "*"}:
        return ""
    return floor if floor[0] in "<>=!~" else f"=={floor}"


def satisfied_by(found: str, floor: str) -> bool:
    """Whether the version `found` already meets what the workspace declares.

    A workspace that declares no version is satisfied by any tool at all, since there is nothing
    to compare against and the machine's own is what it asked for. A version neither side can
    parse is not treated as satisfied, so the install happens rather than being skipped on a
    string nobody understood.

    found: the version the machine reports, empty when it runs no tool.
    floor: the version the workspace declares.
    """
    if not found:
        return False
    wanted = specifier(floor)
    if not wanted:
        return True
    try:
        return SpecifierSet(wanted).contains(Version(found), prereleases=True)
    except InvalidSpecifier, InvalidVersion:
        return False


class Existing(Installer):
    """The route that installs nothing, because the machine already runs what was asked for.

    Only ever offered to a workspace that vendors no source: one that does always reinstalls
    from it, since the whole point of shipping the source is that the host runs the tool the
    manifest was written against.
    """

    def __init__(self, shell: RemoteShell, *, floor: str) -> None:
        """shell: the host shell the version is read through.

        floor: the version the workspace declares for the tool.
        """
        super().__init__(shell, probe=f"command -v {_TOOL}", command="true")
        self.floor = floor

    def available(self) -> bool:
        """Whether the machine's own tool already satisfies the declared version."""
        return satisfied_by(installed_version(self.shell), self.floor)


def installed_version(shell: RemoteShell) -> str:
    """The tool version the machine already runs, empty when it runs none.

    shell: the host shell the question is asked through.
    """
    try:
        return shell.run(f"{_TOOL} --version").strip().split()[-1]
    except MissionError, IndexError:
        return ""


def installed_pixi(shell: RemoteShell) -> str:
    """The pixi version the machine runs as `X.Y.Z`, empty when it runs none.

    shell: the host shell the question is asked through.
    """
    try:
        return shell.run("pixi --version").strip().split()[-1]
    except MissionError, IndexError:
        return ""


def installers(
    shell: RemoteShell, source: str = _SOURCE, *, vendored: bool = True, floor: str = ""
) -> Strategy[Installer]:
    """The ordered install routes for `shell`'s host, best first.

    uv installs the tool as its own isolated tool environment, which is why it leads: it needs
    no interpreter on the host new enough to run the tool itself. Where uv is absent but the
    host can fetch it, bootstrapping uv beats falling back to a user-site pip install, which is
    the last route and the only one bound to whatever `python3` the host happens to ship.

    Which family of routes is offered is a fact about the workspace. One that vendors the tool's
    source installs from that source, so the host can never run a tool older than the manifest
    it is about to compile. One that consumes the tool from an index installs the version it
    declares, and is first offered the machine's own tool, since a machine already running what
    the manifest asks for needs nothing installed at all.

    shell: the host shell each route probes and installs through.
    source: the tool's source directory inside the synced workspace.
    vendored: whether that source is actually on the machine.
    floor: the version the workspace declares for the tool, read from its own manifest.
    """
    strategy: Strategy[Installer] = Strategy(f"{_TOOL} installer")
    if vendored:
        quoted = shlex.quote(source)
        strategy.register(
            "uv",
            Installer(
                shell,
                probe="command -v uv",
                command=f"uv tool install --force --editable {quoted}",
            ),
        )
        strategy.register(
            "uv-bootstrap",
            Installer(
                shell,
                probe="command -v curl",
                command=f"{_UV_INSTALLER} && uv tool install --force --editable {quoted}",
            ),
        )
        strategy.register(
            "pip",
            Installer(
                shell,
                probe="python3 -m pip --version",
                command="python3 -m pip install --user --break-system-packages "
                f"--force-reinstall --editable {quoted}",
            ),
        )
        return strategy
    wanted = shlex.quote(f"{_TOOL}{specifier(floor)}")
    strategy.register("present", Existing(shell, floor=floor))
    strategy.register(
        "uv-index",
        Installer(shell, probe="command -v uv", command=f"uv tool install --force {wanted}"),
    )
    strategy.register(
        "uv-bootstrap-index",
        Installer(
            shell,
            probe="command -v curl",
            command=f"{_UV_INSTALLER} && uv tool install --force {wanted}",
        ),
    )
    strategy.register(
        "pip-index",
        Installer(
            shell,
            probe="python3 -m pip --version",
            command=f"python3 -m pip install --user --break-system-packages --upgrade {wanted}",
        ),
    )
    return strategy


class Bootstrap:
    """Puts this workspace's tool and environment onto a machine that already answers ssh.

    The half of onboarding that is about the machine rather than about the record kept of it:
    install the tool from the synced source through the first route the machine supports, then
    have that tool compile the synced manifest and install the environment from the lock this
    workspace already solved. A declared host and a machine rented for one job need exactly these
    two steps and differ in everything around them, so both drive this one class rather than two
    copies drifting apart.

    shell: the machine's shell both steps probe, install and provision through.
    resolve: let the machine run its own dependency solve instead of installing the shipped lock.
    floor: the version the workspace declares for the tool, used only when the workspace vendors
        no source and the tool therefore comes from an index.
    """

    def __init__(self, shell: RemoteShell, *, resolve: bool = False, floor: str = "") -> None:
        self.shell = shell
        self.resolve = resolve
        self.floor = floor

    @property
    def env(self) -> str:
        """The environment being provisioned, the plan's own."""
        return self.shell.plan.env

    def tool(self) -> Resolution[Installer]:
        """Install the tool through the first route the machine supports, keeping the rejections.

        The routes offered depend on whether the mirror actually carries the tool's source, which
        is asked of the machine rather than assumed: a workspace that consumes the tool from an
        index ships no such directory, and every route used to test for it, so all three refused
        and the refusal blamed the host's tooling for something the workspace had never sent.

        A machine that supports no route fails here rather than inside a provisioning step that
        assumed the tool, and the refusal names the condition that actually decided it.
        """
        host = self.shell.plan.host
        vendored = self.shell.ok(f"[ -d {shlex.quote(_SOURCE)} ]")
        routes = installers(self.shell, vendored=vendored, floor=self.floor)
        try:
            resolution = routes.cascade()
        except StrategyError as refused:
            raise MissionError(
                self.unreachable(host, vendored=vendored, refused=refused)
            ) from None
        routes.select(resolution.winner).install()
        return resolution

    def unreachable(self, host: str, *, vendored: bool, refused: StrategyError) -> str:
        """Why no route could put the tool on `host`, in terms of what was actually missing.

        The rejection log alone says every route reported itself unavailable, which reads as a
        host with no tooling and was wrong about the one case it mattered: the routes were all
        testing for a source directory the workspace never ships. So the line leads with which
        family was being tried and why, and carries the log behind it.

        host: the alias being onboarded.
        vendored: whether the mirror carries the tool's own source.
        refused: what the cascade said about each route it passed over.
        """
        where = (
            f"installing from the source this workspace vendors at {_SOURCE}"
            if vendored
            else f"installing {_TOOL}{specifier(self.floor) or ' (no version declared)'} from an "
            "index, since this workspace vendors no source"
        )
        return f"cannot install {_TOOL} on {host!r} by {where}: {refused}"

    def environment(self) -> None:
        """Have the machine's own tool compile the synced manifest and install `env` from it.

        The machine is told which declared profile describes it, so the activation script it
        generates carries that profile's module stack rather than this machine's.
        """
        host, root = self.shell.plan.host, self.shell.root
        resolve = " --resolve" if self.resolve else ""
        self.shell.run(
            f"{_TOOL} install {shlex.quote(self.env)}{resolve} --profile {shlex.quote(host)}"
        )
        script = activation(root, env=self.env)
        if not self.shell.ok(f"test -f {shlex.quote(script)}"):
            raise MissionError(
                f"{host!r} has no {script} after installing {self.env!r}; "
                "the environment was not provisioned"
            )


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
    digest: the manifest digest this onboarding provisions from, stamped onto the recorded
        `HostSetup` so `doctor` can later tell this host apart from one the manifest outgrew.
    floor: the version this workspace declares for the tool itself, which is what a host with no
        vendored source installs from an index; empty when the workspace declares none.
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
        digest: str = "",
        floor: str = "",
    ) -> None:
        self.dispatcher = dispatcher
        self.plan = plan
        self.root = root
        self.artifact = tuple(artifact)
        self.resolve = resolve
        self.watch = watch or announce
        self.digest = digest
        self.floor = floor

    @property
    def env(self) -> str:
        """The environment provisioned, the plan's own."""
        return self.plan.env

    def align_pixi(self, shell: RemoteShell, *, host: str) -> str:
        """Put the fleet's one pixi on `host`, whichever one it runs now, and say which that is.

        A lock is pixi's file, not this package's, and every pixi version writes some of it
        differently. A host on another version therefore rewrites the lock it was shipped while
        provisioning, and an environment addressed by the content of that lock moves address
        under a wave that was dispatched against the old one: on 2026-09-05 a workstation on
        0.77 pinned 4950b228a3eaf208, this host on 0.79 built 4e0f0670076776b1, and every job
        died with `found no built environment`.

        So a host is brought in line rather than tolerated, in either direction. The older-is-
        refused check this replaces let a newer pixi through, which is exactly the case that
        happened. The installer is pixi's own, reading the pinned version from its `PIXI_VERSION`,
        and it writes into `$HOME/.pixi/bin`, which every wrapped command already puts ahead of
        the system PATH. A host that still disagrees afterwards is refused rather than provisioned
        into a lock nothing here can predict.

        shell: the host shell the version is read and the installer run through.
        host: the alias being brought in line.
        """
        theirs = installed_pixi(shell)
        if theirs == PIXI_VERSION:
            return theirs
        self.watch(f"putting pixi {PIXI_VERSION} on {host}, which runs {theirs or 'none'}")
        shell.run(POSIX_INSTALLER)
        aligned = installed_pixi(shell)
        if aligned != PIXI_VERSION:
            raise MissionError(
                f"{host!r} still runs pixi {aligned or 'none'} after installing {PIXI_VERSION}; "
                f"every lock this workspace ships was solved by {PIXI_VERSION}, and a host on "
                "another one rewrites it while provisioning and builds a different environment "
                "than the one a dispatch pins. Put that version on the host's PATH by hand, then "
                "set it up again."
            )
        return aligned

    def verify_queue(self, shell: RemoteShell, *, host: str) -> None:
        """Make sure the queue daemon a plain ssh host dispatches through is answering.

        pueue is assumed running on such a host and every later `submit` fails on its socket
        when it is not, so the daemon is started here when it is down and the host refused,
        naming the fix, when it still does not answer.
        """
        if not isinstance(pick(self.plan.profile), Pueue) or shell.ok("pueue status"):
            return
        shell.run("pueued -d")
        if not shell.ok("pueue status"):
            raise MissionError(
                f"pueued is not answering on {host!r}; install pueue there and start it with "
                "`pueued -d`, then set the host up again"
            )

    def undisturbed(self, host: str) -> None:
        """Refuse to change the environment under runs `host` still owes an outcome for.

        Every pinned tree on a host symlinks its environment back to the one prefix in the
        mirror, so shipping a compiled manifest that differs from the one those runs were pinned
        and primed against replaces that environment underneath them. Their own pixi then finds
        a prefix that does not match the manifest they hold, the new wave's finds one that does
        not match theirs, and the two fight over it: job 3296353 of a five-job wave died that
        way while a batch from a newer commit was dispatched around it (2026-09-05).

        Only a real change is refused. A host whose recorded digest already matches this
        manifest is being re-mirrored rather than re-described, which is what a sync between two
        waves of one campaign does all day, and a host with nothing owed can be told anything.

        host: the alias about to be mirrored to.
        """
        if not self.digest:
            return
        try:
            recorded = self.dispatcher.cache.host(host)
        except LookupError:
            return
        if not recorded.digest or recorded.digest == self.digest:
            return
        owed = [run for run in self.dispatcher.cache.live() if run.target == host]
        if not owed:
            return
        named = ", ".join(run.handle for run in owed[:4])
        more = f" and {len(owed) - 4} more" if len(owed) > 4 else ""
        raise MissionError(
            f"{host!r} still owes {len(owed)} run(s) an outcome ({named}{more}) and this "
            f"workspace's environment has changed since {host!r} was set up. Every one of those "
            "runs activates the environment this would replace, so shipping it now would change "
            f"what they run in while they wait. Let them settle (`{_TOOL} jobs`), or "
            f"`{_TOOL} cancel <handle>` the ones you no longer need, then set the host up again."
        )

    def run(self, *, sync_only: bool = False) -> HostSetup:
        """Onboard the host and return (and record) what it became.

        The mirror carries the compiled artifact alongside the sources, so the install step
        below has a lock this workspace already solved and never asks the host to solve one.

        Refuses outright when the environment this would install differs from the one runs still
        in flight on that host are activating, since they all share the one prefix the mirror
        holds and nothing here can give them the environment they were dispatched with once it
        has been replaced.

        sync_only: skip the bootstrap and the hardware probe, re-mirroring and re-provisioning
            an already onboarded host instead of onboarding it from nothing; see `_sync`.
        """
        host = self.plan.host
        self.undisturbed(host)
        if sync_only:
            return self._sync(host)
        with connection(host) as remote:
            self.watch(f"probing {host}")
            capabilities = probe_capabilities(remote, host)
            root = self.root or find_root(remote)
            shell = RemoteShell(remote, self.plan, root)
            bootstrap = Bootstrap(shell, resolve=self.resolve, floor=self.floor)
            self.watch(f"mirroring the workspace to {host}:{root}")
            self.dispatcher.rsync_up(
                self.plan, root, required=[self.artifact] if self.artifact else []
            )
            self.watch(f"installing {_TOOL} on {host}")
            winner = bootstrap.tool()
            self.watch(f"checking pixi and the queue on {host}")
            pixi = self.align_pixi(shell, host=host)
            self.verify_queue(shell, host=host)
            self.watch(f"provisioning {self.env} on {host}")
            bootstrap.environment()
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
                pixi=pixi,
                capabilities=capabilities,
                hardware=hardware,
                digest=self.digest,
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

        THE PIXI IS ALIGNED HERE TOO, before the provision rather than only at setup. A sync is
        what a campaign runs between waves, so a host whose pixi moved under it would otherwise
        rewrite the shipped lock and build a different environment than the wave was dispatched
        against, with nothing in between ever asking.

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
            pixi = self.align_pixi(shell, host=host)
            self.watch(f"provisioning {self.env} on {host}")
            Bootstrap(shell, resolve=self.resolve).environment()
        fresh = self.dispatcher.cache.host(host)
        updated = self.dispatcher.cache.save_host(
            fresh.model_copy(
                update={
                    "root": root,
                    "env": self.env,
                    "activate": activation(root, env=self.env),
                    "pixi": pixi,
                    "digest": self.digest or fresh.digest,
                }
            )
        )
        logger.info("synced %s at %s", host, root)
        return updated
