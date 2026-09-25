# Onboarding a host: mirror the workspace, put the tool on the machine, provision the manifest's
# environment there, and read the host back through the activation that install just wrote. It
# runs over the transports dispatch already owns rather than a second, parallel way to reach a
# host, and succeeds the shell script the previous generation shipped.

import shlex
from importlib.metadata import metadata
from typing import TYPE_CHECKING

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from patos import FrozenModel, Resolution, Strategy, StrategyError
from tenacity import Retrying, retry_if_result, stop_after_attempt, wait_fixed

from ..core.errors import MissionError
from ..core.project import Project
from ..engines.compile.backend import PIXI_VERSION
from ..probe.snapshot import HostFacts
from .schedulers.pueue import Pueue
from .schedulers.registry import pick
from .shared import Watcher, announce, logger
from .shells import HostShell, is_windows, open_shell
from .targets import Facts, probe_capabilities, resolve, rooted

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context.plan import ExecutionPlan
    from .dispatcher import Dispatcher

# The tool this workspace answers to, so nothing below spells the name of the binary it installs.
_TOOL = Project().name

# Where the tool's own source sits inside a synced workspace that vendors it.
_SOURCE = f"packages/{_TOOL}"


def gpus_command() -> str:
    """The command a remote host runs to say who holds each of its cards, as JSON."""
    return f"{_TOOL} gpus --json"


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
    activate: the activation script the workspace now carries.
    installer: the install route that won, `in-place` when the tool was already running here.
    rejected: the routes passed over, each with the reason it was not usable.
    tool: the tool version the machine reports once installed.
    pixi: the pixi version the machine runs, which every setup and sync brings to the fleet's
        one pinned version, so a host that quietly moved shows here instead of in a dead wave.
    capabilities: the host as the bootstrap probe found it, carrying the home a `~` root was
        placed under and never a root of its own, since `root` is the one set up there; None
        for an in-place install.
    hardware: the host's hardware snapshot, read back through the new activation.
    onboarded_at: ISO-8601 time the install finished.
    synced_at: ISO-8601 time the workspace was last mirrored here, empty until one lands after
        the onboarding that first mirrored it.
    digest: the manifest digest this host was last provisioned from, empty for a host onboarded
        before this field existed; `doctor` compares it with the manifest to spot drift.
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
        """When this host's workspace copy was last brought up to date: the latest mirror, else
        the onboarding. The watermark a transfer set measures a delta against."""
        return max(self.synced_at, self.onboarded_at)


class Installer:
    """One route to putting the tool on a host, probed before the cascade commits to it.

    probe: the shell test deciding whether this route applies on the host.
    command: the shell line that installs the tool once the route wins.
    """

    def __init__(self, shell: HostShell, *, probe: str, command: str) -> None:
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

    A manifest writes `">=0.4.8"` with its operator and `"0.4.8"` without, which means that
    exact version.

    floor: the version the workspace declares for the tool, `*` or empty for any.
    """
    if floor in {"", "*"}:
        return ""
    return floor if floor[0] in "<>=!~" else f"=={floor}"


def satisfied_by(found: str, floor: str) -> bool:
    """Whether the version `found` (empty when the machine runs no tool) meets `floor`.

    No declared version is satisfied by any tool. An unparsable version is not satisfied, so
    the install happens rather than being skipped on a string nobody understood.
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

    Only offered to a workspace that vendors no source: one that does always reinstalls from it,
    since shipping the source is how the host runs the tool the manifest was written against.
    """

    def __init__(self, shell: HostShell, *, floor: str) -> None:
        super().__init__(shell, probe=shell.dialect.has(_TOOL), command=shell.dialect.noop)
        self.floor = floor

    def available(self) -> bool:
        """Whether the machine's own tool already satisfies the declared version."""
        return satisfied_by(_version(self.shell, _TOOL), self.floor)


def _version(shell: HostShell, program: str) -> str:
    """The version `program --version` reports on the machine (`X.Y.Z`), empty without one."""
    try:
        return shell.run(f"{program} --version").strip().split()[-1]
    except MissionError, IndexError:
        return ""


def installers(
    shell: HostShell,
    source: str = _SOURCE,
    *,
    vendored: bool = True,
    floor: str = "",
    extras: Sequence[str] = (),
) -> Strategy[Installer]:
    """The ordered install routes for `shell`'s host, best first.

    uv leads because its isolated tool environment needs no host interpreter new enough to run
    the tool. Bootstrapping uv beats a user-site pip install, the last route and the only one
    bound to whatever `python3` the host ships.

    The family offered is a fact about the workspace, not the machine. One that vendors the
    tool's source installs from it, so a host never runs a tool older than the manifest it
    compiles. One that consumes the tool from an index installs the version it declares, after
    first offering the machine's own tool when that already satisfies it.

    source: the tool's source directory inside the synced workspace.
    vendored: whether that source is actually on the machine.
    floor: the version the workspace declares for the tool, read from its own manifest.
    extras: the tool's optional extras to install with it, what a center carries for `plot`.
    """
    strategy: Strategy[Installer] = Strategy(f"{_TOOL} installer")
    wanted_extras = f"[{','.join(extras)}]" if extras else ""
    dialect = shell.dialect
    fetch_probe, fetch = dialect.uv_bootstrap
    pip_probe, pip = dialect.pip
    python = shlex.quote(metadata(_TOOL)["Requires-Python"])
    uv = f"uv tool install --force --python {python}"
    if vendored:
        quoted = shlex.quote(f"{source}{wanted_extras}")
        editable = f"{uv} --editable {quoted}"
        strategy.register("uv", Installer(shell, probe=dialect.has("uv"), command=editable))
        strategy.register(
            "uv-bootstrap",
            Installer(shell, probe=fetch_probe, command=dialect.chain(fetch, editable)),
        )
        strategy.register(
            "pip",
            Installer(
                shell, probe=pip_probe, command=f"{pip} --force-reinstall --editable {quoted}"
            ),
        )
        return strategy
    wanted = shlex.quote(f"{_TOOL}{wanted_extras}{specifier(floor)}")
    indexed = f"{uv} {wanted}"
    strategy.register("present", Existing(shell, floor=floor))
    strategy.register("uv-index", Installer(shell, probe=dialect.has("uv"), command=indexed))
    strategy.register(
        "uv-bootstrap-index",
        Installer(shell, probe=fetch_probe, command=dialect.chain(fetch, indexed)),
    )
    strategy.register(
        "pip-index", Installer(shell, probe=pip_probe, command=f"{pip} --upgrade {wanted}")
    )
    return strategy


class Bootstrap:
    """Puts this workspace's tool and environment onto a machine that already answers ssh.

    Install the tool through the first route the machine supports, then have it compile the
    synced manifest and install the environment from the lock this workspace already solved. A
    declared host and a machine rented for one job both drive this one class.

    resolve: let the machine run its own dependency solve instead of installing the shipped lock.
    floor: the version the workspace declares for the tool, used only when the workspace vendors
        no source and the tool therefore comes from an index.
    extras: the tool's optional extras to install with it.
    """

    def __init__(
        self,
        shell: HostShell,
        *,
        resolve: bool = False,
        floor: str = "",
        extras: Sequence[str] = (),
    ) -> None:
        self.shell = shell
        self.resolve = resolve
        self.floor = floor
        self.extras = tuple(extras)

    @property
    def env(self) -> str:
        """The environment being provisioned, the plan's own."""
        return self.shell.plan.env

    def tool(self) -> Resolution[Installer]:
        """Install the tool through the first route the machine supports, keeping the rejections.

        Whether the mirror carries the tool's source is asked of the machine, not assumed: an
        index-consuming workspace ships no such directory, and when every route tested for it,
        all refused and the refusal blamed the host's tooling (miyabi-g, 2026-09-05). So a
        machine no route reaches fails here, before anything assumes the tool, and the refusal
        leads with which family was tried and why, the rejection log behind it.
        """
        vendored = self.shell.ok(self.shell.dialect.is_directory(_SOURCE))
        routes = installers(self.shell, vendored=vendored, floor=self.floor, extras=self.extras)
        try:
            resolution = routes.cascade()
        except StrategyError as refused:
            where = (
                f"installing from the source this workspace vendors at {_SOURCE}"
                if vendored
                else f"installing {_TOOL}{specifier(self.floor) or ' (no version declared)'} "
                "from an index, since this workspace vendors no source"
            )
            raise MissionError(
                f"cannot install {_TOOL} on {self.shell.plan.host!r} by {where}: {refused}"
            ) from None
        routes.select(resolution.winner).install()
        return resolution

    def environment(self) -> None:
        """Have the machine's own tool compile the synced manifest and install `env` from it.

        The machine is told which declared profile describes it, so its activation script carries
        that profile's module stack rather than this machine's.
        """
        host = self.shell.plan.host
        resolving = " --resolve" if self.resolve else ""
        self.shell.run(
            f"{_TOOL} install {shlex.quote(self.env)}{resolving} --profile {shlex.quote(host)}"
        )
        if not self.shell.ok(self.shell.provisioned):
            raise MissionError(
                f"{host!r} has no {self.shell.proof} after installing {self.env!r}; "
                "the environment was not provisioned"
            )


def read_facts(text: str) -> HostFacts:
    """The `HostFacts` in a remote command's output, read from its first `{` so shell chatter
    above is ignored."""
    start = text.find("{")
    if start < 0:
        raise MissionError(
            f"no host facts in the probe output: {text.strip()[-240:] or '(empty)'}"
        )
    return HostFacts.model_validate_json(text[start:])


class Onboarding:
    """Brings one host from bare ssh access to a workspace that runs jobs.

    The steps a person would take by hand, in order: probe the host, mirror the workspace, install
    the tool from that mirror, have it install the manifest's environment, then read the host
    back through the activation it now carries.

    The environment is the plan's own, the host profile's declared choice unless overridden, so
    a host declaring `env = "serving"` gets serving and can never drift from what later commands
    activate.

    The workstation solves, the host installs. `artifact` rides the mirror past the denylist that
    keeps the generated directory local, and the host installs from that lock. Solving on the
    host means building source distributions to read metadata, which puts the host's compiler in
    the lock's dependency path, where one machine's toolchain decides whether another platform's
    requirement can be read at all.

    The workspace goes to the plan's root, a `~` in it placed under the home the probe (or, for
    a sync, the recorded probe) found.

    artifact: the compiled manifest, lock and state shipped with the mirror; empty leaves the
        host to solve.
    resolve: the escape hatch for the rare host that genuinely must solve for itself.
    watch: announces each stage as it begins.
    digest: the manifest digest stamped onto the recorded `HostSetup`, for `doctor`.
    floor: the version this workspace declares for the tool, which a host with no vendored
        source installs from an index; empty when none is declared.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        plan: ExecutionPlan,
        *,
        artifact: Sequence[str] = (),
        resolve: bool = False,
        watch: Watcher | None = None,
        digest: str = "",
        floor: str = "",
    ) -> None:
        self.dispatcher = dispatcher
        self.plan = plan
        self.artifact = tuple(artifact)
        self.resolve = resolve
        self.watch = watch or announce
        self.digest = digest
        self.floor = floor

    @property
    def env(self) -> str:
        """The environment provisioned, the plan's own."""
        return self.plan.env

    def align_pixi(self, shell: HostShell, *, host: str) -> str:
        """Put the fleet's one pixi on `host`, whichever one it runs now, and say which that is.

        Every pixi version writes a lock differently, so a host on another one rewrites the lock
        it was shipped and builds an environment at another content address than the wave pinned:
        on 2026-09-05 a workstation on 0.77 pinned 4950b228a3eaf208, this host on 0.79 built
        4e0f0670076776b1, and every job died with `found no built environment`. So a host is
        brought in line in either direction (the older-is-refused check this replaces let the
        newer one through), with pixi's own installer at `PIXI_VERSION`, writing into
        `$HOME/.pixi/bin`, which every wrapped command puts ahead of the system PATH. A host that
        still disagrees is refused rather than provisioned into a lock nothing can predict.
        """
        theirs = _version(shell, "pixi")
        if theirs == PIXI_VERSION:
            return theirs
        self.watch(f"putting pixi {PIXI_VERSION} on {host}, which runs {theirs or 'none'}")
        shell.run(shell.dialect.pixi_installer)
        aligned = _version(shell, "pixi")
        if aligned != PIXI_VERSION:
            raise MissionError(
                f"{host!r} still runs pixi {aligned or 'none'} after installing {PIXI_VERSION}; "
                f"every lock this workspace ships was solved by {PIXI_VERSION}, and a host on "
                "another one rewrites it while provisioning and builds a different environment "
                "than the one a dispatch pins. Put that version on the host's PATH by hand, then "
                "set it up again."
            )
        return aligned

    def verify_queue(self, shell: HostShell, *, host: str) -> None:
        """Make sure the pueue daemon a plain ssh host dispatches through is answering.

        Every later `submit` fails on its socket otherwise, so a down daemon is started here and
        the host refused, naming the fix, when it still does not answer. A Windows host is set up
        without one, since nothing here can daemonize pueue there yet: it runs commands and
        collects, and a `submit` to it refuses on its own until a pueue answers.
        """
        if not isinstance(pick(self.plan.profile), Pueue) or shell.ok(
            "pueue status", activate=True
        ):
            return
        if is_windows(self.plan.profile):
            logger.warning(
                "%s answers no pueue; `submit` cannot queue there until pueue is installed and "
                "`pueued` started, then the host set up again",
                host,
            )
            return
        shell.run("pueued -d </dev/null >/dev/null 2>&1", activate=True)
        ready = Retrying(
            retry=retry_if_result(lambda answered: not answered),
            stop=stop_after_attempt(10),
            wait=wait_fixed(0.5),
            retry_error_callback=lambda state: False,
        )
        if not ready(shell.ok, "pueue status", activate=True):
            raise MissionError(
                f"pueued is not answering on {host!r}; install pueue there and start it with "
                "`pueued -d`, then set the host up again"
            )

    def undisturbed(self, host: str) -> None:
        """Refuse to change the environment under runs `host` still owes an outcome for.

        Every pinned tree on a host symlinks its environment back to the one prefix in the
        mirror, so shipping a different compiled manifest replaces the environment under runs
        pinned against the old one, and the two waves fight over the prefix: job 3296353 of a
        five-job wave died that way while a newer commit's batch was dispatched around it
        (2026-09-05). A host whose recorded digest matches is only being re-mirrored, which a
        sync between two waves does all day, and a host owing nothing can be told anything.
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

        sync_only: skip the bootstrap and the hardware probe, re-mirroring and re-provisioning
            an already onboarded host instead; see `_sync`.
        """
        host = self.plan.host
        self.undisturbed(host)
        if sync_only:
            return self._sync(host)
        self.watch(f"probing {host}")
        capabilities = probe_capabilities(host)
        self.plan = self.resolved(capabilities)
        root = rooted(self.plan.profile, host=host)
        with open_shell(self.plan, root) as shell:
            bootstrap = Bootstrap(shell, resolve=self.resolve, floor=self.floor)
            self._mirror(host, root)
            self.watch(f"installing {_TOOL} on {host}")
            winner = bootstrap.tool()
            self.watch(f"checking pixi on {host}")
            pixi = self.align_pixi(shell, host=host)
            self.watch(f"provisioning {self.env} on {host}")
            bootstrap.environment()
            self.watch(f"checking the queue on {host}")
            self.verify_queue(shell, host=host)
            self.watch(f"reading {host} back through its activation")
            hardware = read_facts(shell.run(facts_command(), activate=True))
            setup = HostSetup(
                host=host,
                root=root,
                env=self.env,
                activate=shell.activation_record,
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

    def resolved(self, facts: Facts) -> ExecutionPlan:
        """The plan with every gap its profile left open filled from what the probe found."""
        return self.plan.model_copy(update={"profile": resolve(self.plan.profile, facts)})

    def _mirror(self, host: str, root: str) -> None:
        """Mirror the workspace, the compiled artifact riding along, to `host:root`."""
        self.watch(f"mirroring the workspace to {host}:{root}")
        self.dispatcher.mirror(self.plan, root, required=[self.artifact] if self.artifact else [])

    def _sync(self, host: str) -> HostSetup:
        """Re-mirror and re-provision `host`, its bootstrap and hardware probe skipped.

        The fast path back to a host whose manifest moved since setup: neither the tool nor the
        hardware changed, so nothing reinstalls or re-probes. Refuses a host never onboarded.

        THE RECORD IS RE-READ AFTER THE MIRROR, NOT BEFORE. `mirror` stamps `synced_at` on
        its own, mid-block, and building the saved record from a copy taken before that would
        overwrite the very stamp it just wrote.

        THE PIXI IS ALIGNED HERE TOO, since a sync is what a campaign runs between waves, and a
        host whose pixi moved would otherwise rewrite the shipped lock under the queued wave.
        """
        recorded = self.dispatcher.cache.host(host)
        if recorded.capabilities is not None:
            self.plan = self.resolved(recorded.capabilities)
        root = rooted(self.plan.profile, host=host)
        with open_shell(self.plan, root) as shell:
            self._mirror(host, root)
            pixi = self.align_pixi(shell, host=host)
            self.watch(f"provisioning {self.env} on {host}")
            Bootstrap(shell, resolve=self.resolve).environment()
            activate = shell.activation_record
        fresh = self.dispatcher.cache.host(host)
        updated = self.dispatcher.cache.save_host(
            fresh.model_copy(
                update={
                    "root": root,
                    "env": self.env,
                    "activate": activate,
                    "pixi": pixi,
                    "digest": self.digest or fresh.digest,
                }
            )
        )
        logger.info("synced %s at %s", host, root)
        return updated
