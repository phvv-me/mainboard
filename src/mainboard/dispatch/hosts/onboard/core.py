# Onboarding a host: put uv's CPython there, mirror the workspace, put the tool on the machine,
# provision the manifest's environment there, and read the host back through the activation
# that install just wrote. It runs over the transports dispatch already owns rather than a
# second, parallel way to reach a host, and succeeds the shell script the previous generation
# shipped.

import shlex
import shutil
from typing import TYPE_CHECKING

from plumbum import local
from tenacity import (
    Retrying,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
)

from ....constants import BYTES_PER_KIBIBYTE
from ....core.errors import MissionError
from ....core.project import Project
from ....engines.compile import backend
from ....engines.compile.backend.process import Process
from ....engines.compile.pixi.policy import LockPolicy
from ....log import logger
from ...schedulers.pueue import Pueue
from ...schedulers.registry import pick
from ...shared import Watcher, announce
from ...transport.shells import HostShell, open_shell
from ...transport.targets import Facts, probe_capabilities, resolve, rooted
from .bootstrap import Bootstrap, reported
from .constants import TOOL
from .record import HostSetup, facts_command, read_facts

_CHEZMOI = 'pixi exec chezmoi --force --source "$HOME/.local/share/chezmoi"'

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ....context.plan import ExecutionPlan
    from ....probe.readings.snapshot import HostFacts
    from ...dispatcher import Dispatcher


class Onboarding:
    """Brings one host from bare ssh access to a workspace that runs jobs.

    The steps a person would take by hand, in order: probe the host, put uv's CPython there,
    mirror the workspace, install the tool from that mirror, have it install the manifest's
    environment, then read the host back through the activation it now carries.

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
    policy: what the host's install may do to the shipped lock, `locked` unless a host
        genuinely must solve for itself (`update`).
    watch: announces each stage as it begins.
    digest: the manifest digest stamped onto the recorded `HostSetup`, for `doctor`.
    floor: the version this workspace declares for the tool, which a host with no vendored
        source installs from an index; empty when none is declared.
    dotfiles: the owner's chezmoi repository, applied on the host; empty applies none.
    """

    @property
    def env(self) -> str:
        """The environment provisioned, the plan's own."""
        return self.plan.env

    @staticmethod
    def _herded(host: str) -> None:
        """Save `host` in this machine's herdr, so its sessions sit beside the local ones.

        Quiet when herdr is not here or already lists the host. herdr asks before installing
        anything on the far side, and with its input closed the answer is no, so a host needing
        that is left to a `herdr machine add` in a terminal, which is said.
        """
        herdr = shutil.which("herdr")
        if herdr is None:
            return
        listed = Process.capture(local[herdr]["machine", "list", "--json"], timeout=60)
        if f'"{host}"' in listed.stdout:
            return
        added = Process.capture(local[herdr]["machine", "add", host, "--label", host], timeout=300)
        if not added.succeeded:
            logger.warning(
                "herdr does not list {} yet; run `herdr machine add {}` once", host, host
            )

    @staticmethod
    def _measured(hardware: HostFacts | None, shell: HostShell) -> HostFacts | None:
        """`hardware` with the free space at its workspace root read again (`df -Pk`).

        That is the figure a sync's disk row reports, which the setup's census had frozen at
        67 GB on a disk down to 30.
        """
        if hardware is None or not hardware.system.root:
            return hardware
        said = shell.run(f"df -Pk {shlex.quote(hardware.system.root)}").split()
        system = hardware.system.model_copy(
            update={"free_bytes": int(said[-3]) * BYTES_PER_KIBIBYTE}
        )
        return hardware.model_copy(update={"system": system})

    def __init__(
        self,
        dispatcher: Dispatcher,
        plan: ExecutionPlan,
        *,
        artifact: Sequence[str] = (),
        policy: LockPolicy = LockPolicy.LOCKED,
        watch: Watcher | None = None,
        digest: str = "",
        floor: str = "",
        dotfiles: str = "",
    ) -> None:
        self.dispatcher = dispatcher
        self.plan = plan
        self.artifact = tuple(artifact)
        self.policy = policy
        self.watch = watch or announce
        self.digest = digest
        self.floor = floor
        self.dotfiles = dotfiles

    def _installing(self, host: str) -> str:
        """The stage that installs the environment, named for what it installs.

        This is the mirror's own environment, the one `run --on`, `shell --on` and every
        dispatch's preflight enter, brought in line with the shipped lock: seconds when it
        already is. It was announced as `provisioning`, which read as the job's environment
        being built and then finished in four seconds (crimson, 2026-09-30).
        """
        return f"installing {self.env} in {host}'s mirror from the shipped lock"

    def _mirror(self, host: str, *, root: str) -> None:
        """Mirror the workspace, the compiled artifact riding along, to `host:root`."""
        self.watch(f"mirroring the workspace to {host}:{root}")
        self.dispatcher.mirror(self.plan, root, required=[self.artifact] if self.artifact else [])

    def _onboard(self, host: str) -> HostSetup:
        """Probe `host`, set it up from bare ssh and read it back through its new activation."""
        self.watch(f"probing {host}")
        capabilities = probe_capabilities(host)
        self.plan = self.resolved(capabilities)
        root = rooted(self.plan.profile, host=host)
        with open_shell(self.plan, root) as shell:
            bootstrap = self._prepared(shell, host=host, root=root)
            self.watch(f"installing {TOOL} on {host}")
            winner = bootstrap.tool()
            pixi = self._provisioned(shell, bootstrap, host=host)
            self.watch(f"reading {host} back through its activation")
            hardware = read_facts(shell.run(facts_command(), activate=True))
            if not self.plan.containerized:
                self.watch(self._pinned_later(host))
            setup = HostSetup(
                host=host,
                root=root,
                env=self.env,
                activate=shell.proof,
                installer=winner.winner,
                rejected=winner.rejected,
                tool=shell.run(f"{TOOL} --version").strip(),
                pixi=pixi,
                capabilities=capabilities,
                hardware=hardware,
                digest=self.digest,
                dotfiles=bool(self.dotfiles),
                requires=bootstrap.requirements,
            )
        recorded = self.dispatcher.cache.hosts.save(setup)
        logger.info("onboarded {} at {} through {}", host, root, recorded.installer)
        return recorded

    def _pinned_later(self, host: str) -> str:
        """What a setup leaves for the first dispatch, said so its minutes are expected there."""
        return (
            f"{host} runs commands in {self.env}; a job activates its own pinned copy, which "
            f"the first dispatch of each lock builds there (`built {self.env} on {host}`)"
        )

    def _prepared(self, shell: HostShell, *, host: str, root: str) -> Bootstrap:
        """The bootstrap of `host`, uv's CPython put there and the workspace mirrored to it."""
        bootstrap = Bootstrap(
            shell, workspace=self.dispatcher.root, policy=self.policy, floor=self.floor
        )
        self.watch(f"putting uv's CPython on {host}")
        bootstrap.python()
        self._mirror(host, root=root)
        return bootstrap

    def _provisioned(self, shell: HostShell, bootstrap: Bootstrap, *, host: str) -> str:
        """Bring `host`'s pixi, environment, lake, queue and dotfiles in line, answering its pixi.

        A sync does all of it too: a daemon that died or a host cleaned since setup would refuse
        every submit.
        """
        self.watch(f"checking pixi on {host}")
        pixi = self.align_pixi(shell, host=host)
        self.watch(self._installing(host))
        bootstrap.environment()
        self.watch(f"checking {host}'s lake against its DuckDB")
        bootstrap.lake()
        self.watch(f"checking the queue on {host}")
        self.verify_queue(shell, host=host)
        self.apply_dotfiles(shell, host=host)
        return pixi

    def _sync(self, host: str) -> HostSetup:
        """Re-mirror and re-provision `host`, skipping its bootstrap and hardware probe.

        Only the free disk is measured again, since a sync is what fills it.

        The fast path back to a host whose manifest moved since setup: the hardware is not probed
        again, uv's CPython is only confirmed, and the tool is reinstalled only when its declared
        requirements moved since the host last installed it (`HostSetup.requires`). Refuses a
        host never onboarded.

        THE RECORD IS RE-READ AFTER THE MIRROR, NOT BEFORE. `mirror` stamps `synced_at` on
        its own, mid-block, and building the saved record from a copy taken before that would
        overwrite the very stamp it just wrote.

        THE PIXI IS ALIGNED HERE TOO, since a sync is what a campaign runs between waves, and a
        host whose pixi moved would otherwise rewrite the shipped lock under the queued wave.
        """
        recorded = self.dispatcher.cache.hosts.recorded(host)
        if recorded.capabilities is not None:
            self.plan = self.resolved(recorded.capabilities)
        root = rooted(self.plan.profile, host=host)
        with open_shell(self.plan, root) as shell:
            bootstrap = self._prepared(shell, host=host, root=root)
            tool: dict[str, str | tuple[tuple[str, str], ...]] = {}
            if recorded.requires != (requirements := bootstrap.requirements):
                self.watch(f"reinstalling {TOOL} on {host}, its requirements moved")
                winner = bootstrap.tool()
                tool = {
                    "installer": winner.winner,
                    "rejected": winner.rejected,
                    "tool": shell.run(f"{TOOL} --version").strip(),
                    "requires": requirements,
                }
            pixi = self._provisioned(shell, bootstrap, host=host)
            activate = shell.proof
            hardware = self._measured(recorded.hardware, shell)
        fresh = self.dispatcher.cache.hosts.recorded(host)
        updated = self.dispatcher.cache.hosts.save(
            fresh.model_copy(
                update={
                    "root": root,
                    "env": self.env,
                    "activate": activate,
                    "pixi": pixi,
                    "digest": self.digest or fresh.digest,
                    "hardware": hardware,
                    **tool,
                }
            )
        )
        logger.info("synced {} at {}", host, root)
        return updated

    def align_pixi(self, shell: HostShell, *, host: str) -> str:
        """Put the fleet's one pixi on `host`, whichever one it runs now, and say which that is.

        Every pixi version writes a lock differently, so a host on another one rewrites the lock
        it was shipped and builds an environment at another content address than the wave pinned:
        on 2026-09-05 a workstation on 0.77 pinned 4950b228a3eaf208, this host on 0.79 built
        4e0f0670076776b1, and every job died with `found no built environment`. So a host is
        brought in line in either direction (the older-is-refused check this replaces let the
        newer one through), with pixi's own installer at `PixiEngine.VERSION`, writing into
        `$HOME/.pixi/bin`, which every wrapped command puts ahead of the system PATH. A host that
        still disagrees is refused rather than provisioned into a lock nothing can predict.
        """
        theirs = reported(shell, "pixi")
        if theirs == backend.PixiEngine.VERSION:
            return theirs
        self.watch(
            f"putting pixi {backend.PixiEngine.VERSION} on {host}, which runs {theirs or 'none'}"
        )
        shell.run(shell.dialect.pixi_installer)
        aligned = reported(shell, "pixi")
        if aligned != backend.PixiEngine.VERSION:
            raise MissionError(
                f"{host!r} still runs pixi {aligned or 'none'} after installing "
                f"{backend.PixiEngine.VERSION}; every lock this workspace ships was solved by "
                f"{backend.PixiEngine.VERSION}, and a host on another one rewrites it while "
                "provisioning and builds a different environment than the one a dispatch pins. "
                "Put that version on the host's PATH by hand, then set it up again."
            )
        return aligned

    def apply_dotfiles(self, shell: HostShell, *, host: str) -> None:
        """Apply the owner's dotfiles on `host`: the same shell, commands and editor everywhere.

        chezmoi runs through the pixi `align_pixi` put there, cloning on the first setup and
        pulling on every later one. A host keeping its own shell still runs jobs, so a failure is
        said and the setup goes on.
        """
        if not self.dotfiles:
            return
        self.watch(f"applying the dotfiles on {host}")
        # The source is named, since a host's own chezmoi config may point anywhere.
        repository = shlex.quote(self.dotfiles)
        chezmoi = _CHEZMOI
        command = (
            f'if [ -d "$HOME/.local/share/chezmoi/.git" ]; then {chezmoi} update; '
            f"else {chezmoi} init --apply {repository}; fi"
        )
        try:
            shell.run(command)
        except MissionError as fault:
            logger.warning("{} keeps its own shell setup: {}", host, fault)
            return
        Onboarding._herded(host)

    def resolved(self, facts: Facts) -> ExecutionPlan:
        """The plan with every gap its profile left open filled from what the probe found."""
        return self.plan.model_copy(update={"profile": resolve(self.plan.profile, facts)})

    def run(self, *, sync_only: bool = False) -> HostSetup:
        """Onboard the host and return (and record) what it became.

        sync_only: skip the bootstrap and the hardware probe, re-mirroring and re-provisioning
            an already onboarded host instead; see `_sync`.
        """
        host = self.plan.host
        self.undisturbed(host)
        return self._sync(host) if sync_only else self._onboard(host)

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
            recorded = self.dispatcher.cache.hosts.recorded(host)
        except LookupError:
            return
        if not recorded.digest or recorded.digest == self.digest:
            return
        owed = [run for run in self.dispatcher.cache.listed("live") if run.target == host]
        if not owed:
            return
        named = ", ".join(run.handle for run in owed[:4])
        more = f" and {len(owed) - 4} more" if len(owed) > 4 else ""
        raise MissionError(
            f"{host!r} still owes {len(owed)} run(s) an outcome ({named}{more}) and this "
            f"workspace's environment has changed since {host!r} was set up. Every one of those "
            "runs activates the environment this would replace, so shipping it now would change "
            "what they run in while they wait. Let them settle "
            f"(`{Project().name} job list`), or `{Project().name} job cancel <handle>` the ones "
            "you no longer need, then set the host up again."
        )

    def verify_queue(self, shell: HostShell, *, host: str) -> None:
        """Make sure the pueue daemon a plain ssh host dispatches through is answering.

        Every later `submit` fails on its socket otherwise, so a down daemon is started here and
        the host refused, naming the fix, when it still does not answer.
        """
        if not isinstance(pick(self.plan.profile), Pueue) or shell.ok(
            "pueue status", activate=True
        ):
            return
        if not shell.ok("command -v pueued", activate=True):
            # A host-level tool, never an environment's (a lean one has none): a host without
            # the dotfiles' toolbox gets it from pixi, which this setup just aligned.
            self.watch(f"installing pueue on {host}")
            shell.run("pixi global install pueue")
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
