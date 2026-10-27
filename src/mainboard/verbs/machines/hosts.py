from json import dumps
from pathlib import Path
from typing import TYPE_CHECKING

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ...center.migration.migrate import Migration
from ...context.resolver import Resolver
from ...core.errors import MissionError
from ...dispatch.mirroring.spool import Spool
from ...engines.compile.pixi.policy import LockPolicy
from ...fleet.board.environments import Onboard
from ...fleet.reliability import reliability, window
from ...holding.holds import Holds
from ...holding.serve import Server
from ...log import logger
from ...probe.readings.occupancy import rows as occupancy_rows
from ...probe.system import System
from ...render import progress
from ...state.storage.holds import holding
from ...upkeep import machine as upkeep
from ..base import Verbs
from ..options import Output, judged, sectioned, titled

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...dispatch.hosts.onboard import HostSetup
    from ...fleet.board import Board
    from ...probe.readings.occupancy import Occupancy
    from ...render.values import Node


@Parameter(name="*")
class Onboarding(FrozenModel):
    """What `host setup` installs on a job host."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    env: str = ""
    """an environment name overriding the host profile's own."""
    resolve: bool = False
    """let the host solve for itself (`install` there) instead of installing the shipped lock
    (`install --locked`)."""
    dotfiles: bool = False
    """also apply the workspace's dotfiles (zsh, lvim, herdr and the toolbox their mb.lock pins);
    every later sync keeps them. Without it a host gets only what jobs need: this tool, pixi,
    pueue and the environment."""

    @property
    def policy(self) -> LockPolicy:
        return LockPolicy.UPDATE if self.resolve else LockPolicy.LOCKED


@Parameter(name="*")
class Relocation(FrozenModel):
    """Whether `host setup` moves the center there instead, and where its workspace goes."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    center: bool = False
    """make this host the workspace's center instead of a job host."""
    root: str = ""
    """with `--center`, where the workspace goes there, `~/projects` when omitted."""


@Parameter(name="*")
class Detail(FrozenModel):
    """The tables `host list` adds, each keyed by host."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    facts: bool = False
    """each host's probed hardware and software, then what they mean for this workspace
    (platform, lock, CUDA floor, card memory), each finding with its fix."""
    gpus: bool = False
    """who holds each card now: utilization, memory and the processes on it."""
    audit: bool = False
    """what each host could and should update, read-only; `host upgrade` applies it. Exits 1
    when an audit row fails."""
    plan: bool = False
    """the execution plan each host resolves to: root, environment, container, scheduler,
    modules, exports."""
    env: str = ""
    """with `--plan`, an environment overriding the host profile's own."""
    reliability: bool = False
    """how often each host or provider ran what was dispatched to it, from the run registry:
    `ran` (the command reached an exit, its own failures included), `unstarted` (a landing or
    setup that never started the job), `lost` (the machine vanished, or no exit code),
    `delivered` = ran / judged. Least reliable first."""
    days: int = 30
    """with `--reliability`, how far back the registry is read."""

    @property
    def asked(self) -> bool:
        """Whether any per-host table was asked for."""
        return self.facts or self.gpus or self.audit or self.plan


# One row of a host's card table.
type _Row = dict[str, str | int | float | bool]

# The verbs' defaults: compact output, a job host installing the shipped lock, the bare listing.
_COMPACT = Output()
_ONBOARDING = Onboarding()
_RELOCATION = Relocation()
_DETAIL = Detail()


def _onboarded(workspace: Board, report: HostSetup, output: Output, *, title: str) -> None:
    """Print a setup record, then the findings its read-back census adds up to.

    The JSON mode keeps the record whole and adds the findings under `findings`, so a script
    reads both from one document.
    """
    census = report.hardware.system if report.hardware else System()
    findings = workspace.findings(census)
    if output.mode == "json":
        payload: dict[str, Node] = {
            **report.model_dump(),
            "findings": [row.model_dump() for row in findings],
        }
        output.print_record(payload, title=title)
        return
    output.print_record(report.model_dump(), title=title)
    judged(findings, mode=output.mode, title=f"findings: {report.host}")


class HostVerbs(Verbs):
    """Hosts: set up, synced, upgraded, served, and listed with what each can do."""

    def _audits(self, names: Sequence[str], output: Output) -> bool:
        """Print each host's update audit, answering whether any row failed."""
        failed = False
        for alias in names:
            with progress(f"auditing {alias}"):
                sections = self.session.board(alias).audit()
            failed |= bool(sectioned(sections, output, title=f"audit: {alias}", heading=True))
        return failed

    def _detailed(self, hosts: Sequence[str], detail: Detail, output: Output) -> int:
        """Print each table `detail` asks for, for `hosts` or here; 1 when an audit fails."""
        named = list(hosts) or ["local"]
        if detail.plan:
            self._plans(named, detail.env, output)
        if detail.facts:
            self._facts(named, output)
        if detail.gpus:
            self._gpus(named if hosts else self.session.ssh_hosts(), output)
        return int(detail.audit and self._audits(named, output))

    def _facts(self, names: Sequence[str], output: Output) -> None:
        """Print each host's probed facts, then the findings judging it against the workspace."""
        for alias in names:
            workspace = self.session.board(alias)
            with progress(f"probing {alias}"):
                found = workspace.facts()
            titled(output, f"facts: {alias}")
            output.print_record(found.model_dump(), title=f"facts: {alias}")
            if output.mode != "json":
                findings = workspace.findings(found.system)
                judged(findings, mode=output.mode, title=f"findings: {alias}")

    def _gpus(self, names: Sequence[str], output: Output) -> None:
        """Print who holds each card on each named host, an unreachable one as a row saying why."""
        read = {name: self._read(name) for name in names}
        if output.json_:
            # The wire form a remote read parses: readings keyed by host.
            readings = {
                name: reading for name, (_, reading) in read.items() if reading is not None
            }
            logger.result(dumps(readings))
            return
        titled(output, "gpus")
        output.print_rows([row for rows, _ in read.values() for row in rows], title="gpus")

    def _occupancy(self, name: str) -> Occupancy:
        """Who holds each card on `name` now."""
        with progress(f"reading the cards of {name}"):
            return self.session.board(name).occupancy()

    def _paths(self, workspace: Board, hosts: Sequence[str], output: Output) -> int:
        """Print every compute path, or those `hosts` name."""
        with progress("probing every compute path"):
            paths = workspace.compute().paths()
        # A provider host's rows are named by its kind (`hpcai` is kind `hpc-ai`), its rentals
        # `<kind>:<handle>`, so naming the host names them too.
        declared = workspace.manifest.hosts
        wanted = {*hosts, *(declared[alias].kind for alias in hosts if alias in declared)}
        chosen = [
            path
            for path in paths
            if not hosts or path.name in wanted or path.name.split(":")[0] in wanted
        ]
        output.print_rows([path.model_dump() for path in chosen], title="compute")
        return 0

    def _plans(self, names: Sequence[str], env: str, output: Output) -> None:
        """Print the execution plan each host resolves to."""
        resolver = Resolver(holding(self.session.project.manifest(self.session.workspace_root())))
        for alias in names:
            titled(output, f"plan: {alias}")
            output.print_record(resolver.plan(alias, env=env).model_dump(), title=f"plan: {alias}")

    def _read(self, name: str) -> tuple[list[_Row], Node | None]:
        """Who holds each card on `name`: table rows, and its wire reading or None unreachable."""
        try:
            occupancy = self._occupancy(name)
        except (MissionError, OSError, ValueError) as error:
            return [
                {
                    "host": name,
                    "card": "",
                    "holders": f"unreachable: {str(error).splitlines()[0][:80]}",
                }
            ], None
        return occupancy_rows(name, occupancy), occupancy.model_dump(mode="json")

    def compute(self, *hosts: str, detail: Detail = _DETAIL, output: Output = _COMPACT) -> int:
        """List every compute path this workspace can reach; flags add detail per host.

        One row per path: this machine, each declared host (answering? set up?), each provider
        (credentials here? credit left? a live rate where cheap). Held machines past their
        deadline are released first. Probes run in parallel, so the fleet answers in the time
        the slowest host takes; a host that is down is a row, not a failure. No credential is
        ever printed. Each flag prints one more table keyed by host, for the hosts named or,
        when none is named, this machine (`--gpus` reads every ssh host).

        Args:
            hosts: narrow the listing to these aliases, `local` for this machine.
        """
        workspace = self.session.board("local")
        for released in Holds(workspace).expire():
            logger.info(f"released {released.alias}, {released.provider} {released.handle}")
        if detail.reliability:
            scored = reliability(workspace.dispatcher.cache.listed(since=window(detail.days)))
            asked = [row for row in scored if not hosts or {row.target, row.kind} & set(hosts)]
            output.print_rows([row.model_dump() for row in asked], title="reliability")
            return 0
        if not detail.asked:
            return self._paths(workspace, hosts, output)
        return self._detailed(hosts, detail, output)

    def host_upgrade(self, on: str = "local", *, dry_run: bool = False) -> int:
        """Bring this machine or a host up to date in one pass, printing each step as it runs.

        The system's managers first (apt: update, full-upgrade, autoremove, autoclean; dnf, brew,
        snap, winget), then the pixi global toolbox, the dotfiles and this tool. Firmware is only
        audited. A host runs its own copy over a terminal, so `sudo` can ask for its password.

        Args:
            on: the host alias, `local` for this machine.
            dry_run: print the steps without running them.
        """
        if on != "local":
            flags = ("--dry-run",) if dry_run else ()
            self.session.board(on).interact(self.session.project.name, "host", "upgrade", *flags)
        return upkeep.upgrade(dry_run=dry_run)

    def serving(self, spool: str, *, gen: int, walltime: str) -> int:
        """Serve a held line's queue inside this PBS allocation, the verb its keeper runs.

        Claims the oldest queued job that fits the time left, runs it through its own runner,
        and ends before the allocation does. It refuses to run anywhere but an allocated node.

        Args:
            spool: the line's spool directory on the shared filesystem.
            gen: the allocation generation the keeper opened this for.
            walltime: how long this allocation lasts, `HH:MM:SS`, from now.
        """
        return Server(Spool(Path(spool)), gen=gen, walltime=walltime).run()

    def setup(
        self,
        host: str,
        *,
        onboarding: Onboarding = _ONBOARDING,
        relocation: Relocation = _RELOCATION,
        output: Output = _COMPACT,
    ) -> int:
        """Onboard a host until it can run jobs, then show what it became and what that means.

        The host installs the environment its declared profile names from the lock this
        workspace solved, shipped with the mirror: the mirror's own environment, which `run
        --on`, `shell --on` and every dispatch's preflight enter. A job activates a pinned copy
        of it instead, addressed by the lock's content and built by the first dispatch of each
        lock (`built <env> on <host>`), from the packages this setup already fetched. Then the
        findings `host list --facts` shows, judged from the census read back through the new
        activation.

        `--center` instead moves the center there (macOS or Linux): probes it, signs gh
        in, carries ssh config and keys, clones the monorepo at this HEAD with every owned
        submodule, carries what git does not hold (`.env`, the lake, agent configuration and
        memory), installs this tool and the default environment, and ends with `doctor --center`
        run there. Secrets ride ssh's stdin only. Running it again continues or re-verifies.
        """
        if relocation.center:
            local = self.session.board("local")
            with progress(f"moving the center to {host}") as stage:
                sections = Migration(local, host, root=relocation.root, watch=stage).run()
            return sectioned(sections, output, title="center")
        workspace = self.session.board(host)
        with progress(f"setting up {host}") as stage:
            report = workspace.setup(
                onboarding.env,
                policy=onboarding.policy,
                watch=stage,
                onboard=Onboard.DOTFILES if onboarding.dotfiles else Onboard.MINIMAL,
            )
        _onboarded(workspace, report, output, title="setup")
        return 0

    def sync(self, host: str, *, env: str = "", output: Output = _COMPACT) -> None:
        """Re-mirror a host already set up and re-provision it from the shipped lock.

        The fast path back after source moved: the tool is not reinstalled and the hardware is
        not probed again, so a Python edit reaches the host in the time the mirror takes. A
        host never set up needs `setup` first, which is where the probe and the tool come from.

        Args:
            env: an environment name overriding the host profile's own.
        """
        workspace = self.session.board(host)
        with progress(f"syncing {host}") as stage:
            report = workspace.setup(
                env, policy=LockPolicy.LOCKED, watch=stage, onboard=Onboard.SYNC
            )
        _onboarded(workspace, report, output, title="sync")
