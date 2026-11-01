# Moving the center: the one machine that holds the monorepo, runs this tool and runs the agents.
#
# The destination is any macOS or Linux machine ssh reaches, with nothing of ours on it. The move
# is a sequence of converging steps, each of which looks at what the destination already holds and
# does only the difference, so running it again after an interruption carries on where it stopped
# and running it once more after it finished re-verifies everything and changes nothing. Nothing
# the center knows is typed twice: the tree comes from the `git` module's ownership rules, the
# environment from the lock this center solved, the machine findings from the same census and judge
# `facts` uses, and the final word from the destination running `doctor --center` on itself.

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from patos import FrozenModel
from pydantic import TypeAdapter

from ...core.errors import MissionError
from ...core.project import Project
from ...core.section import Section, staged
from ...dispatch.hosts.onboard import Bootstrap, Onboarding
from ...dispatch.shared import announce
from ...dispatch.transport.core import HostUnreachable, SshTransport
from ...dispatch.transport.shells import Dialect, open_shell
from ...dispatch.transport.targets import probe_capabilities
from ...engines.compile.pixi.lockfile import Lockfile
from ...enums import Role, Verdict
from ...fleet.fitness import Fitness
from ...probe.system import System
from .carrier import Carrier
from .state import Carried, Destination, Parcel, packed
from .steps.cloning import Cloning
from .steps.credentials import Credentials

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...context.plan import ExecutionPlan
    from ...dispatch.shared import Watcher
    from ...dispatch.transport.targets import Facts
    from ...fleet.board import Board

# What the owner types, and what the destination runs: the script every release installs, since
# the destination may still carry an older one.
_NAME = Project().name
_TOOL = Project().package

# The findings about the destination that make every later step pointless: a platform the
# workspace never declared, and a lock with nothing for it, both of which the install would only
# rediscover after the clone and the copy had spent their time.
_BLOCKING = frozenset({"platform", "lock"})

# The optional extras of this tool worth carrying, each with the distribution that shows this
# center installed it: the destination gets the same tool, plotting included.
_EXTRAS = {"plot": "matplotlib"}

# Where the workspace goes on the destination unless told otherwise: the center, unlike a dispatch
# target, holds the human checkout, and the destination expands the `~` in its own spelling.
_CHECKOUT = "~/projects"

# How long the destination's own `doctor --center` may run, a torch and CUDA smoke and every lint
# tool's probe included, and the silence its ssh rides out while that smoke loads the machine:
# the default 45 s dropped it with `Timeout, server not responding` (pedro-home, 2026-09-26).
_VERIFY_SECONDS = 1800.0
_PATIENT = {"server_alive_interval": 30.0, "server_alive_count": 10}

# A readiness report as `doctor --center --json` prints it.
_SECTIONS = TypeAdapter(list[Section])


class Written(FrozenModel):
    """How many shipped files changed bytes on the destination."""

    written: int


class _Landing(NamedTuple):
    """The destination once reached: its agent, facts, workspace place and fitness."""

    carrier: Carrier
    facts: Facts
    place: Destination
    findings: list[Section]


class Migration:
    """Move the center to `destination`, answering one report of what works and what does not.

    board: this center's workspace.
    destination: the ssh alias of the machine becoming the center.
    root: where the workspace goes there, `~/projects` when empty.
    watch: announces each step as it begins.
    """

    @staticmethod
    def _extras() -> list[str]:
        """The extras this center's own tool was installed with."""
        return [extra for extra, package in _EXTRAS.items() if Migration._installed(package)]

    @staticmethod
    def _installed(package: str) -> bool:
        """Whether the distribution `package` is installed beside this tool."""
        with suppress(PackageNotFoundError):
            distribution(package)
            return True
        return False

    def __init__(
        self,
        board: Board,
        destination: str,
        *,
        root: str = "",
        watch: Watcher | None = None,
    ) -> None:
        self.board = board
        self.destination = destination
        self.root = root
        self.transport = SshTransport()
        self.watch = watch or announce
        self.home = Path.home()
        self.cloning = Cloning(board, destination)
        self.credentials = Credentials(destination)

    def _install(self, plan: ExecutionPlan, root: str) -> list[Section]:
        """The tool from the cloned source, the fleet's pixi, and the environment from the lock."""
        rows: list[Section] = []
        with open_shell(plan, root, ssh=self.transport) as shell:
            bootstrap = Bootstrap(shell, workspace=self.board.root, extras=self._extras())
            steps = [
                ("install tool", lambda: bootstrap.tool().winner),
                (
                    "install pixi",
                    lambda: Onboarding(self.board.dispatcher, plan).align_pixi(
                        shell, host=self.destination
                    ),
                ),
                ("install default", lambda: shell.run(f"{_TOOL} install").strip()[-160:]),
            ]
            for name, step in steps:
                self.watch(f"{name} on {self.destination}")
                try:
                    done = step() or "done"
                except MissionError as refusal:
                    rows.append(
                        Section(
                            section=name,
                            verdict=Verdict.FAIL,
                            detail=str(refusal).splitlines()[0][:240],
                            fix=f"{_NAME} host setup {self.destination} --center",
                        )
                    )
                    break
                rows.append(Section(section=name, verdict=Verdict.PASS, detail=done))
        return rows

    def _landed(self) -> _Landing:
        """The destination reached: its agent, where the workspace goes, and how fit it is."""
        facts = self._reach()
        carrier = Carrier(
            self.destination,
            uv=facts.uv,
            transport=self.transport,
            python=Lockfile(self.board.root).python_request(facts.pixi_platform),
        )
        where = carrier.call("where", {"root": self.root or _CHECKOUT}, Destination)
        system = carrier.call("census", {"root": where.root}, System)
        findings = Fitness(self.board.root, self.board.manifest).judge(
            system, host=self.destination, role=Role.CENTER
        )
        place = where.model_copy(update={"system": system.system})
        return _Landing(carrier=carrier, facts=facts, place=place, findings=findings)

    def _moved(self, landing: _Landing) -> list[Section]:
        """Sign in, carry the keys, clone, carry the rest, then install and verify there."""
        carrier, place = landing.carrier, landing.place
        carried = Carried(self.board.root, self.board.manifest, place, home=self.home)
        self.watch(f"signing {self.destination} in to GitHub")
        report = [
            self.credentials.sign_in(carrier),
            self._ship(carrier, place, "ssh", carried.ssh()),
            self.credentials.trust(carrier, carried, self.cloning.urls()),
        ]
        self.watch(f"cloning the workspace to {self.destination}:{place.root}")
        report += (cloned := self.cloning.run(carrier, place))
        if any(row.verdict is Verdict.FAIL for row in cloned):
            return report
        report += [
            self._ship(carrier, place, "workspace", carried.workspace()),
            self._ship(carrier, place, "agents", carried.agents()),
            self.credentials.claude(carrier, carried),
        ]
        plan = self._plan(landing.facts, place.root)
        return [
            *report,
            *self._install(plan, place.root),
            *self._verify(plan, place.root),
            *carried.left(),
        ]

    def _plan(self, facts: Facts, root: str) -> ExecutionPlan:
        """The destination's execution plan, its platform and root filled from the probe."""
        plan = self.board.resolver.plan(self.destination, container="none")
        profile = plan.profile.model_copy(update={"platform": facts.pixi_platform, "root": root})
        return plan.model_copy(update={"profile": profile})

    def _preflight(self) -> list[Section]:
        """Whether every owned repository's HEAD can be fetched, and what stays behind unsaved."""
        tree = self.board.git()
        unserved = {
            repo.name for repo in tree.owned() if not repo.remote.serves(repo.commits.head())
        }
        rows: list[Section] = []
        for state in tree.status():
            if state.repo in unserved:
                rows.append(
                    Section(
                        section=f"publish {state.repo}",
                        verdict=Verdict.FAIL,
                        detail=f"HEAD {state.head} is on no remote branch, so no clone fetches it",
                        fix=f"{_NAME} git push",
                    )
                )
            if state.changed or state.untracked:
                rows.append(
                    Section(
                        section=f"unsaved {state.repo}",
                        verdict=Verdict.WARN,
                        detail=(
                            f"{state.changed} changed and {state.untracked} untracked paths stay "
                            "on this machine"
                        ),
                        fix=f'{_NAME} git commit -m "..." && {_NAME} git push',
                    )
                )
        return rows or [
            Section(
                section="publish", verdict=Verdict.PASS, detail="every owned HEAD is on a remote"
            )
        ]

    def _reach(self) -> Facts:
        """The destination as the stock-tools probe finds it, with uv put there when missing."""
        facts = probe_capabilities(self.destination, ssh=self.transport)
        if facts.uv:
            return facts
        self.watch(f"putting uv on {self.destination}")
        with open_shell(self._plan(facts, facts.home), facts.home, ssh=self.transport) as shell:
            shell.run(shell.dialect.uv_bootstrap[1])
        facts = probe_capabilities(self.destination, ssh=self.transport)
        if not facts.uv:
            raise MissionError(f"{self.destination!r} still has no uv after its installer ran")
        return facts

    def _ship(
        self, carrier: Carrier, place: Destination, label: str, parcels: Sequence[Parcel]
    ) -> Section:
        """Send the parcels the destination does not already hold, answering one row.

        label: what the parcels are, the row's name.
        """
        self.watch(f"carrying {label} to {self.destination}")
        listing = [parcel.listing() for parcel in parcels]
        missing = set(
            carrier.call("inventory", {"root": place.root, "parcels": listing}, list[str])
        )
        sending = [parcel for parcel in parcels if parcel.key in missing]
        written = 0
        if sending:
            secret = [parcel.key for parcel in sending if parcel.secret]
            written = carrier.call(
                "place", {"root": place.root, "secret": secret}, Written, stream=packed(sending)
            ).written
        kept = len(parcels) - len(sending)
        return Section(
            section=f"carry {label}",
            verdict=Verdict.PASS,
            detail=f"{len(parcels)} files, {written} written, {kept} already there",
        )

    def _verify(self, plan: ExecutionPlan, root: str) -> list[Section]:
        """The destination's own `doctor --center`, which is the readiness suite run there.

        It rides one ssh process whose keepalives are patient enough for the smoke that loads
        the machine, bounded by a wall clock past any full verify.
        """
        self.watch(f"verifying {self.destination}")
        dialect = Dialect()
        ssh = self.transport.model_copy(update=_PATIENT)
        line = dialect.stage(plan, root, command=f"{_TOOL} doctor --center --json", activate=False)
        argv = dialect.one_shot(ssh, self.destination, line=line)
        try:
            _, out, err = ssh.invoke(
                argv, self.destination, operation="verify", timeout=_VERIFY_SECONDS
            )
        except HostUnreachable as dropped:
            out, err = "", str(dropped)
        start = out.find("[")
        try:
            sections = _SECTIONS.validate_json(out[start:] if start >= 0 else "")
        except ValueError:
            said = (err or out).strip()[-240:]
            return [
                Section(
                    section="verify",
                    verdict=Verdict.FAIL,
                    detail=f"no readiness report came back: {said}",
                    fix=f"{_NAME} doctor --center (on {self.destination})",
                )
            ]
        return [staged("verify", row) for row in sections]

    def run(self) -> list[Section]:
        """Every step in order, stopping only where continuing could not change the answer."""
        report = self._preflight()
        if any(row.verdict is Verdict.FAIL for row in report):
            return report
        self.watch(f"probing {self.destination}")
        landing = self._landed()
        report += [staged("destination", row) for row in landing.findings]
        if any(
            row.section in _BLOCKING and row.verdict is Verdict.FAIL for row in landing.findings
        ):
            return report
        return report + self._moved(landing)
