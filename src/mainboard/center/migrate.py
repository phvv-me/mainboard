# Moving the center: the one machine that holds the monorepo, runs this tool and runs the agents.
#
# The destination is any machine ssh reaches, Windows, macOS or Linux, with nothing of ours on
# it. The move is a sequence of converging steps, each of which looks at what the destination
# already holds and does only the difference, so running it again after an interruption carries
# on where it stopped and running it once more after it finished re-verifies everything and
# changes nothing. Nothing the center knows is typed twice: the tree comes from the `git`
# module's ownership rules, the environment from the lock this center solved, the machine
# findings from the same census and judge `facts` uses, and the final word from the destination
# running `center verify` on itself.

import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import TypeAdapter

from ..core.errors import MissionError
from ..core.project import Project
from ..core.section import Section, Verdict, staged
from ..dispatch.onboard import Bootstrap, Onboarding
from ..dispatch.shared import announce
from ..dispatch.shells import open_shell
from ..dispatch.targets import probe_capabilities
from ..dispatch.transport import SshTransport
from ..fitness import Fitness, Role
from ..probe.system import System
from .carrier import Carrier
from .state import Carried, Destination, Parcel, packed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..board import Board
    from ..context.plan import ExecutionPlan
    from ..dispatch.shared import Watcher
    from ..dispatch.targets import Facts

_TOOL = Project().name

# The findings about the destination that make every later step pointless: a platform the
# workspace never declared, and a lock with nothing for it, both of which the install would only
# rediscover after the clone and the copy had spent their time.
_BLOCKING = frozenset({"platform", "lock"})

# The optional extras of this tool worth carrying, each with the distribution that shows this
# center installed it: the destination gets the same tool, plotting included.
_EXTRAS = {"plot": "seaborn", "wandb": "wandb"}

# A readiness report as `center verify --json` prints it.
_SECTIONS = TypeAdapter(list[Section])


class Signed(FrozenModel):
    """What the destination's gh said to the login it was handed."""

    signed: bool
    detail: str = ""


class Written(FrozenModel):
    """How many shipped files changed bytes on the destination."""

    written: int


class Merged(FrozenModel):
    """How many entries a JSON merge changed."""

    changed: int


class Cloned(FrozenModel):
    """One repository's line from the destination's clone."""

    repo: str
    outcome: str
    detail: str = ""


class Migration:
    """Move the center to `destination`, answering one report of what works and what does not.

    board: this center's workspace.
    destination: the ssh alias of the machine becoming the center.
    root: where the workspace goes there, the probe's own guess (`~/projects`) when empty.
    transport: the ssh policy every call rides.
    watch: announces each step as it begins.
    home: this machine's home directory, whose agent state and ssh keys are carried.
    token: answers this machine's GitHub token, empty when it has none.
    """

    def __init__(
        self,
        board: Board,
        destination: str,
        *,
        root: str = "",
        transport: SshTransport | None = None,
        watch: Watcher | None = None,
        home: Path | None = None,
        token: Callable[[], str] | None = None,
    ) -> None:
        self.board = board
        self.destination = destination
        self.root = root
        self.transport = transport or SshTransport()
        self.watch = watch or announce
        self.home = home or Path.home()
        self.token = token or github_token

    def run(self) -> list[Section]:
        """Every step in order, stopping only where continuing could not change the answer."""
        report = self.preflight()
        if any(row.verdict is Verdict.FAIL for row in report):
            return report
        self.watch(f"probing {self.destination}")
        facts = self.reach()
        carrier = Carrier(self.destination, facts.uv, self.transport)
        where = carrier.call("where", {"root": self.root or facts.root}, Destination)
        system = carrier.call("census", {"root": where.root}, System)
        place = where.model_copy(update={"system": system.system})
        findings = Fitness(self.board.root, self.board.manifest).judge(
            system, host=self.destination, role=Role.CENTER
        )
        report += [staged("destination", row) for row in findings]
        if any(row.section in _BLOCKING and row.verdict is Verdict.FAIL for row in findings):
            return report
        carried = Carried(self.board.root, self.board.manifest, place, home=self.home)
        self.watch(f"signing {self.destination} in to GitHub")
        report.append(self.sign_in(carrier))
        report.append(self.ship(carrier, place, "ssh", carried.ssh()))
        self.watch(f"cloning the workspace to {self.destination}:{place.root}")
        cloned = self.clone(carrier, place)
        report += cloned
        if any(row.verdict is Verdict.FAIL for row in cloned):
            return report
        report.append(self.ship(carrier, place, "workspace", carried.workspace()))
        report.append(self.ship(carrier, place, "agents", carried.agents()))
        report.append(self.claude(carrier, carried))
        plan = self.plan(facts, place.root)
        report += self.install(plan, place.root)
        report += self.verify(plan, place.root)
        return report + carried.left()

    def preflight(self) -> list[Section]:
        """Whether every owned repository's HEAD can be fetched, and what stays behind unsaved."""
        rows: list[Section] = []
        for state in self.board.git().status():
            if not state.published:
                rows.append(
                    Section(
                        section=f"publish {state.repo}",
                        verdict=Verdict.FAIL,
                        detail=f"HEAD {state.head} is on no remote branch, so no clone fetches it",
                        fix=f"{_TOOL} center git push",
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
                        fix=f'{_TOOL} center git commit -m "..." && {_TOOL} center git push',
                    )
                )
        return rows or [
            Section(
                section="publish", verdict=Verdict.PASS, detail="every owned HEAD is on a remote"
            )
        ]

    def reach(self) -> Facts:
        """The destination as the stock-tools probe finds it, with uv put there when missing."""
        facts = probe_capabilities(self.destination, ssh=self.transport)
        if facts.uv:
            return facts
        self.watch(f"putting uv on {self.destination}")
        home = str(PurePosixPath(facts.root).parent)
        with open_shell(self.plan(facts, home), home, ssh=self.transport) as shell:
            shell.run(shell.dialect.uv_bootstrap[1])
        facts = probe_capabilities(self.destination, ssh=self.transport)
        if not facts.uv:
            raise MissionError(f"{self.destination!r} still has no uv after its installer ran")
        return facts

    def sign_in(self, carrier: Carrier) -> Section:
        """Hand this machine's GitHub login to gh there, which git then uses for https."""
        token = self.token()
        if not token:
            return Section(
                section="github",
                verdict=Verdict.WARN,
                detail="no gh login here to carry",
                fix=f"gh auth login && gh auth setup-git (on {self.destination})",
            )
        answer = carrier.call("login", {"token": token}, Signed)
        if answer.signed:
            return Section(
                section="github", verdict=Verdict.PASS, detail="gh signed in, git uses it"
            )
        return Section(
            section="github",
            verdict=Verdict.FAIL,
            detail=f"gh did not take the login: {answer.detail}",
            fix="install gh there (winget install --id GitHub.cli -e), then migrate again",
        )

    def ship(
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

    def clone(self, carrier: Carrier, place: Destination) -> list[Section]:
        """The root at this HEAD and each owned submodule at its pointer, then the foreign rest."""
        tree = self.board.git()
        owned = tree.owned()
        submodules = [
            [parent.name, parent.relative(child)]
            for parent in owned
            for child in parent.children
            if child.initialized and child.owned
        ]
        foreign = [
            child.name
            for parent in owned
            for child in parent.children
            if child.initialized and not child.owned
        ]
        steps = carrier.call(
            "clone",
            {
                "root": place.root,
                "url": tree.root.url,
                "branch": tree.root.branch(),
                "commit": tree.root.head(),
                "submodules": submodules,
            },
            list[Cloned],
        )
        again = f"{_TOOL} center migrate {self.destination} --root <an empty directory>"
        rows = [
            Section(
                section=f"clone {step.repo}",
                verdict=Verdict.PASS if step.outcome == "done" else Verdict.FAIL,
                detail=step.detail,
                fix="" if step.outcome == "done" else again,
            )
            for step in steps
        ]
        if foreign:
            rows.append(
                Section(
                    section="clone references",
                    verdict=Verdict.PASS,
                    detail=f"{len(foreign)} foreign submodules left to fetch on demand",
                    fix=f"git submodule update --init -- {foreign[0]}",
                )
            )
        return rows

    def claude(self, carrier: Carrier, carried: Carried) -> Section:
        """This workspace's Claude Code project settings, merged in under the new path."""
        entries = carried.claude_project()
        if not entries:
            return Section(
                section="carry claude project", verdict=Verdict.PASS, detail="nothing to carry"
            )
        answer = carrier.call(
            "merge", {"path": ".claude.json", "key": "projects", "entries": entries}, Merged
        )
        return Section(
            section="carry claude project",
            verdict=Verdict.PASS,
            detail=f"trust and tool settings under the new path, {answer.changed} changed",
        )

    def install(self, plan: ExecutionPlan, root: str) -> list[Section]:
        """The tool from the cloned source, the fleet's pixi, and the environment from the lock."""
        rows: list[Section] = []
        with open_shell(plan, root, ssh=self.transport) as shell:
            steps = (
                ("install tool", lambda: Bootstrap(shell, extras=self.extras()).tool().winner),
                (
                    "install pixi",
                    lambda: Onboarding(self.board.dispatcher, plan).align_pixi(
                        shell, host=self.destination
                    ),
                ),
                ("install default", lambda: shell.run(f"{_TOOL} install").strip()[-160:]),
            )
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
                            fix=f"{_TOOL} center migrate {self.destination}",
                        )
                    )
                    break
                rows.append(Section(section=name, verdict=Verdict.PASS, detail=done))
        return rows

    def verify(self, plan: ExecutionPlan, root: str) -> list[Section]:
        """The destination's own `center verify`, which is the readiness suite run there."""
        self.watch(f"verifying {self.destination}")
        with open_shell(plan, root, ssh=self.transport) as shell:
            line = shell.stage(f"{_TOOL} center verify --json", activate=False)
            _, out, err = shell.execute(line)
        start = out.find("[")
        try:
            sections = _SECTIONS.validate_json(out[start:] if start >= 0 else "")
        except ValueError:
            return [
                Section(
                    section="verify",
                    verdict=Verdict.FAIL,
                    detail=f"no readiness report came back: {(err or out).strip()[-240:]}",
                    fix=f"{_TOOL} center verify (on {self.destination})",
                )
            ]
        return [staged("verify", row) for row in sections]

    def plan(self, facts: Facts, root: str) -> ExecutionPlan:
        """The destination's execution plan, its platform and root filled from the probe."""
        plan = self.board.resolver.plan(self.destination, container="none")
        profile = plan.profile.model_copy(update={"platform": facts.pixi_platform, "root": root})
        return plan.model_copy(update={"profile": profile})

    @staticmethod
    def extras() -> list[str]:
        """The extras this center's own tool was installed with."""
        found = []
        for extra, package in _EXTRAS.items():
            try:
                distribution(package)
            except PackageNotFoundError:
                continue
            found.append(extra)
        return found


def github_token() -> str:
    """This machine's GitHub token as gh holds it, empty when gh is absent or signed out.

    Read into memory for the one call that hands it on over ssh's stdin, and never printed.
    """
    try:
        done = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False, timeout=30
        )
    except OSError, subprocess.SubprocessError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""
