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

import re
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import TypeAdapter

from ..core.errors import MissionError
from ..core.project import Project
from ..core.section import Section, Verdict, staged
from ..dispatch.onboard import Bootstrap, Onboarding
from ..dispatch.shared import announce
from ..dispatch.shells import dialect_for, open_shell, plain_errors
from ..dispatch.targets import probe_capabilities
from ..dispatch.transport import HostUnreachable, SshTransport
from ..fitness import Fitness, Role
from ..probe.system import System
from .carrier import Carrier
from .state import GITHUB, Carried, Destination, Parcel, packed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..board import Board
    from ..context.plan import ExecutionPlan
    from ..dispatch.shared import Watcher
    from ..dispatch.targets import Facts
    from ..git.repo import Repo
    from ..git.tree import Tree

_TOOL = Project().name

# The findings about the destination that make every later step pointless: a platform the
# workspace never declared, and a lock with nothing for it, both of which the install would only
# rediscover after the clone and the copy had spent their time.
_BLOCKING = frozenset({"platform", "lock"})

# The optional extras of this tool worth carrying, each with the distribution that shows this
# center installed it: the destination gets the same tool, plotting included.
_EXTRAS = {"plot": "seaborn", "wandb": "wandb"}

# Where the workspace goes on the destination unless told otherwise: the center, unlike a dispatch
# target, holds the human checkout, and the destination expands the `~` in its own spelling.
_CHECKOUT = "~/projects"

# A remote reached over ssh at GitHub, `git@github.com:owner/name` or `ssh://git@github.com/...`.
_GITHUB_SSH = re.compile(r"(ssh://)?([^@/:]+@)?github\.com[:/]")

# What a Windows filesystem cannot hold in a path, as git for Windows judges it: a character NTFS
# forbids, a name ending in a space or a dot, or a device name, whatever extension follows it.
_UNHOLDABLE = re.compile(
    r'[<>:"|?*\x01-\x1f]|[ .](/|$)'
    r"|(^|/)(con|prn|aux|nul|conin\$|conout\$|com[1-9]|lpt\d) *(\.[^/]*)?(/|$)",
    re.IGNORECASE,
)

# How long the destination's own `center verify` may run, a torch and CUDA smoke and every lint
# tool's probe included, and the silence its ssh rides out while that smoke loads the machine:
# the default 45 s dropped it with `Timeout, server not responding` (pedro-home, 2026-09-26).
_VERIFY_SECONDS = 1800.0
_PATIENT = {"server_alive_interval": 30.0, "server_alive_count": 10}

# A readiness report as `center verify --json` prints it.
_SECTIONS = TypeAdapter(list[Section])


class Signed(FrozenModel):
    """What the destination's gh said to the login it was handed."""

    signed: bool
    detail: str = ""


class Written(FrozenModel):
    """How many shipped files changed bytes on the destination."""

    written: int


class Trusted(FrozenModel):
    """How many ssh host blocks and known host lines the destination took."""

    blocks: int
    keys: int


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
    root: where the workspace goes there, `~/projects` when empty.
    transport: the ssh policy every call rides.
    watch: announces each step as it begins.
    home: this machine's home directory, whose agent state and ssh keys are carried.
    token: answers this machine's GitHub token, empty when it has none.
    host_keys: answers GitHub's ssh host keys as GitHub publishes them, none when unreachable.
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
        host_keys: Callable[[], list[str]] | None = None,
    ) -> None:
        self.board = board
        self.destination = destination
        self.root = root
        self.transport = transport or SshTransport()
        self.watch = watch or announce
        self.home = home or Path.home()
        self.token = token or github_token
        self.host_keys = host_keys or github_host_keys

    def run(self) -> list[Section]:
        """Every step in order, stopping only where continuing could not change the answer."""
        report = self.preflight()
        if any(row.verdict is Verdict.FAIL for row in report):
            return report
        self.watch(f"probing {self.destination}")
        facts = self.reach()
        carrier = Carrier(self.destination, facts.uv, self.transport)
        where = carrier.call("where", {"root": self.root or _CHECKOUT}, Destination)
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
        report.append(self.trust(carrier, carried))
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
        tree = self.board.git()
        unserved = {repo.name for repo in tree.owned() if not repo.serves(repo.head())}
        rows: list[Section] = []
        for state in tree.status():
            if state.repo in unserved:
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
        with open_shell(self.plan(facts, facts.home), facts.home, ssh=self.transport) as shell:
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

    def trust(self, carrier: Carrier, carried: Carried) -> Section:
        """The ssh host blocks and known host lines the destination lacks, added beside its own.

        GitHub's host keys come from GitHub's API over the gh login when a clone reaches GitHub
        over ssh and this machine never recorded them, since a first connection must not be what
        vouches for them.
        """
        tree = self.board.git()
        known = carried.known_hosts()
        urls = [tree.root.url, *(child.declared for _, child in _submodules(tree))]
        wanted = any(_GITHUB_SSH.match(url) for url in urls) and not any(
            GITHUB in line.split()[0].split(",") for line in known
        )
        fetched = self.host_keys() if wanted else []
        known += [f"{GITHUB} {key}" for key in fetched]
        added = carrier.call("ssh", {"blocks": carried.ssh_config(), "known": known}, Trusted)
        detail = f"{added.blocks} host blocks and {added.keys} known host lines added"
        if wanted and not fetched:
            return Section(
                section="carry ssh config",
                verdict=Verdict.WARN,
                detail=f"{detail}; GitHub's host keys unknown here and gh could not fetch them",
                fix=f"gh auth login, then {_TOOL} center migrate {self.destination}",
            )
        return Section(section="carry ssh config", verdict=Verdict.PASS, detail=detail)

    def clone(self, carrier: Carrier, place: Destination) -> list[Section]:
        """The root at this HEAD and each owned submodule at its pointer, then the foreign rest.

        On Windows each repository's checkout leaves out the tracked paths NTFS cannot hold,
        which a warning names, instead of failing that repository whole.
        """
        tree = self.board.git()
        pairs = _submodules(tree)
        submodules = [
            [parent.name, parent.relative(child), child.declared] for parent, child in pairs
        ]
        foreign = [
            child.name
            for parent in tree.owned()
            for child in parent.children
            if child.initialized and not child.owned
        ]
        repos = [tree.root, *(child for _, child in pairs)]
        excluded = {
            repo.name: paths
            for repo in (repos if place.system == "Windows" else [])
            if (paths := [path for path in repo.tracked() if _UNHOLDABLE.search(path)])
        }
        steps = carrier.call(
            "clone",
            {
                "root": place.root,
                "url": tree.root.url,
                "branch": tree.root.branch(),
                "commit": tree.root.head(),
                "submodules": submodules,
                "excluded": excluded,
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
        rows += [
            Section(
                section=f"unholdable {repo}",
                verdict=Verdict.WARN,
                detail=f"{len(paths)} tracked paths Windows cannot hold left out: "
                + ", ".join(paths[:3]),
                fix=f'rename them in {repo} (no <>:"|?*, trailing space or dot, or device name)',
            )
            for repo, paths in excluded.items()
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
        """The destination's own `center verify`, which is the readiness suite run there.

        It rides one ssh process whose keepalives are patient enough for the smoke that loads
        the machine, bounded by a wall clock past any full verify.
        """
        self.watch(f"verifying {self.destination}")
        dialect = dialect_for(plan.profile)
        ssh = self.transport.model_copy(update=_PATIENT)
        line = dialect.stage(plan, root, command=f"{_TOOL} center verify --json", activate=False)
        argv = dialect.one_shot(ssh, self.destination, line)
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
            said = plain_errors(err or out).strip()[-240:]
            return [
                Section(
                    section="verify",
                    verdict=Verdict.FAIL,
                    detail=f"no readiness report came back: {said}",
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


def _submodules(tree: Tree) -> list[tuple[Repo, Repo]]:
    """Each checked-out owned submodule with its parent, parents first."""
    return [
        (parent, child)
        for parent in tree.owned()
        for child in parent.children
        if child.initialized and child.owned
    ]


def github_token() -> str:
    """This machine's GitHub token as gh holds it, empty when gh is absent or signed out.

    Read into memory for the one call that hands it on over ssh's stdin, and never printed.
    """
    return _gh("auth", "token")


def github_host_keys() -> list[str]:
    """GitHub's ssh host keys as its API publishes them over gh's https, none when unreachable."""
    return _gh("api", "meta", "--jq", ".ssh_keys[]").splitlines()


def _gh(*arguments: str) -> str:
    """What `gh arguments` prints, empty when gh is absent or refuses."""
    try:
        done = subprocess.run(
            ["gh", *arguments], capture_output=True, text=True, check=False, timeout=30
        )
    except OSError, subprocess.SubprocessError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""
