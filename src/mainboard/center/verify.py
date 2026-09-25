# `center verify`: is this machine ready to be the center, asked of the machine itself.
#
# The readiness suite a center answers on its own, and the last step of `center migrate`, which
# runs this same verb on the destination and reports what it says. It is every question at once:
# this machine's git tooling (with its safe settings applied), the machine against the workspace
# (the census judged the way `facts` judges any host), the workspace doctor, the plan the manifest
# resolves to here, a smoke run of Python, torch and CUDA in the default environment, whether
# every lint tool can start, the repository tree, every agent's configuration, the default
# environment on every agent shell's PATH, and the scripts that would behave differently here.

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from ..core.section import Section, Verdict, staged
from ..engines.compile.provisioner import Provisioner
from ..fitness import Fitness, Role
from ..probe.system import System
from ..workstation import Readiness, Workstation
from .agents import Agents, dotenv
from .exposure import Exposure, Spawn, directories, executables
from .portable import Portability

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ..board import Board

# What the smoke run asks the default environment's Python, printed as one JSON line.
_SMOKE = (
    "import json, torch; available = torch.cuda.is_available(); print(json.dumps({"
    "'torch': torch.__version__, 'cuda': torch.version.cuda or '', 'available': available, "
    "'device': torch.cuda.get_device_name(0) if available else ''}))"
)

# How long the smoke run may take: a first import of torch on a cold disk is tens of seconds.
_SMOKE_SECONDS = 600.0

# How long one shell may take to say where each tool resolves.
_SHELL_SECONDS = 60.0


def spawned(command: Sequence[str], environment: Mapping[str, str]) -> tuple[int, str]:
    """Run `command` under `environment`, bounded, answering its status and joined output."""
    try:
        done = subprocess.run(
            list(command),
            env=dict(environment),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SHELL_SECONDS,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return 127, ""
    return done.returncode, done.stdout + done.stderr


class Verification:
    """The readiness suite, run on the machine that asks.

    workstation: this machine's git tooling, its own when None.
    system: this machine's census, read when None.
    smoke: runs the smoke command in the default environment, answering status and output.
    """

    def __init__(
        self,
        board: Board,
        *,
        workstation: Workstation | None = None,
        system: System | None = None,
        home: Path | None = None,
        spawn: Spawn = spawned,
        smoke: Callable[[], tuple[int, str]] | None = None,
    ) -> None:
        self.board = board
        self.workstation = workstation or Workstation(board.root)
        self.system = system or System.collected(board.root)
        self.home = home or Path.home()
        self.spawn = spawn
        self.smoked = smoke or self._smoke

    @property
    def provisioner(self) -> Provisioner:
        """The compile and install surface of this workspace."""
        return Provisioner(self.board.root, self.board.manifest)

    def sections(self) -> list[Section]:
        """Every readiness row, the tooling first since every later repair is a git command."""
        judged = Fitness(self.board.root, self.board.manifest).judge(self.system, role=Role.CENTER)
        tree = self.board.git()
        agents = Agents(
            self.board.root,
            home=self.home,
            environment=os.environ,
            dotenv=dotenv(self.board.root / ".env"),
            which=shutil.which,
        )
        return [
            *self.tooling(),
            *(staged("machine", row) for row in judged),
            *self.board.doctor().sections(),
            self.plan(),
            self.smoke(),
            self.lint(),
            *self.tree(),
            *agents.sections(),
            *self.path(),
            *Portability(tree.owned(), self.board.manifest.lint.exclude).sections(),
        ]

    def tooling(self) -> list[Section]:
        """This machine's git tooling, each check after its safe repair ran."""
        return [
            Section(
                section=f"workstation: {found.check}",
                verdict=_verdict(found),
                detail=found.detail,
                fix=found.fix,
            )
            for found in self.workstation.examine()
        ]

    def plan(self) -> Section:
        """What `mainboard check --on local` resolves to: the plan this center runs under."""
        try:
            plan = self.board.plan(container="none")
        except MissionError as refusal:
            return Section(
                section="check", verdict=Verdict.FAIL, detail=str(refusal), fix="mainboard check"
            )
        return Section(
            section="check",
            verdict=Verdict.PASS,
            detail=f"local runs {plan.env} bare, {len(self.board.manifest.tasks)} tasks declared",
        )

    def smoke(self) -> Section:
        """Whether Python, torch and CUDA answer in the default environment."""
        status, said = self.smoked()
        line = next((line for line in reversed(said.splitlines()) if line.startswith("{")), "")
        if status or not line:
            spoken = [line for line in said.splitlines() if line.strip()]
            return Section(
                section="smoke",
                verdict=Verdict.FAIL,
                detail=f"python -c 'import torch' failed: {spoken[-1] if spoken else status}",
                fix="mainboard install",
            )
        found = json.loads(line)
        if self.system.gpus and not found["available"]:
            return Section(
                section="smoke",
                verdict=Verdict.FAIL,
                detail=(
                    f"torch {found['torch']} sees no CUDA device, though the driver lists "
                    f"{self.system.gpus[0].name}"
                ),
                fix="compare `machine: cuda-builds` above, then mainboard install",
            )
        where = f"on {found['device']} (CUDA {found['cuda']})" if found["available"] else "on CPU"
        return Section(
            section="smoke", verdict=Verdict.PASS, detail=f"torch {found['torch']} {where}"
        )

    def lint(self) -> Section:
        """Whether every declared lint tool's program is one this center can start."""
        tools = self.board.manifest.lint.tools
        search = os.pathsep.join(
            [
                *map(str, self._folders()),
                *map(str, self.provisioner.binaries("default")),
                os.environ.get("PATH", ""),
            ]
        )
        missing = sorted(
            {
                program
                for tool in tools.values()
                for check in (True, False)
                if shutil.which(program := tool.argv(check=check)[0], path=search) is None
            }
        )
        if missing:
            return Section(
                section="lint",
                verdict=Verdict.FAIL,
                detail=f"lint runs {', '.join(missing)}, which the environment does not provide",
                fix="mainboard install",
            )
        return Section(
            section="lint", verdict=Verdict.PASS, detail=f"{len(tools)} lint tools can start"
        )

    def tree(self) -> list[Section]:
        """The repository tree: every owned repository checked out, current and saved."""
        tree = self.board.git()
        absent = [
            child.name
            for repo in tree.owned()
            for child in repo.children
            if child.owned and not child.initialized
        ]
        states = tree.status()
        behind = [state.repo for state in states if state.behind]
        unsaved = [
            state.repo for state in states if state.changed or state.untracked or state.ahead
        ]
        rows = []
        if absent:
            rows.append(
                Section(
                    section="git: checkout",
                    verdict=Verdict.WARN,
                    detail=f"owned submodules not checked out: {', '.join(absent)}",
                    fix="mainboard center git pull",
                )
            )
        if behind:
            rows.append(
                Section(
                    section="git: behind",
                    verdict=Verdict.WARN,
                    detail=f"behind upstream: {', '.join(behind)}",
                    fix="mainboard center git pull",
                )
            )
        rows.append(
            Section(
                section="git: status",
                verdict=Verdict.PASS,
                detail=f"{len(states)} owned repositories"
                + (f"; unsaved work in {', '.join(unsaved)}" if unsaved else ", all saved"),
            )
        )
        return rows

    def path(self) -> list[Section]:
        """The default environment on every shell's PATH, then where each tool resolves there."""
        prefix = self.provisioner.pixi_for("default").env_prefix("default")
        if not (prefix / "conda-meta").is_dir():
            return [
                Section(
                    section="path",
                    verdict=Verdict.WARN,
                    detail="the default environment is not installed, so nothing is on PATH",
                    fix="mainboard install",
                )
            ]
        exposure = Exposure(
            self._folders(),
            system=self.system.system,
            home=self.home,
            shells=self.system.shells,
            spawn=self.spawn,
        )
        names = executables(prefix, sorted(self.board.manifest.deps), self.system.system)
        return [exposure.apply(), *exposure.verify(names)]

    def _folders(self) -> list[Path]:
        """The default environment's executable directories, in PATH order."""
        prefix = self.provisioner.pixi_for("default").env_prefix("default")
        return directories(prefix, self.system.system or platform.system())

    def _smoke(self) -> tuple[int, str]:
        """Run the smoke command in the default environment, bounded."""
        result = self.provisioner.capture(
            ["python", "-c", _SMOKE], "default", timeout=_SMOKE_SECONDS
        )
        return result.returncode, result.stdout + result.stderr


def _verdict(found: Readiness) -> Verdict:
    """FAIL for tooling that is broken, WARN for a fix still owed, PASS otherwise."""
    if found.broken:
        return Verdict.FAIL
    return Verdict.WARN if found.fix else Verdict.PASS
