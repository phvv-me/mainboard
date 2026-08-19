# The verdict behind `mainboard doctor`: is this workspace fit to work in right now. Nothing
# here probes anything of its own. Each section asks the subsystem that already owns the
# question, the manifest loader, the compile state and the wheel audit, the compute survey and
# the proof workbench, and turns its answer into one line with the command that repairs it.

import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum import local as localhost
from plumbum.commands.processes import ProcessTimedOut

from .compute import Access, Survey
from .core.errors import MissionError
from .core.project import Project
from .engines.compile.backend import EnvironmentAudit, Process
from .engines.compile.provisioner import Provisioner
from .engines.compile.state import SyncState

if TYPE_CHECKING:
    from collections.abc import Callable

    from .board import Board

# The tool this workspace answers to, so no message below spells the binary's name.
_TOOL = Project().name

# The proof workbench, its root marker, and the verb that reports on it. mainboard knows only
# that much of it: a config naming a workspace, a doctor verb, an exit status, and a list of
# breakages. Everything inside that report belongs to the workbench.
_MATH = "atpx"
_MATH_CONFIG = f"{_MATH}.toml"
_MATH_VERB = f"{_MATH} doctor"

# The math probe's deadline. It reads files and settles nothing, so an answer takes seconds and
# anything past this is a workbench that has gone looking at the network or hung.
_MATH_TIMEOUT = 90.0

# The compute paths that are usable as they stand, so a survey row outside this set is
# something the report has to say a word about.
_USABLE = frozenset({Access.HERE, Access.READY, Access.KEYED})


class Verdict(StrEnum):
    """How one section came back: fit, fit with something worth saying, or broken."""

    PASS = auto()
    WARN = auto()
    FAIL = auto()


class Section(FrozenModel):
    """One area of the workspace, judged, with the single command that repairs it.

    section: the area reported on.
    verdict: whether it is fit, worth a word, or broken.
    detail: the one line behind the verdict.
    fix: the command that repairs it, empty when nothing needs repairing.
    """

    section: str
    verdict: Verdict
    detail: str
    fix: str = ""


class Doctor:
    """One verdict over the workspace, composed from the probes each subsystem already owns.

    The sections answer the four questions asked before starting work: does the manifest still
    say something coherent, is the environment on this disk the one it describes, what compute
    can be reached, and does the mathematics still hold. Every probe is bounded and they run
    together, so the whole report takes as long as its slowest single question, and a section
    that cannot answer says so rather than taking the report down with it.
    """

    def __init__(
        self,
        board: Board,
        *,
        survey: Survey | None = None,
        math: Callable[[str], tuple[int, str]] | None = None,
    ) -> None:
        """board: the workspace being examined.

        survey: the fleet probe, the workspace's own when None.
        math: runs the workbench's verb and answers with its exit status and output, the
            workspace runner when None.
        """
        self.board = board
        self.given = survey
        self.math_probe = math or self.through_runner

    def sections(self) -> list[Section]:
        """Every section, the manifest first because everything after it reads the manifest.

        A manifest that will not load is the whole report, since an environment digest, a host
        roster and a task list are all things that manifest was going to supply, and inventing
        verdicts for them from a file nobody could parse would say nothing true.
        """
        manifest = self.manifest()
        if manifest.verdict is Verdict.FAIL:
            return [manifest]
        probes: list[Callable[[], Section]] = [self.environment, self.fleet]
        if self.mathematical():
            probes.append(self.math)
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            return [manifest, *pool.map(lambda probe: probe(), probes)]

    def manifest(self) -> Section:
        """Whether the workspace manifest still parses, interpolates and validates."""
        try:
            loaded = self.board.manifest
        except MissionError as refusal:
            return Section(
                section="manifest",
                verdict=Verdict.FAIL,
                detail=str(refusal).splitlines()[0],
                fix=f"{_TOOL} check",
            )
        return Section(
            section="manifest",
            verdict=Verdict.PASS,
            detail=(
                f"{loaded.workspace.name}: {len(loaded.envs) + 1} environments, "
                f"{len(loaded.profiles())} hosts, {len(loaded.tasks)} tasks"
            ),
        )

    def environment(self) -> Section:
        """Whether what is installed answers to the manifest, and still imports.

        Three separate ways a workspace goes wrong and one line covering all three. The lock
        may have been solved from a manifest this one no longer is, an environment may have
        been provisioned before an edit nobody re-installed, and a wheel may have lost the
        files it declared underneath pixi, which no lock ever notices because the lock only
        knows the package is recorded as installed.
        """
        provisioner = Provisioner(self.board.root, self.board.manifest)
        pixi = provisioner.pixi
        if not pixi.manifest.exists():
            return Section(
                section="environment",
                verdict=Verdict.WARN,
                detail="nothing compiled yet",
                fix=f"{_TOOL} install --resolve",
            )
        state = SyncState.load(provisioner.out)
        declared = ["default", *sorted(self.board.manifest.envs)]
        installed = [env for env in declared if pixi.ready(env)]
        digest = provisioner.compiler.digest()
        faults = []
        if not pixi.lock.exists() or state.solved_from != provisioner.compiler.resolution_digest():
            faults.append("pixi.lock was not solved from this manifest")
        if stale := [env for env in installed if state.envs.get(env) != digest]:
            faults.append(f"compiled before the current manifest: {', '.join(stale)}")
        damaged = sorted(
            {
                package
                for env in installed
                for package in EnvironmentAudit(pixi.env_prefix(env)).suspect()
            }
        )
        if damaged:
            faults.append(f"needs reinstalling: {', '.join(damaged)}")
        if faults:
            return Section(
                section="environment",
                verdict=Verdict.FAIL,
                detail="; ".join(faults),
                fix=f"{_TOOL} install {stale[0]} --resolve"
                if stale
                else f"{_TOOL} install --resolve",
            )
        if absent := [env for env in declared if env not in installed]:
            return Section(
                section="environment",
                verdict=Verdict.WARN,
                detail=f"{len(installed)} provisioned and fresh, never installed: "
                f"{', '.join(absent)}",
                fix=f"{_TOOL} install {absent[0]}",
            )
        return Section(
            section="environment",
            verdict=Verdict.PASS,
            detail=f"{len(installed)} environments provisioned, fresh and whole",
        )

    def fleet(self) -> Section:
        """What compute this workspace can reach, and what stands between it and the rest.

        Nothing here fails. A host that is asleep and a provider nobody has a key for are both
        facts about the world rather than about this workspace, and calling them broken would
        make the exit status mean the network instead of the code.
        """
        paths = (self.given or Survey(self.board)).paths()
        ready = [path for path in paths if path.access in _USABLE]
        cold = [path.name for path in paths if path.access is Access.REACHABLE]
        down = [path.name for path in paths if path.access is Access.UNREACHABLE]
        unkeyed = [path.name for path in paths if path.access is Access.UNKEYED]
        notes = [
            note
            for note in (
                f"answering but never set up: {', '.join(cold)}" if cold else "",
                f"not answering: {', '.join(down)}" if down else "",
                f"no credentials here: {', '.join(unkeyed)}" if unkeyed else "",
            )
            if note
        ]
        if not notes:
            return Section(
                section="fleet", verdict=Verdict.PASS, detail=f"{len(ready)} paths usable now"
            )
        return Section(
            section="fleet",
            verdict=Verdict.WARN,
            detail=f"{len(ready)} usable, {'; '.join(notes)}",
            fix=f"{_TOOL} setup {cold[0]}" if cold else f"{_TOOL} compute",
        )

    def math(self) -> Section:
        """The proof workbench's own verdict on every blueprint this workspace holds.

        mainboard reads two things out of that report, the exit status and the list of
        breakages, both of which the workbench documents as its contract. What counts as a
        breakage is the workbench's judgment and stays there, so a claim that failed and a link
        that points at nothing arrive here already named and are passed on as written.
        """
        try:
            status, output = self.math_probe(_MATH_VERB)
        except ProcessTimedOut:
            return Section(
                section="math",
                verdict=Verdict.WARN,
                detail=f"`{_MATH_VERB}` did not answer within {_MATH_TIMEOUT:.0f}s",
                fix=f"{_TOOL} run -- {_MATH_VERB}",
            )
        if status and not (breakages := _breakages(output)):
            return Section(
                section="math",
                verdict=Verdict.WARN,
                detail=f"`{_MATH_VERB}` exited {status} without a report, is it installed",
                fix=f"{_TOOL} add {_MATH} -l python",
            )
        if status:
            return Section(
                section="math",
                verdict=Verdict.FAIL,
                detail=f"{len(breakages)} breakages: {', '.join(breakages)}",
                fix=f"{_TOOL} run -- {_MATH_VERB}",
            )
        return Section(section="math", verdict=Verdict.PASS, detail="every claim settled")

    def mathematical(self) -> bool:
        """Whether this workspace roots a proof workbench worth reporting on."""
        try:
            config = (self.board.root / _MATH_CONFIG).read_text(encoding="utf-8")
        except OSError:
            return False
        return "workspace" in tomllib.loads(config)

    def through_runner(self, command: str) -> tuple[int, str]:
        """Run `command` through this workspace's own runner, bounded, and capture what it said.

        The same staged line `run` uses, so the workbench is reached through the environment
        this workspace installed rather than through whatever interpreter happens to be on PATH
        when the report is asked for.
        """
        staged = self.board.line(command, container="none")
        result = Process.capture(localhost["bash"]["-lc", staged], timeout=_MATH_TIMEOUT)
        return result.returncode, result.stdout


def _breakages(output: str) -> list[str]:
    """The breakages a workbench certificate reports, empty when the output carries none.

    A tool that is not installed, or one that died before it could stamp anything, leaves
    output no certificate can be read out of, and that is the difference between a workspace
    whose mathematics is broken and one whose workbench never ran.
    """
    start = output.find("{")
    if start < 0:
        return []
    try:
        certificate = json.loads(output[start:])
    except json.JSONDecodeError:
        return []
    found = certificate.get("result", {}).get("breakages", [])
    return [str(breakage) for breakage in found] if isinstance(found, list) else []
