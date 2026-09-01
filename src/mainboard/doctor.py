# The verdict behind `mainboard doctor`: is this workspace fit to work in right now. Nothing
# here probes anything of its own. Each section asks the subsystem that already owns the
# question, the manifest loader, the compile state and the wheel audit, the compute survey, and
# every verification gate the workspace declares, then turns its answer into one line with the
# command that repairs it.

from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum, auto
from functools import partial
from shlex import join, split
from typing import TYPE_CHECKING

from patos import FrozenModel
from plumbum.commands.processes import ProcessTimedOut

from . import staleness
from .compute import Access, Survey
from .core.errors import MissionError
from .core.project import Project
from .engines.compile.backend import EnvironmentAudit
from .engines.compile.provisioner import Provisioner
from .engines.compile.state import SyncState

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .board import Board
    from .dispatch.onboard import HostSetup

# The tool this workspace answers to, so no message below spells the binary's name.
_TOOL = Project().name

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

    The sections answer the questions asked before starting work: does the manifest still say
    something coherent, is the environment on this disk the one it describes, what compute can
    be reached, and does every gate this workspace declares still come back clean. A gate is a
    command in the manifest, so which further questions get asked is the workspace's decision
    rather than this package's, and a tool joins the report by being declared instead of by
    being known here. Every probe is bounded and they run together, so the whole report takes as
    long as its slowest single question, and a section that cannot answer says so rather than
    taking the report down with it.
    """

    def __init__(
        self,
        board: Board,
        *,
        env: str = "",
        survey: Survey | None = None,
        probe: Callable[[str, float], tuple[int, str]] | None = None,
    ) -> None:
        """board: the workspace being examined.

        env: the environment to examine, the board profile's own when empty.
        survey: the fleet probe, the workspace's own when None.
        probe: runs a declared gate's command under its deadline and answers with its exit
            status and output, the workspace runner when None.
        """
        self.board = board
        self.env = env
        self.survey = survey or Survey(board)
        self.probe = probe or self.through_runner

    def environment(self) -> Section:
        """Whether what is installed answers to the manifest, and still imports.

        Three separate ways a workspace goes wrong and one line covering all three. The lock
        may have been solved from a manifest this one no longer is, an environment may have
        been provisioned before an edit nobody re-installed, and a wheel may have lost the
        files it declared underneath pixi, which no lock ever notices because the lock only
        knows the package is recorded as installed.
        """
        provisioner = Provisioner(self.board.root, self.board.manifest)
        environment = self.board.plan(env=self.env, container="none").env
        directory = provisioner.environment_dir(environment)
        pixi = provisioner.pixi_for(environment)
        compiler = provisioner.compiler_for(environment)
        install = f"{_TOOL} install {environment}"
        if not pixi.manifest.exists():
            return Section(
                section="environment",
                verdict=Verdict.WARN,
                detail=f"{environment}: nothing compiled yet",
                fix=f"{install} --resolve",
            )
        state = SyncState.load(directory)
        installed = pixi.ready(environment)
        faults: list[str] = []
        stale = False
        lock_stale = (
            not pixi.lock.exists()
            or state.environment != environment
            or state.solved_from != compiler.resolution_digest()
        )
        if lock_stale:
            faults.append("pixi.lock was not solved from this manifest")
        if installed and state.compiled_from != compiler.digest():
            stale = True
            faults.append(f"compiled before the current manifest: {environment}")
        damaged = (
            sorted(EnvironmentAudit(pixi.env_prefix(environment)).suspect()) if installed else []
        )
        if damaged:
            faults.append(f"needs reinstalling: {', '.join(damaged)}")
        if faults:
            return Section(
                section="environment",
                verdict=Verdict.FAIL,
                detail="; ".join(faults),
                fix=f"{install} --resolve" if stale or lock_stale else install,
            )
        if not installed:
            return Section(
                section="environment",
                verdict=Verdict.WARN,
                detail=f"never installed: {environment}",
                fix=install,
            )
        return Section(
            section="environment",
            verdict=Verdict.PASS,
            detail=f"{environment} is provisioned, fresh and whole",
        )

    def fleet(self, setups: Mapping[str, HostSetup] | None = None) -> Section:
        """What compute this workspace can reach, and what stands between it and the rest.

        Nothing here fails. A host that is asleep and a provider nobody has a key for are both
        facts about the world rather than about this workspace, and calling them broken would
        make the exit status mean the network instead of the code.

        setups: the onboarding records, read on the cache's own thread when this section runs
            inside the report's pool; the survey reads them itself when None.
        """
        paths = self.survey.paths(setups)
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

    def hosts(self, setups: Mapping[str, HostSetup] | None = None) -> Section:
        """Whether an onboarded host's environment still matches the manifest as it reads now.

        A host is provisioned once and the manifest can move any number of times after that,
        and nothing before this ever asked the two again whether they still agreed. This is the
        same digest `environment` already asks of this machine, compared instead against what
        each host was last provisioned from. A host onboarded before this field existed answers
        `unrecorded` rather than diverged, since there is nothing to compare against yet.

        setups: the onboarding records, read on the cache's own thread when this section runs
            inside the report's pool; the survey reads them itself when None.
        """
        setups = self.survey.onboarded() if setups is None else setups
        provisioner = Provisioner(self.board.root, self.board.manifest)
        diverged: list[str] = []
        for host, setup in setups.items():
            try:
                current = provisioner.compiler_for(setup.env).digest()
            except MissionError:
                current = ""
            if setup.digest and setup.digest != current:
                diverged.append(host)
        diverged.sort()
        if not diverged:
            return Section(
                section="hosts",
                verdict=Verdict.PASS,
                detail=f"{len(setups)} onboarded, none diverged from the current manifest",
            )
        return Section(
            section="hosts",
            verdict=Verdict.WARN,
            detail=f"diverged from the current manifest: {', '.join(diverged)}",
            fix=f"{_TOOL} setup {diverged[0]} --sync-only",
        )

    def gate(self, name: str) -> Section:
        """One declared verification gate's own verdict on this workspace.

        Four findings, and the order they are told apart in is the order they mean different
        things. A clean exit is the gate saying so. A gate that named where its failures live
        and printed them is broken, in the words it chose, since what counts as a failure is its
        judgment and stays there. A gate that promised a report and produced none never ran at
        all, usually because nothing installed it, which is a word rather than a broken
        workspace. Anything else is a plain command that exited nonzero, and its last line is
        where such a command puts its complaint.

        name: the `[gates.<name>]` table this section reports on.
        """
        gate = self.board.manifest.gates[name]
        repair = f"{_TOOL} run -- {gate.run}"
        try:
            status, output = self.probe(gate.run, gate.timeout)
        except ProcessTimedOut:
            return Section(
                section=name,
                verdict=Verdict.WARN,
                detail=f"`{gate.run}` did not answer within {gate.timeout:.0f}s",
                fix=repair,
            )
        if not status:
            return Section(
                section=name, verdict=Verdict.PASS, detail=f"`{gate.run}` reports nothing broken"
            )
        if breakages := gate.breakages(output):
            return Section(
                section=name,
                verdict=Verdict.FAIL,
                detail=f"{len(breakages)} breakages: {', '.join(breakages)}",
                fix=repair,
            )
        if gate.report:
            return Section(
                section=name,
                verdict=Verdict.WARN,
                detail=f"`{gate.run}` exited {status} without a report, is it installed",
                fix=gate.install or repair,
            )
        return Section(
            section=name,
            verdict=Verdict.FAIL,
            detail=Doctor._complaint(output) or f"`{gate.run}` exited {status}",
            fix=repair,
        )

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

    def sections(self) -> list[Section]:
        """Every section, the manifest first because everything after it reads the manifest.

        A manifest that will not load is the whole report, since an environment digest, a host
        roster, a task list and the gate roster itself are all things that manifest was going to
        supply, and inventing verdicts for them from a file nobody could parse would say nothing
        true.

        Everything durable the pool will need is read here first, on this thread. The dispatch
        cache is one SQLite connection that only its opening thread may use, so a fleet probe
        that reached for the onboarding records from inside the pool would open that connection
        there and leave the interpreter closing it from here at exit. The shared subsystems are
        built here for the same reason, which is also why building them is locked.
        """
        manifest = self.manifest()
        if manifest.verdict is Verdict.FAIL:
            return [manifest]
        setups = self.survey.onboarded()
        asked: list[Callable[[], Section]] = [
            self.environment,
            self.snapshot,
            partial(self.fleet, setups),
            partial(self.hosts, setups),
            *(partial(self.gate, name) for name in self.board.manifest.gates),
        ]
        with ThreadPoolExecutor(max_workers=len(asked)) as pool:
            return [manifest, *pool.map(lambda question: question(), asked)]

    def snapshot(self) -> Section:
        """Whether the installed CLI snapshot still answers for the source tree it was built from.

        The one drift a lock never notices, since the snapshot is a uv tool environment beside
        the workspace rather than inside it. A checkout running its own source has nothing to
        be stale against and passes with that word.
        """
        found = staleness.check()
        if found.stale:
            return Section(
                section="snapshot", verdict=Verdict.FAIL, detail=found.detail, fix=join(found.fix)
            )
        return Section(section="snapshot", verdict=Verdict.PASS, detail=found.detail)

    def through_runner(self, command: str, timeout: float) -> tuple[int, str]:
        """Run `command` through this workspace's own runner, bounded, and capture what it said.

        The same staged line `run` uses, so a gate is reached through the environment this
        workspace installed rather than through whatever interpreter happens to be on PATH when
        the report is asked for.

        command: the gate's command line.
        timeout: the gate's own deadline in seconds.
        """
        plan = self.board.plan(env=self.env, container="none")
        result = Provisioner(self.board.root, self.board.manifest).capture(
            split(command), plan.env, timeout=timeout
        )
        return result.returncode, result.stdout

    @staticmethod
    def _complaint(output: str) -> str:
        """The last thing a command said, which is where a command line tool puts its complaint."""
        spoken = [line.strip() for line in output.splitlines() if line.strip()]
        return spoken[-1] if spoken else ""
