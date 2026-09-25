# The verdict behind `mainboard doctor`: is this workspace fit to work in right now. Nothing
# here probes anything of its own. Each section asks the subsystem that already owns the
# question, the manifest loader, the compile state and the wheel audit, the compute survey, and
# every verification gate the workspace declares, then turns its answer into one line with the
# command that repairs it. The center's own tooling is `center verify`'s question, not this one.

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from shlex import join, split
from typing import TYPE_CHECKING

from plumbum.commands.processes import ProcessTimedOut

from . import durable, staleness
from .compute import Access, Survey
from .core.errors import MissionError
from .core.project import Project
from .core.section import Section, Verdict
from .engines.compile.backend import PIXI_VERSION, POSIX_INSTALLER, EnvironmentAudit
from .engines.compile.provisioner import Provisioner
from .engines.compile.state import SyncState

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .board import Board
    from .dispatch.onboard import HostSetup
    from .durable import Settler

# The tool this workspace answers to, so no message below spells the binary's name.
_TOOL = Project().name

# The compute paths usable as they stand; every other survey row earns a word.
_USABLE = frozenset({Access.HERE, Access.KEYED})

# What the fleet row says about the paths in each state that is not usable, in reading order.
_UNUSABLE = {
    Access.REACHABLE: "answering but never set up",
    Access.PROVISIONED: "cached setup, job readiness unverified",
    Access.UNREACHABLE: "not answering",
    Access.UNKEYED: "no credentials here",
}

# What a gate cannot mean, because a gate is argv: a declared `a && b` runs `a` with three
# arguments, and the pipeline nobody ran reads as a gate that passed. These are refused by name.
_SHELL_GRAMMAR = frozenset({"&&", "||", "|", ";", ">", ">>", "<", "&", "2>", "2>&1"})


class Doctor:
    """One verdict over the workspace, composed from the probes each subsystem already owns.

    The sections answer the questions asked before starting work: does the manifest still say
    something coherent, is the environment on this disk the one it describes, what compute can
    be reached, and does every declared gate still come back clean. A gate is a manifest
    command, so a tool joins the report by being declared rather than known here. Every probe is
    bounded and they run together, so the report takes as long as its slowest question, and a
    section that cannot answer says so rather than taking the report down.
    """

    def __init__(
        self,
        board: Board,
        *,
        env: str = "",
        survey: Survey | None = None,
        probe: Callable[[str, float], tuple[int, str]] | None = None,
        settler: Settler | None = None,
    ) -> None:
        """env: the environment to examine, the board profile's own when empty.
        survey: the fleet probe, the workspace's own when None.
        probe: runs a declared gate's command under its deadline and answers with its exit
            status and output, the workspace runner when None.
        settler: the machine's periodic runner, the one this platform offers when None.
        """
        self.board = board
        self.env = env
        self.survey = survey or Survey(board)
        self.probe = probe or self.through_runner
        self.settler = settler or durable.settler(board.root)

    def environment(self, env: str = "") -> Section:
        """Whether what is installed answers to the manifest, and still imports.

        The lock may have been solved from a manifest this one no longer is, the environment
        provisioned before an edit nobody re-installed, or a wheel may have lost its files
        underneath pixi, which no lock notices. Each finding names the command that repairs THAT
        finding (a stale lock needs `--resolve`, the others the install the lock describes), since
        one command for the row left the reader guessing which findings it covered. The row names
        its environment, since the report carries one row per declared environment.

        env: the environment to examine, this report's own when empty.
        """
        provisioner = Provisioner(self.board.root, self.board.manifest)
        environment = self.board.plan(env=env or self.env, container="none").env
        directory = provisioner.environment_dir(environment)
        pixi = provisioner.pixi_for(environment)
        compiler = provisioner.compiler_for(environment)
        install = f"{_TOOL} install {environment}"
        row = partial(Section, section="environment")
        if not pixi.manifest.exists():
            return row(
                verdict=Verdict.WARN,
                detail=f"{environment}: nothing compiled yet",
                fix=f"{install} --resolve",
            )
        state = SyncState.load(directory)
        installed = pixi.ready(environment)
        damaged = (
            sorted(EnvironmentAudit(pixi.env_prefix(environment)).suspect()) if installed else []
        )
        lock_stale = (
            not pixi.lock.exists()
            or state.environment != environment
            or state.solved_from != compiler.resolution_digest()
        )
        findings: list[tuple[str, str]] = []
        solver = provisioner.solver_version()
        if solver != PIXI_VERSION:
            # Each pixi version writes some of the lock differently, so a machine off the fleet's
            # pixi rewrites the lock it is handed and builds at an address nothing dispatched
            # against: a whole dead wave and no message (2026-09-05, 0.77 against a host on 0.79).
            here = f"pixi {solver or 'is not installed'} here"
            findings.append(
                (f"{here}, and the fleet is pinned to {PIXI_VERSION}", POSIX_INSTALLER)
            )
        if lock_stale:
            findings.append(
                ("pixi.lock was not solved from this manifest", f"{install} --resolve")
            )
        if installed and state.compiled_from != compiler.digest():
            findings.append((f"compiled before the current manifest: {environment}", install))
        if damaged:
            findings.append((f"needs reinstalling: {', '.join(damaged)}", install))
        if findings:
            return row(
                verdict=Verdict.FAIL,
                detail=f"{environment}: " + "; ".join(fault for fault, _ in findings),
                fix="; ".join(dict.fromkeys(repair for _, repair in findings)),
            )
        if not installed:
            return row(verdict=Verdict.WARN, detail=f"never installed: {environment}", fix=install)
        return row(
            verdict=Verdict.PASS,
            detail=f"{environment} is provisioned, fresh and whole, on pixi {solver}",
        )

    def fleet(self, setups: Mapping[str, HostSetup] | None = None) -> Section:
        """What compute this workspace can reach, and what stands between it and the rest.

        Nothing here fails: a sleeping host and an unkeyed provider are facts about the world, and
        calling them broken would make the exit status mean the network instead of the code.

        setups: the onboarding records (see `sections`), read by the survey itself when None.
        """
        paths = self.survey.paths(setups)
        ready = [path for path in paths if path.access in _USABLE]
        notes = [
            f"{label}: {', '.join(names)}"
            for access, label in _UNUSABLE.items()
            if (names := [path.name for path in paths if path.access is access])
        ]
        if not notes:
            return Section(
                section="fleet", verdict=Verdict.PASS, detail=f"{len(ready)} paths usable now"
            )
        return Section(
            section="fleet",
            verdict=Verdict.WARN,
            detail=f"{len(ready)} usable, {'; '.join(notes)}",
            fix=f"{_TOOL} compute",
        )

    def hosts(self, setups: Mapping[str, HostSetup] | None = None) -> Section:
        """Whether an onboarded host's environment still matches the manifest as it reads now.

        The same digest `environment` asks of this machine, compared against what each host was
        last provisioned from, since the manifest moves after a host is provisioned. A host with
        no recorded digest has nothing to compare against and never counts as diverged; one whose
        environment the manifest no longer declares does.

        setups: the onboarding records (see `sections`), read from the survey when None.
        """
        setups = self.survey.onboarded() if setups is None else setups
        provisioner = Provisioner(self.board.root, self.board.manifest)
        diverged = sorted(
            host
            for host, setup in setups.items()
            if setup.digest and setup.digest != _digest(provisioner, setup.env)
        )
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

        Told apart in the order they mean different things. A clean exit is the gate saying so.
        A gate that printed where its failures live is broken in the words it chose, since what
        counts as a failure is its judgment. A gate that promised a report and produced none never
        ran, usually because nothing installed it, which is a word rather than a broken
        workspace. Anything else is a plain command that exited nonzero, whose last line is its
        complaint.

        name: the `[gates.<name>]` table this section reports on.
        """
        gate = self.board.manifest.gates[name]
        row = partial(Section, section=name)
        repair = f"{_TOOL} run -- {gate.run}"
        try:
            status, output = self.probe(gate.run, gate.timeout)
        except ProcessTimedOut:
            detail = f"`{gate.run}` did not answer within {gate.timeout:.0f}s"
            return row(verdict=Verdict.WARN, detail=detail, fix=repair)
        if not status:
            return row(verdict=Verdict.PASS, detail=f"`{gate.run}` reports nothing broken")
        if breakages := gate.breakages(output):
            detail = f"{len(breakages)} breakages: {', '.join(breakages)}"
            return row(verdict=Verdict.FAIL, detail=detail, fix=repair)
        if gate.report:
            detail = f"`{gate.run}` exited {status} without a report, is it installed"
            return row(verdict=Verdict.WARN, detail=detail, fix=gate.install or repair)
        complaint = next(
            (line.strip() for line in reversed(output.splitlines()) if line.strip()), ""
        )
        return row(
            verdict=Verdict.FAIL, detail=complaint or f"`{gate.run}` exited {status}", fix=repair
        )

    def layout(self) -> Section:
        """Whether the old root still holds an environment directory, one line per name.

        A workspace provisioned before every pixi prefix moved under its own environment
        directory keeps the old root's `envs/`. An environment the current layout has since
        reproduced is dead weight, safe to remove; one it has not is still served by its own
        activation script, and deleting it would take a working environment down. The old root
        is derived from the current layout, so the check follows the layout.
        """
        provisioner = Provisioner(self.board.root, self.board.manifest)
        prefix = provisioner.pixi_for().env_prefix("default")
        held = prefix.relative_to(provisioner.environment_dir()).parts[0]
        old_envs = provisioner.out / held / "envs"
        names = (
            sorted(found.name for found in old_envs.iterdir() if found.is_dir())
            if old_envs.is_dir()
            else []
        )
        if not names:
            return Section(
                section="layout",
                verdict=Verdict.PASS,
                detail=f"only the current environment layout under {provisioner.out.name}",
            )
        superseded = [name for name in names if self._reprovisioned(provisioner, name)]
        legacy = [name for name in names if name not in superseded]
        notes = [f"superseded, safe to remove: {', '.join(superseded)}"] if superseded else []
        if legacy:
            notes.append(
                f"legacy, still served by its own .mainboard/activate-<env>.sh: "
                f"{', '.join(legacy)}; `{_TOOL} install <env>` reprovisions one under the "
                "current layout, after which it joins the superseded ones"
            )
        return Section(
            section="layout",
            verdict=Verdict.WARN,
            detail="; ".join(notes),
            fix=" && ".join(f"rm -rf {old_envs / name}" for name in superseded),
        )

    @staticmethod
    def _reprovisioned(provisioner: Provisioner, name: str) -> bool:
        """Whether `name` already exists again under the current layout.

        A name the manifest no longer declares raises, and is legacy like one not yet reinstalled.
        """
        try:
            return provisioner.pixi_for(name).env_prefix(name).is_dir()
        except MissionError:
            return False

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

        A manifest that will not load is the whole report, since every other section reads it.

        The onboarding records are read here, on this thread: the dispatch cache is one SQLite
        connection only its opening thread may use, and a fleet probe reaching for them inside the
        pool opened it there and left the interpreter closing it from here at exit. The shared
        subsystems are built here for the same reason, which is also why building them is locked.
        """
        manifest = self.manifest()
        if manifest.verdict is Verdict.FAIL:
            return [manifest]
        setups = self.survey.onboarded()
        asked: list[Callable[[], Section]] = [
            *(partial(self.environment, name) for name in self.examined()),
            self.layout,
            self.snapshot,
            self.settling,
            partial(self.fleet, setups),
            partial(self.hosts, setups),
            *(partial(self.gate, name) for name in self.board.manifest.gates),
        ]
        with ThreadPoolExecutor(max_workers=len(asked)) as pool:
            return [manifest, *pool.map(lambda question: question(), asked)]

    def examined(self) -> tuple[str, ...]:
        """Every environment this report covers: the one it was asked about, or all declared.

        A report on `default` alone says nothing about the one a serving host runs, which stays
        invisible until a command asks it for an interpreter.
        """
        return (self.env,) if self.env else ("default", *self.board.manifest.envs)

    def settling(self) -> Section:
        """Whether a periodic pass settles dispatched jobs with no session holding it open.

        A sweep inside the dispatching terminal dies with it, and an outcome must never depend on
        the agent staying alive, so this asks the workspace's own settler (never another
        workspace's timer) whether its pass is installed, armed, and when it last ran. Nothing
        fails: a machine with no periodic pass is one to configure, not a broken workspace.
        """
        found = self.settler.state()
        return Section(
            section="settling",
            verdict=Verdict.PASS if found.active else Verdict.WARN,
            detail=found.detail,
            fix=found.fix,
        )

    def snapshot(self) -> Section:
        """Whether the installed CLI snapshot still answers for the source tree it was built from.

        The one drift a lock never notices, since the snapshot is a uv tool environment beside the
        workspace. A checkout running its own source passes with that word. The refresh reads the
        source files as they are, whatever version control says.
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
        workspace installed rather than whatever interpreter is on PATH.

        timeout: the gate's own deadline in seconds.
        """
        argv = split(command)
        if grammar := [token for token in argv if token in _SHELL_GRAMMAR]:
            return 1, (
                f"`{command}` is run as argv, so its {' '.join(grammar)} reaches the command as "
                f"an ordinary argument rather than as shell grammar; declare it as a [tasks] "
                f"entry and point the gate at that task"
            )
        plan = self.board.plan(env=self.env, container="none")
        result = Provisioner(self.board.root, self.board.manifest).capture(
            argv, plan.env, timeout=timeout
        )
        return result.returncode, result.stdout


def _digest(provisioner: Provisioner, env: str) -> str:
    """`env`'s current manifest digest, empty when the manifest no longer declares it."""
    try:
        return provisioner.compiler_for(env).digest()
    except MissionError:
        return ""
