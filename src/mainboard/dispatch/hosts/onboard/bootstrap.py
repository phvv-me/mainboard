# Putting the tool and the workspace environment on a machine: uv's CPython first, then the tool
# through the first install route the machine supports, then the environment from the shipped
# lock.

import hashlib
import shlex
from importlib.metadata import requires
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from patos import Resolution, Strategy, StrategyError

from ....core.errors import MissionError
from ....core.project import Project
from ....engines.compile.pixi.lockfile import Lockfile
from ....engines.compile.pixi.policy import LockPolicy
from ....state.lake import Lake
from .constants import SOURCE, TOOL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...transport.shells import HostShell


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


def _specifier(floor: str) -> str:
    """The declared version as a requirement operator carries it, empty when nothing is declared.

    A manifest writes `">=0.4.8"` with its operator and `"0.4.8"` without, which means that
    exact version.

    floor: the version the workspace declares for the tool, `*` or empty for any.
    """
    if floor in {"", "*"}:
        return ""
    return floor if floor[0] in "<>=!~" else f"=={floor}"


class Existing(Installer):
    """The route that installs nothing, because the machine already runs what was asked for.

    Only offered to a workspace that vendors no source: one that does always reinstalls from it,
    since shipping the source is how the host runs the tool the manifest was written against.
    """

    @staticmethod
    def _satisfied_by(found: str, *, floor: str) -> bool:
        """Whether the version `found` (empty when the machine runs no tool) meets `floor`.

        No declared version is satisfied by any tool. An unparsable version is not satisfied, so
        the install happens rather than being skipped on a string nobody understood.
        """
        if not found:
            return False
        wanted = _specifier(floor)
        if not wanted:
            return True
        try:
            return SpecifierSet(wanted).contains(Version(found), prereleases=True)
        except InvalidSpecifier, InvalidVersion:
            return False

    def __init__(self, shell: HostShell, *, floor: str, python: str) -> None:
        """python: the interpreter request the tool installs on (`>=3.14.0,<3.15`)."""
        super().__init__(shell, probe=shell.dialect.has(TOOL), command=shell.dialect.noop)
        self.floor = floor
        self.minor = ".".join(python.removeprefix(">=").split(".")[:2])

    def available(self) -> bool:
        """Whether uv's installed tool satisfies both the version and the solved interpreter."""
        if not Existing._satisfied_by(reported(self.shell, TOOL), floor=self.floor):
            return False
        probe = (
            "import sys; "
            "sys.exit(f'{sys.version_info.major}.{sys.version_info.minor}' != "
            f"{self.minor!r})"
        )
        return self.shell.ok(f'"$(uv tool dir)/{TOOL}/bin/python" -c {shlex.quote(probe)}')


def reported(shell: HostShell, program: str) -> str:
    """The version `program --version` reports on the machine (`X.Y.Z`), empty without one."""
    try:
        return shell.run(f"{program} --version").strip().split()[-1]
    except MissionError, IndexError:
        return ""


class Bootstrap:
    """Puts this workspace's tool and environment onto a machine that already answers ssh.

    Put uv and the CPython it manages there before anything ships, install the tool onto that
    interpreter through the first route the machine supports, then have it compile the synced
    manifest and install the environment from the lock this workspace already solved. A
    declared host and a machine rented for one job both drive this one class.

    workspace: the local workspace whose default lock chooses the tool's interpreter.
    policy: what the machine's install may do to the shipped lock: `locked` refuses one this
        manifest did not solve, the default, and `update` lets the machine solve for itself.
    floor: the version the workspace declares for the tool, used only when the workspace vendors
        no source and the tool therefore comes from an index.
    extras: the tool's optional extras to install with it.
    """

    @property
    def env(self) -> str:
        """The environment being provisioned, the plan's own."""
        return self.shell.plan.env

    @property
    def requirements(self) -> str:
        """A digest of what the tool installs at, so a sync reinstalls it when that moved.

        It covers the interpreter, the declared dependencies and the lock's releases of those.
        """
        declared = requires(TOOL) or []
        direct = {canonicalize_name(Requirement(line).name) for line in declared}
        held = [f"{name}=={pin}" for name, pin in sorted(self.pins.items()) if name in direct]
        text = "\n".join([self.python_request, *sorted(declared), *held])
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def __init__(
        self,
        shell: HostShell,
        *,
        workspace: Path,
        policy: LockPolicy = LockPolicy.LOCKED,
        floor: str = "",
        extras: Sequence[str] = (),
    ) -> None:
        self.shell = shell
        lockfile = Lockfile(workspace)
        self.python_request = lockfile.python_request(shell.plan.profile.platform)
        self.pins = lockfile.pins(shell.plan.profile.platform)
        self.out = Path(shell.root) / Project().out_dir(workspace)
        self.constraints = (self.out / "run" / "tool-constraints.txt").as_posix()
        self.policy = policy
        self.floor = floor
        self.extras = tuple(extras)

    def environment(self) -> None:
        """Have the machine's own tool compile the synced manifest and install `env` from it.

        The machine is told which declared profile describes it, so its activation script carries
        that profile's module stack rather than this machine's.
        """
        host = self.shell.plan.host
        flag = f" {self.policy.flag}" if self.policy.flag else ""
        self.shell.run(
            f"{TOOL} install {shlex.quote(self.env)}{flag} --profile {shlex.quote(host)}"
        )
        if not self.shell.ok(self.shell.provisioned):
            raise MissionError(
                f"{host!r} has no {self.shell.proof} after installing {self.env!r}; "
                "the environment was not provisioned"
            )

    def lake(self) -> None:
        """Bring the host's staging lake to the DuckLake spec its new DuckDB writes.

        Every reader there runs that DuckDB now, the one condition a one-way migration waits for,
        and until it moves an older catalog refuses every attach (DuckDB 2.0.0 refused 1.1-dev1,
        2026-10-22). A host without a staging lake is left alone.
        """
        catalog = Lake(root=Path(self.shell.root), home=self.out).catalog
        if self.shell.ok(self.shell.dialect.is_file(catalog.as_posix())):
            self.shell.run(f"{TOOL} lake upgrade")

    def python(self) -> None:
        """Put uv and the uv-managed CPython the tool runs on onto the machine, first of all.

        Mainboard runs there on that interpreter until the workspace environment exists
        (`Dialect.python`), the agent shipping the mirror included, and the tool is installed
        onto it. The root is made first, since every line stands in it. A machine with neither
        uv nor curl to fetch it is refused: nothing stands in, never its own `python3` and never
        a pip into its user site.
        """
        shell = self.shell
        shell.place()
        probe, fetch = shell.dialect.uv_bootstrap
        if not shell.ok(shell.dialect.has("uv")):
            if not shell.ok(probe):
                raise MissionError(
                    f"{shell.plan.host!r} has neither uv nor curl to fetch it, and {TOOL} runs "
                    "on no other Python there; install uv on it, then set it up again"
                )
            shell.run(fetch)
        # A uv older than a Python release knows only that release's candidates (miyabi-g's
        # 0.8.11 offered 3.14.0rc2). One its installer placed updates itself; one copied in
        # refuses, and the installer replaces it.
        elif not shell.ok("uv self update --quiet") and shell.ok(probe):
            shell.run(fetch)
        # `--upgrade`: a request an older patch satisfies would otherwise keep it, and the tool's
        # venv follows uv's minor link to whatever patch is newest.
        shell.run(f"uv python install --upgrade --no-bin {shlex.quote(self.python_request)}")

    def routes(self, *, vendored: bool) -> Strategy[Installer]:
        """The ordered install routes for the machine, best first.

        uv installs every one, onto the uv-managed CPython `python` put there, so the tool never
        runs on the host's own interpreter and nothing lands in its user site.

        The family offered is a fact about the workspace, not the machine. One that vendors the
        tool's source installs from it, so a host never runs a tool older than the manifest it
        compiles. One that consumes the tool from an index installs the version it declares,
        after first offering the machine's own tool when that already satisfies it.

        vendored: whether the tool's source is on the machine.
        """
        strategy: Strategy[Installer] = Strategy(f"{TOOL} installer")
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        has_uv = self.shell.dialect.has("uv")
        # `--reinstall` because uv reuses a cached build of an unchanged version number.
        install = (
            "uv tool install --force --reinstall --managed-python "
            f"--python {shlex.quote(self.python_request)} "
            f"--constraints {shlex.quote(self.constraints)}"
        )
        if vendored:
            editable = f"{install} --editable {shlex.quote(f'{SOURCE}{extras}')}"
            strategy.register("uv", Installer(self.shell, probe=has_uv, command=editable))
            return strategy
        indexed = f"{install} {shlex.quote(f'{TOOL}{extras}{_specifier(self.floor)}')}"
        strategy.register(
            "present", Existing(self.shell, floor=self.floor, python=self.python_request)
        )
        strategy.register("uv-index", Installer(self.shell, probe=has_uv, command=indexed))
        return strategy

    def tool(self) -> Resolution[Installer]:
        """Install the tool through the first route the machine supports, keeping the rejections.

        Whether the mirror carries the tool's source is asked of the machine, not assumed: an
        index-consuming workspace ships no such directory, and when every route tested for it,
        all refused and the refusal blamed the host's tooling (miyabi-g, 2026-09-05). So a
        machine no route reaches fails here, before anything assumes the tool, and the refusal
        leads with which family was tried and why, the rejection log behind it.

        The tool's dependencies are held to the shipped lock's releases (`Lockfile.pins`), so
        the tool never re-solves to a release its jobs do not run: an open `duckdb>=2.0.0` takes
        whatever DuckDB is newest the day a host is set up, and a DuckLake catalog or a Quack
        peer refuses a DuckDB other than the one that wrote it (2026-10-22).
        """
        vendored = self.shell.ok(self.shell.dialect.is_directory(SOURCE))
        self.shell.write(
            self.constraints,
            text="".join(f"{name}=={pin}\n" for name, pin in sorted(self.pins.items())),
        )
        routes = self.routes(vendored=vendored)
        try:
            resolution = routes.cascade()
        except StrategyError as refused:
            where = (
                f"installing from the source this workspace vendors at {SOURCE}"
                if vendored
                else f"installing {TOOL}{_specifier(self.floor) or ' (no version declared)'} "
                "from an index, since this workspace vendors no source"
            )
            raise MissionError(
                f"cannot install {TOOL} on {self.shell.plan.host!r} by {where}: {refused}"
            ) from None
        routes.select(resolution.winner).install()
        return resolution
