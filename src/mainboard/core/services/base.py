# What this tool installs into the machine's own service manager, never a session's terminal. A
# cron an agent starts dies with that agent, and thirty five PBS jobs owed to it died unsettled, so
# both entries belong to the machine: the periodic settling pass of `mb job list`, and the one
# resident unit `mb serve` keeps its supervisor in. Linux answers with systemd user units and macOS
# with launchd user agents, neither needing root and both surviving every terminal. Another
# platform is a refusal naming itself.

import platform
import re
from abc import ABC, abstractmethod
from hashlib import blake2b
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, ClassVar

from patos import FrozenModel
from plumbum import CommandNotFound, local
from plumbum.commands.processes import ProcessTimedOut

from ...dispatch.state.monitor import last_settlement
from ..errors import MissionError
from ..project import Project

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# The one pass a period runs, the same line a person types at a terminal.
_PASS = ("job", "list", "--json")
# The period a message suggests when nothing is installed, the one the campaign cron ran at.
_SUGGESTED = "20m"
# How long a service manager may take to answer before it is read as saying nothing.
_DEADLINE = 10.0
# `20m`, `1h`, `90s`: a whole number of one unit, the way systemd itself writes a period.
_WRITTEN = re.compile(r"(\d+)([smh])")
_UNITS = {"s": 1, "m": 60, "h": 3600}

# Runs one command on this machine and answers with its exit status and everything it said.
type Shell = Callable[[Sequence[str]], tuple[int, str]]


def locally(command: Sequence[str], deadline: float = _DEADLINE) -> tuple[int, str]:
    """Run `command` here under a deadline, its output joined, answering rather than raising.

    A service manager that hangs or is not installed says nothing, and every caller already
    reads silence as "no periodic pass runs", so neither is worth an exception.

    deadline: seconds the command may take before it counts as no answer.
    """
    program, *arguments = command
    try:
        status, out, err = local[program][arguments].run(retcode=None, timeout=deadline)
    except CommandNotFound:
        return 1, f"{program} is not installed here"
    except ProcessTimedOut:
        return 1, f"{program} did not answer within {deadline:.0f}s"
    return status, out + err


class Every(FrozenModel):
    """How often the durable pass runs, a zero period meaning it does not run at all.

    seconds: the period in whole seconds, zero for a pass that is removed rather than installed.
    written: the spelling the caller used, which is also what the installed unit carries.
    """

    seconds: int
    written: str

    @classmethod
    def parse(cls, written: str) -> Every:
        """`20m`, `1h`, `90s` or a bare `0`, refusing anything else by naming the spellings."""
        said = written.strip()
        if said == "0":
            return cls(seconds=0, written=said)
        found = _WRITTEN.fullmatch(said)
        if found is None:
            raise MissionError(
                f"a period is written like 20m, 1h or 90s, not {written!r}; "
                f"`{Project().name} job list --every 0` removes the pass"
            )
        return cls(seconds=int(found[1]) * _UNITS[found[2]], written=said)


class Settling(FrozenModel):
    """What durable settling this machine holds right now.

    installed: whether the periodic pass exists on this machine at all.
    active: whether the service manager has it armed and running it.
    root: the workspace the installed pass sweeps, empty when nothing is installed.
    every: the period between passes as the installed unit spells it, empty when none is.
    last_run: when the service manager last started a pass or a pass last wrote its log, which
        passes that find the lock held keep refreshing, empty when it never did.
    last_success: when a pass last ran to its end, empty when none ever did.
    log: the command that reads what passes said, empty when nothing is installed.
    detail: the one line behind the answer.
    fix: the single command that gets a durable pass running, empty when one already does.
    """

    installed: bool = False
    active: bool = False
    root: str = ""
    every: str = ""
    last_run: str = ""
    last_success: str = ""
    log: str = ""
    detail: str
    fix: str = ""


class Managed:
    """An entry this tool installs into the machine's service manager for one workspace.

    It belongs to one workspace, fixed at construction, so one machine can hold several at once.
    """

    # The tool's legacy name, so no unit file below spells the binary's name. Units keep it on
    # purpose: an armed timer is then found and replaced under the name it was installed with,
    # never doubled beside a renamed one, and the script it runs exists in every release.
    tool: ClassVar[str] = Project().package

    @property
    def stamp(self) -> str:
        """Eight hex digits of blake2b over the resolved root, keeping workspaces apart."""
        return blake2b(str(self.root.resolve()).encode(), digest_size=4).hexdigest()

    def __init__(self, root: Path) -> None:
        """root: the workspace this entry serves."""
        self.root = root

    def _refusal(self, what: str) -> str:
        """The one sentence a platform with no implementation here is refused with.

        what: what this machine cannot hold, as the sentence names it.
        """
        return f"{platform.system()} has no service manager {self.tool} can install {what} into"


class Settler(Managed, ABC):
    """One machine's way of running the settling pass on a period, and of saying whether it does.

    The seam a second platform fills: an implementation subclasses here and registers in
    `_SETTLERS` ahead of the null answer, and no caller changes.
    """

    @abstractmethod
    def install(self, every: Every) -> Settling:
        """Install and arm the pass at `every`, answering with what now runs."""

    @abstractmethod
    def remove(self) -> Settling:
        """Remove the periodic pass, answering with what is left."""

    @abstractmethod
    def start(self) -> str:
        """Arm the installed pass again, answering what that did.

        Idempotent: a pass already armed is left alone and said to be. A refusal when no pass is
        installed, since there is nothing to arm.
        """

    @abstractmethod
    def state(self) -> Settling:
        """What periodic settling this machine holds right now."""

    @abstractmethod
    def stop(self) -> str:
        """Disarm the pass but keep it installed, so `start` arms it again; answers what it did.

        Idempotent: a pass already stopped, or never installed, is said to be. The service
        manager may arm it again by itself at the next login or boot.
        """

    @property
    def _arming(self) -> str:
        """The command that arms an installed pass again."""
        return f"{Project().name} job monitor start"

    @property
    def last_success(self) -> str:
        """When a pass last ran to its end, which only the pass itself can say."""
        return last_settlement(Project().out(self.root) / "run")

    def _absent(self) -> Settling:
        """The answer while no periodic pass is installed."""
        return Settling(
            detail=(
                "no periodic pass installed, so a dispatched job settles only while a session "
                "sweeps it"
            ),
            fix=f"{Project().name} job list --every {_SUGGESTED}",
        )

    def _pass(self, runner: str) -> list[str]:
        """The pass's command line, the installed tool by its absolute path.

        runner: what will run it, named in the refusal when no tool is on PATH.
        """
        found = which(self.tool)
        if found is None:
            raise MissionError(
                f"no {self.tool} on PATH for {runner} to run; install the tool first"
            )
        return [found, *_PASS]

    def _uninstalled(self) -> MissionError:
        """The refusal of arming a pass that was never installed, naming the command that does."""
        return MissionError(
            f"no periodic pass is installed to start; `{self._absent().fix}` installs one"
        )


class Unit(Managed, ABC):
    """The one resident entry `mb serve` keeps in this machine's service manager.

    The manager starts it at login or boot and starts it again whenever it exits. What it runs is
    handed in whole (the keeper of `mainboard.serve`), so this seam knows nothing of what is
    served, and a second platform fills it by subclassing here and registering in `_RESIDENTS`.
    """

    @abstractmethod
    def install(self, command: Sequence[str], *, stop_seconds: int, log: Path) -> None:
        """Install `command` as the unit and start it, replacing any copy already installed.

        command: the argv the unit runs from the workspace root, absolute paths throughout.
        stop_seconds: how long a stop may take before the manager kills what is left.
        log: where the manager writes what the unit says before it opens its own log.
        """

    @abstractmethod
    def installed(self) -> bool:
        """Whether the unit's file is on disk, so the manager starts it at the next login."""

    @abstractmethod
    def loaded(self) -> bool:
        """Whether the manager holds the unit right now, running or about to run it again."""

    @abstractmethod
    def remove(self) -> None:
        """Stop the unit and delete its file, whether or not either is there.

        The manager answers before what the unit ran has finished stopping, so a caller that
        needs everything gone waits for that itself.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name the service manager knows the unit by, for messages and for its own tools."""


class Unsupported(Settler):
    """The answer on a machine whose service manager this tool cannot install a pass into.

    Registered last, so it answers for exactly the platforms nothing above claimed. It refuses
    in one sentence naming the platform, since a pass a person believes runs and does not is
    worse than no pass at all.
    """

    @property
    def _said(self) -> str:
        """The refusal, with the sweep a person can run instead."""
        return (
            f"{self._refusal('a periodic pass')}; sweep with `{Project().name} job list` from a "
            "scheduler this machine already runs"
        )

    def install(self, every: Every) -> Settling:
        del every
        raise MissionError(self._said)

    def remove(self) -> Settling:
        raise MissionError(self._said)

    def start(self) -> str:
        raise MissionError(self._said)

    def state(self) -> Settling:
        return Settling(
            detail=(
                f"{platform.system()} has no periodic runner {self.tool} installs into, so a "
                "dispatched job settles only while a session sweeps it"
            )
        )

    def stop(self) -> str:
        raise MissionError(self._said)


class UnsupportedUnit(Unit):
    """The resident unit on a machine whose service manager this tool cannot install into.

    Registered last, like `Unsupported`. Nothing is ever installed, so removing is nothing to do,
    and installing refuses in one sentence naming the platform.
    """

    @property
    def name(self) -> str:
        return f"{self.tool} serve"

    def install(self, command: Sequence[str], *, stop_seconds: int, log: Path) -> None:
        del command, stop_seconds, log
        raise MissionError(f"{self._refusal('a resident unit')}, so `serve` cannot run here")

    def installed(self) -> bool:
        return False

    def loaded(self) -> bool:
        return False

    def remove(self) -> None:
        """Nothing was installed, so there is nothing to remove."""


def complaint(output: str) -> str:
    """The last thing a service manager said, which is where it puts its refusal."""
    spoken = [line.strip() for line in output.splitlines() if line.strip()]
    return spoken[-1] if spoken else "it said nothing"
