# The durable form of `mb job list`: the periodic settling pass installed into the
# machine's own service manager, not a session's terminal. A cron an agent starts dies with that
# agent, and thirty five PBS jobs owed to it died unsettled, so the pass belongs to the machine.
# Linux answers with a user systemd timer and macOS with a launchd user agent, neither needing
# root and both surviving every terminal. Another platform is a refusal naming itself.

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

from ...core.errors import MissionError
from ...core.project import Project
from ...dispatch.state.monitor import last_settlement

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


class Settler(ABC):
    """One machine's way of running the settling pass on a period, and of saying whether it does.

    The seam a second platform fills: a launchd or Task Scheduler implementation subclasses here
    and registers in `SETTLERS` ahead of the null answer, and no caller changes. A settler belongs
    to one workspace, fixed at construction, so one machine can settle several at once.
    """

    # The tool's legacy name, so no unit file below spells the binary's name. Units keep it on
    # purpose: an armed timer is then found and replaced under the name it was installed with,
    # never doubled beside a renamed one, and the script it runs exists in every release.
    tool: ClassVar[str] = Project().package

    @abstractmethod
    def install(self, every: Every) -> Settling:
        """Install and arm the pass at `every`, answering with what now runs."""

    @abstractmethod
    def remove(self) -> Settling:
        """Remove the periodic pass, answering with what is left."""

    @abstractmethod
    def state(self) -> Settling:
        """What periodic settling this machine holds right now."""

    @property
    def last_success(self) -> str:
        """When a pass last ran to its end, which only the pass itself can say."""
        return last_settlement(Project().out(self.root) / "run")

    @property
    def stamp(self) -> str:
        """Eight hex digits of blake2b over the resolved root, keeping workspaces' passes apart."""
        return blake2b(str(self.root.resolve()).encode(), digest_size=4).hexdigest()

    def __init__(self, root: Path) -> None:
        """root: the workspace whose dispatched jobs this settler settles."""
        self.root = root

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


class Unsupported(Settler):
    """The answer on a machine whose service manager this tool cannot install a pass into.

    Registered last, so it answers for exactly the platforms nothing above claimed. It refuses
    in one sentence naming the platform, since a pass a person believes runs and does not is
    worse than no pass at all.
    """

    @property
    def _refusal(self) -> str:
        """The one sentence a platform with no implementation here is refused with."""
        return (
            f"{platform.system()} has no service manager {self.tool} can install a periodic pass "
            f"into; sweep with `{Project().name} job list` from a scheduler this machine "
            "already runs"
        )

    def install(self, every: Every) -> Settling:
        del every
        raise MissionError(self._refusal)

    def remove(self) -> Settling:
        raise MissionError(self._refusal)

    def state(self) -> Settling:
        return Settling(
            detail=(
                f"{platform.system()} has no periodic runner {self.tool} installs into, so a "
                "dispatched job settles only while a session sweeps it"
            )
        )


def complaint(output: str) -> str:
    """The last thing a service manager said, which is where it puts its refusal."""
    spoken = [line.strip() for line in output.splitlines() if line.strip()]
    return spoken[-1] if spoken else "it said nothing"
