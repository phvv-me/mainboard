# The durable form of `mainboard monitor`: the periodic settling pass installed into the
# machine's own service manager, not a session's terminal. A cron an agent starts dies with that
# agent, and thirty five PBS jobs owed to it died unsettled, so the pass belongs to the machine.
# Linux answers with a user systemd timer, which needs no root and survives every terminal.
# Another platform is a refusal naming itself until an implementation is registered below.

import platform
import re
from abc import ABC, abstractmethod
from getpass import getuser
from hashlib import blake2b
from os import environ
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from patos import FrozenModel, Strategy
from plumbum import CommandNotFound, local
from plumbum.commands.processes import ProcessTimedOut

from .core.errors import MissionError
from .core.project import Project

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# The tool this workspace answers to, so no unit file or message below spells the binary's name.
_TOOL = Project().name
# The one pass a period runs, the same line a person types at a terminal.
_PASS = ("monitor", "--json")
# Where a pass appends what it settled, beside the rest of the generated state.
_LOG = "monitor.log"
# The period a message suggests when nothing is installed, the one the campaign cron ran at.
_SUGGESTED = "20m"
# How long a service manager may take to answer before it is read as saying nothing.
_DEADLINE = 10.0
# The timer properties a state read asks the user manager for.
_SHOWN = ("--property=ActiveState", "--property=LastTriggerUSec")

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
                f"`{_TOOL} monitor --every 0` removes the pass"
            )
        return cls(seconds=int(found[1]) * _UNITS[found[2]], written=said)


class Settling(FrozenModel):
    """What durable settling this machine holds right now.

    installed: whether the periodic pass exists on this machine at all.
    active: whether the service manager has it armed and running it.
    root: the workspace the installed pass sweeps, empty when nothing is installed.
    every: the period between passes as the installed unit spells it, empty when none is.
    last_run: when a pass last ran, as the service manager reports it, empty when none ever did.
    log: where a pass appends what it settled, empty when nothing is installed.
    detail: the one line behind the answer.
    fix: the single command that gets a durable pass running, empty when one already does.
    """

    installed: bool = False
    active: bool = False
    root: str = ""
    every: str = ""
    last_run: str = ""
    log: str = ""
    detail: str
    fix: str = ""


class Settler(ABC):
    """One machine's way of running the settling pass on a period, and of saying whether it does.

    The seam a second platform fills: a launchd or Task Scheduler implementation subclasses here
    and registers in `SETTLERS` ahead of the null answer, and no caller changes. A settler belongs
    to one workspace, fixed at construction, so one machine can settle several at once.
    """

    def __init__(self, root: Path) -> None:
        """root: the workspace whose dispatched jobs this settler settles."""
        self.root = root

    @abstractmethod
    def install(self, every: Every) -> Settling:
        """Install and arm the pass at `every`, answering with what now runs."""

    @abstractmethod
    def remove(self) -> Settling:
        """Remove the periodic pass, answering with what is left."""

    @abstractmethod
    def state(self) -> Settling:
        """What periodic settling this machine holds right now."""


class SystemdUser(Settler):
    """The pass as a systemd user timer, the periodic runner a Linux workstation already has.

    A user timer needs no root and outlives every terminal, but a user manager is torn down when
    its last session ends unless that user lingers, so the linger state rides in every answer
    rather than waiting for a reboot to lose a night of settling. The unit names carry an
    eight-hex-digit stamp of this settler's root, which keeps several workspaces' timers apart.
    """

    def __init__(self, root: Path, units: Path | None = None, shell: Shell = locally) -> None:
        """units: the user unit directory, the XDG one when None.

        shell: runs one command and answers with its status and output.
        """
        super().__init__(root)
        self.units = (
            units
            or Path(environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user"
        )
        self.shell = shell

    @property
    def service(self) -> Path:
        """The unit that runs one pass."""
        return self.units / f"{_TOOL}-monitor-{self._stamp}.service"

    @property
    def timer(self) -> Path:
        """The unit that runs the service on a period."""
        return self.units / f"{_TOOL}-monitor-{self._stamp}.timer"

    @property
    def _stamp(self) -> str:
        """Eight hex digits of blake2b over the resolved root."""
        return blake2b(str(self.root.resolve()).encode(), digest_size=4).hexdigest()

    @staticmethod
    def available() -> bool:
        """Whether this machine runs systemd, which is what makes a user timer installable."""
        return platform.system() == "Linux" and which("systemctl") is not None

    def install(self, every: Every) -> Settling:
        """Write both units, reload the user manager and arm the timer.

        Idempotent: the units are rewritten from the root and `every` and the enable is repeatable,
        so changing the period is the same command.
        """
        log = self.root / Project().out_dir / _LOG
        log.parent.mkdir(parents=True, exist_ok=True)
        self.units.mkdir(parents=True, exist_ok=True)
        self.service.write_text(self._service(log), encoding="utf-8")
        self.timer.write_text(self._timer(every), encoding="utf-8")
        self.shell(("systemctl", "--user", "daemon-reload"))
        status, output = self.shell(("systemctl", "--user", "enable", "--now", self.timer.name))
        if status:
            raise MissionError(f"systemd refused {self.timer.name}: {self._complaint(output)}")
        return self.state()

    def remove(self) -> Settling:
        """Disarm the timer and delete both units, on a machine that has them or one that does not.

        The disable is asked before the files go, because a unit systemd no longer has a file
        for is one it cannot be told to stop.
        """
        self.shell(("systemctl", "--user", "disable", "--now", self.timer.name))
        self.timer.unlink(missing_ok=True)
        self.service.unlink(missing_ok=True)
        self.shell(("systemctl", "--user", "daemon-reload"))
        return self.state()

    def state(self) -> Settling:
        """What the installed units and the user manager say together.

        The period, log and workspace are read back out of the units on disk, so the row says
        what actually runs, including a timer an earlier version of this tool wrote.
        """
        if not self.timer.is_file():
            return Settling(
                detail=(
                    "no periodic pass installed, so a dispatched job settles only while a "
                    "session sweeps it"
                ),
                fix=f"{_TOOL} monitor --every {_SUGGESTED}",
            )
        shown = self._shown()
        every = self._setting(self.timer, "OnUnitActiveSec")
        log = self._setting(self.service, "StandardOutput").removeprefix("append:")
        active = shown.get("ActiveState") == "active"
        triggered = shown.get("LastTriggerUSec", "")
        last_run = "" if triggered in ("", "n/a") else triggered
        lingering = self._lingering()
        state = "active" if active else "installed but not armed"
        linger = "" if lingering else ", and a reboot stops it until this user lingers"
        return Settling(
            installed=True,
            active=active,
            root=self._setting(self.service, "WorkingDirectory"),
            every=every,
            last_run=last_run,
            log=log,
            detail=(
                f"{self.timer.name} {state}, one pass every {every} into {log}, "
                f"last run {last_run or 'never'}{linger}"
            ),
            fix=self._repair(active=active, lingering=lingering),
        )

    def _repair(self, *, active: bool, lingering: bool) -> str:
        """The one command that makes an installed timer settle jobs for good.

        Arming comes before lingering, since a timer nothing armed does not run at all while an
        armed one that does not linger runs until the next reboot.
        """
        if not active:
            return f"systemctl --user enable --now {self.timer.name}"
        return "" if lingering else f"loginctl enable-linger {getuser()}"

    def _lingering(self) -> bool:
        """Whether this user lingers, which is what carries a user timer across a reboot."""
        status, output = self.shell(("loginctl", "show-user", getuser(), "--property=Linger"))
        return not status and output.strip().endswith("=yes")

    def _shown(self) -> dict[str, str]:
        """The timer properties the user manager reports, empty when it will not answer."""
        status, output = self.shell(("systemctl", "--user", "show", self.timer.name, *_SHOWN))
        if status:
            return {}
        pairs = (line.partition("=") for line in output.splitlines() if "=" in line)
        return {key: value.strip() for key, _, value in pairs}

    def _service(self, log: Path) -> str:
        """The unit for one pass, run from this settler's root, appending what it says to `log`."""
        found = which(_TOOL)
        if found is None:
            raise MissionError(
                f"no {_TOOL} on PATH for a timer to run; install the snapshot first"
            )
        return "\n".join(
            (
                "[Unit]",
                f"Description={_TOOL} durable job settling for {self.root}",
                "",
                "[Service]",
                "Type=oneshot",
                f"WorkingDirectory={self.root}",
                f"ExecStart={found} {' '.join(_PASS)}",
                f"StandardOutput=append:{log}",
                f"StandardError=append:{log}",
                "",
            )
        )

    def _timer(self, every: Every) -> str:
        """The unit text that runs the service one period after arming and after every pass."""
        return "\n".join(
            (
                "[Unit]",
                f"Description={_TOOL} durable job settling every {every.written}",
                "",
                "[Timer]",
                f"OnActiveSec={every.written}",
                f"OnUnitActiveSec={every.written}",
                "AccuracySec=30s",
                f"Unit={self.service.name}",
                "",
                "[Install]",
                "WantedBy=timers.target",
                "",
            )
        )

    @staticmethod
    def _setting(unit: Path, key: str) -> str:
        """One `Key=value` line out of an installed unit, empty when the unit has no such line.

        A unit deleted by hand leaves the row describing what is left rather than taking down a
        report whose point is saying what is wrong.
        """
        try:
            text = unit.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        found = re.search(rf"^{key}=(.+)$", text, re.MULTILINE)
        return found[1].strip() if found else ""

    @staticmethod
    def _complaint(output: str) -> str:
        """The last thing a service manager said, which is where it puts its refusal."""
        spoken = [line.strip() for line in output.splitlines() if line.strip()]
        return spoken[-1] if spoken else "it said nothing"


class Unsupported(Settler):
    """The answer on a machine whose service manager this tool cannot install a pass into.

    Registered last, so it answers for exactly the platforms nothing above claimed. It refuses
    in one sentence naming the platform, since a pass a person believes runs and does not is
    worse than no pass at all.
    """

    def install(self, every: Every) -> Settling:
        del every
        raise MissionError(self._refusal)

    def remove(self) -> Settling:
        raise MissionError(self._refusal)

    def state(self) -> Settling:
        return Settling(
            detail=(
                f"{platform.system()} has no periodic runner {_TOOL} installs into, so a "
                "dispatched job settles only while a session sweeps it"
            )
        )

    @property
    def _refusal(self) -> str:
        """The one sentence a platform with no implementation here is refused with."""
        return (
            f"{platform.system()} has no service manager {_TOOL} can install a periodic pass "
            f"into; sweep with `{_TOOL} monitor` from a scheduler this machine already runs"
        )


# Walked in order, the first implementation this machine can run winning and the null answer
# catching the rest. Registered by class, since which one wins never depends on the workspace.
SETTLERS: Strategy[type[Settler]] = Strategy("settler")
SETTLERS.register("systemd", SystemdUser)
SETTLERS.register("none", Unsupported)


def settler(root: Path) -> Settler:
    """The periodic runner this machine offers `root`, the refusing one where it offers none."""
    return SETTLERS.first_available()(root)


def schedule(root: Path, every: str) -> Settling:
    """Install the durable settling pass for `root` at `every` (`20m`), or remove it at `0`."""
    period = Every.parse(every)
    machine = settler(root)
    return machine.install(period) if period.seconds else machine.remove()
