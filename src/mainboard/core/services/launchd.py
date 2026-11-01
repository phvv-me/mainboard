# The entries as launchd user agents, the service manager macOS already has.

import os
import plistlib
import shlex
from datetime import UTC, datetime
from os import environ
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import ConfigDict, Field

from ..constants.names import SYSTEM_PATH
from ..constants.units import SECONDS_PER_HOUR
from ..errors import MissionError
from ..execution import host
from ..project import Project
from .base import Every, Settler, Settling, Shell, Unit, complaint, locally

if TYPE_CHECKING:
    from collections.abc import Sequence

# A login to this same Mac, where an agent's command runs: an agent holds no Full Disk Access, so a
# lake on an external volume never answers it (a direct agent's read of the volume hung in open()
# for good, 2026-10-28), while a Remote Login session holds it.
_LOOPBACK = ("/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "localhost")
# The resident agent's login, with a terminal: a bootout ends the ssh client, and only a terminal
# turns that into the SIGHUP the keeper stops on. Without one the keeper never heard of it and a
# child outlived the bootout by 16 s and more (spike, 2026-10-28).
_RESIDENT = (*_LOOPBACK[:-1], "-tt", _LOOPBACK[-1])
# How soon launchd starts a resident agent again after it exits, its own throttle's floor.
_THROTTLE = 5

# A launchd property list, as `plistlib` writes one.
type Plist = dict[str, str | int | bool | list[str] | dict[str, str]]


class _Installed(FrozenModel):
    """What the settler's state reads back out of its installed plist.

    every: the period in seconds, `StartInterval`.
    root: the workspace the pass sweeps, `WorkingDirectory`.
    """

    model_config = ConfigDict(extra="ignore")

    every: int = Field(default=0, alias="StartInterval")
    root: str = Field(default="", alias="WorkingDirectory")


class Launchd:
    """This Mac's launchd, as a user agent is written, loaded, asked about and removed.

    An agent is loaded into the GUI domain while someone is logged in at the console and into
    the background user domain otherwise, which is all a center reached over ssh after a restart
    has.

    agents: the LaunchAgents directory, the user's own when None.
    shell: runs one command and answers with its status and output.
    """

    @property
    def domain(self) -> str:
        """The GUI domain while someone is logged in at the console, else the background one."""
        uid = os.getuid()
        gui = self.shell(("launchctl", "print", f"gui/{uid}"))[0] == 0
        return f"gui/{uid}" if gui else f"user/{uid}"

    @staticmethod
    def available() -> bool:
        """Whether this machine runs launchd, which is what makes a user agent installable."""
        return host.is_macos() and which("launchctl") is not None

    def __init__(self, agents: Path | None = None, shell: Shell = locally) -> None:
        self.agents = agents or Path.home() / "Library" / "LaunchAgents"
        self.shell = shell

    def _bootstrap(self, label: str) -> None:
        """Load `label` from the plist on disk, refused with what launchd said when it will not."""
        plist = self.plist(label)
        status, output = self.shell(("launchctl", "bootstrap", self.domain, str(plist)))
        if status:
            raise MissionError(f"launchd refused {plist.name}: {complaint(output)}")

    def load(self, label: str, agent: Plist) -> None:
        """Write `agent` as `label`'s plist and load it afresh, so a change is the same command."""
        self.agents.mkdir(parents=True, exist_ok=True)
        self.plist(label).write_bytes(plistlib.dumps(agent))
        self.shell(("launchctl", "bootout", self.target(label)))
        self._bootstrap(label)

    def loaded(self, label: str) -> bool:
        """Whether launchd holds `label` in this session's domain."""
        return self.shell(("launchctl", "print", self.target(label)))[0] == 0

    def plist(self, label: str) -> Path:
        return self.agents / f"{label}.plist"

    def require_loopback(self) -> None:
        """Refuse to install unless `ssh localhost` logs in, since agents run commands there."""
        status, output = self.shell((*_LOOPBACK, "true"))
        if status:
            raise MissionError(
                f"`ssh localhost` failed ({complaint(output)}), and an agent runs there to reach "
                "the lake; allow Remote Login with Full Disk Access and a key login to this Mac"
            )

    def start(self, label: str) -> bool:
        """Load `label` from the plist already on disk, answering whether it was not loaded."""
        if self.loaded(label):
            return False
        self._bootstrap(label)
        return True

    def stop(self, label: str) -> bool:
        """Unload `label` and keep its plist, answering whether it was loaded.

        The next login loads it again, as it does every plist in the directory.
        """
        if not self.loaded(label):
            return False
        status, output = self.shell(("launchctl", "bootout", self.target(label)))
        if status:
            raise MissionError(f"launchd refused to unload {label}: {complaint(output)}")
        return True

    def target(self, label: str) -> str:
        return f"{self.domain}/{label}"

    def unload(self, label: str) -> None:
        """Unload `label` and delete its plist, whether or not either is there."""
        self.shell(("launchctl", "bootout", self.target(label)))
        self.plist(label).unlink(missing_ok=True)


class LaunchdAgent(Settler):
    """The pass as a launchd user agent, the periodic runner macOS already has.

    launchd keeps no journal, so what each pass says goes to `run/settler.log` in the workspace's
    state directory. The label carries the same root stamp as the systemd units. The agent runs
    the pass over `ssh localhost` (`_LOOPBACK`), so installing needs that key login to work.
    """

    @property
    def label(self) -> str:
        return f"{self.tool}.monitor.{self.stamp}"

    @property
    def log(self) -> Path:
        return Project().out(self.root) / "run" / "settler.log"

    @property
    def plist(self) -> Path:
        return self.launchd.plist(self.label)

    @staticmethod
    def _spelled(seconds: int) -> str:
        """A period in the largest whole unit, the way `Every.parse` reads it back."""
        for suffix, size in (("h", SECONDS_PER_HOUR), ("m", 60)):
            if seconds and seconds % size == 0:
                return f"{seconds // size}{suffix}"
        return f"{seconds}s"

    @staticmethod
    def available() -> bool:
        return Launchd.available()

    def __init__(self, root: Path, agents: Path | None = None, shell: Shell = locally) -> None:
        """agents: the LaunchAgents directory, the user's own when None.

        shell: runs one command and answers with its status and output.
        """
        super().__init__(root)
        self.launchd = Launchd(agents, shell)

    def _installed(self) -> _Installed:
        """The installed plist read back; the plist must exist."""
        return _Installed.model_validate(plistlib.loads(self.plist.read_bytes()))

    def install(self, every: Every) -> Settling:
        """Write the agent and load it afresh, so changing the period is the same command."""
        command = self._pass("an agent")
        self.launchd.require_loopback()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.launchd.load(
            self.label,
            {
                "Label": self.label,
                "ProgramArguments": [
                    *_LOOPBACK,
                    f"cd {shlex.quote(str(self.root))} && {shlex.join(command)}",
                ],
                "WorkingDirectory": str(self.root),
                "StartInterval": every.seconds,
                "RunAtLoad": True,
                "StandardOutPath": str(self.log),
                "StandardErrorPath": str(self.log),
                "EnvironmentVariables": {"PATH": environ.get("PATH", "")},
            },
        )
        return self.state()

    def remove(self) -> Settling:
        """Unload the agent and delete its plist, whether or not either is there."""
        self.launchd.unload(self.label)
        return self.state()

    def start(self) -> str:
        """Load the installed agent again, which runs a pass at once (`RunAtLoad`)."""
        if not self.plist.is_file():
            raise self._uninstalled()
        every = LaunchdAgent._spelled(self._installed().every)
        if not self.launchd.start(self.label):
            return f"{self.label} is already loaded, one pass every {every}; nothing to start"
        return f"started {self.label}: a pass runs now, then one every {every}"

    def state(self) -> Settling:
        """What the plist on disk and launchd say together."""
        if not self.plist.is_file():
            return self._absent()
        agent = self._installed()
        every = LaunchdAgent._spelled(agent.every)
        active = self.launchd.loaded(self.label)
        try:
            stamp = self.log.stat().st_mtime
        except FileNotFoundError:
            last_run = ""
        else:
            last_run = datetime.fromtimestamp(stamp, UTC).isoformat(timespec="seconds")
        log = f"tail {self.log}"
        state = "loaded" if active else "installed but not loaded"
        success = self.last_success
        return Settling(
            installed=True,
            active=active,
            root=agent.root,
            every=every,
            last_run=last_run,
            last_success=success,
            log=log,
            detail=(
                f"{self.label} {state}, one pass every {every}, read with `{log}`, "
                f"last log activity {last_run or 'never'}, "
                f"last successful pass {success or 'unknown'}"
            ),
            fix="" if active else self._arming,
        )

    def stop(self) -> str:
        """Unload the installed agent and keep its plist, so `start` loads it again."""
        if not self.launchd.stop(self.label):
            return f"{self.label} is already unloaded; nothing to stop"
        return (
            f"stopped {self.label}: no pass runs until `{self._arming}` or the next login, "
            "its plist stays"
        )


class LaunchdUnit(Unit):
    """`mb serve`'s resident agent: launchd starts it at login and again whenever it exits.

    Its command runs over `ssh -tt localhost` (`_RESIDENT`), for the Full Disk Access a direct
    agent lacks and for the terminal whose hangup is how a bootout reaches the keeper. Its output
    is that terminal, so the agent's own log keeps only what ssh says before the keeper starts.
    """

    @property
    def name(self) -> str:
        return f"{self.tool}.serve.{self.stamp}"

    @staticmethod
    def available() -> bool:
        return Launchd.available()

    def __init__(self, root: Path, agents: Path | None = None, shell: Shell = locally) -> None:
        """agents: the LaunchAgents directory, the user's own when None.

        shell: runs one command and answers with its status and output.
        """
        super().__init__(root)
        self.launchd = Launchd(agents, shell)

    def install(self, command: Sequence[str], *, stop_seconds: int, log: Path) -> None:
        self.launchd.require_loopback()
        log.parent.mkdir(parents=True, exist_ok=True)
        self.launchd.load(
            self.name,
            {
                "Label": self.name,
                "ProgramArguments": [
                    *_RESIDENT,
                    f"cd {shlex.quote(str(self.root))} && exec {shlex.join(command)}",
                ],
                "WorkingDirectory": str(self.root),
                "RunAtLoad": True,
                "KeepAlive": True,
                "ThrottleInterval": _THROTTLE,
                "ExitTimeOut": stop_seconds,
                "StandardOutPath": os.devnull,
                "StandardErrorPath": str(log),
                "EnvironmentVariables": {"PATH": SYSTEM_PATH},
            },
        )

    def installed(self) -> bool:
        return self.launchd.plist(self.name).is_file()

    def loaded(self) -> bool:
        return self.launchd.loaded(self.name)

    def remove(self) -> None:
        self.launchd.unload(self.name)
