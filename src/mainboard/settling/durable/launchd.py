# The pass as a launchd user agent, the periodic runner macOS already has.

import os
import plistlib
import shlex
from datetime import UTC, datetime
from os import environ
from pathlib import Path
from shutil import which

from ...core.errors import MissionError
from ...core.execution import host
from ...core.project import Project
from .base import Every, Settler, Settling, Shell, complaint, locally

# A login to this same Mac, where a launchd agent's pass runs: an agent holds no Full Disk Access,
# so a lake on an external volume never answers it, while a Remote Login session holds it.
_LOOPBACK = ("/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "localhost")


class LaunchdAgent(Settler):
    """The pass as a launchd user agent, the periodic runner macOS already has.

    The agent is loaded into the GUI domain while someone is logged in at the console and into the
    background user domain otherwise, which is all a center reached over ssh after a restart has.
    launchd keeps no journal, so what each pass says goes to `run/settler.log` in the workspace's
    state directory. The label carries the same root stamp as the systemd units. The agent runs
    the pass over `ssh localhost` (`_LOOPBACK`), so installing needs that key login to work.
    """

    @property
    def _domain(self) -> str:
        """The GUI domain while someone is logged in at the console, else the background one."""
        uid = os.getuid()
        gui = self.shell(("launchctl", "print", f"gui/{uid}"))[0] == 0
        return f"gui/{uid}" if gui else f"user/{uid}"

    @property
    def _target(self) -> str:
        return f"{self._domain}/{self.label}"

    @property
    def label(self) -> str:
        return f"{self.tool}.monitor.{self.stamp}"

    @property
    def log(self) -> Path:
        return Project().out(self.root) / "run" / "settler.log"

    @property
    def plist(self) -> Path:
        return self.agents / f"{self.label}.plist"

    @staticmethod
    def _spelled(seconds: int) -> str:
        """A period in the largest whole unit, the way `Every.parse` reads it back."""
        for suffix, size in (("h", 3600), ("m", 60)):
            if seconds and seconds % size == 0:
                return f"{seconds // size}{suffix}"
        return f"{seconds}s"

    @staticmethod
    def available() -> bool:
        """Whether this machine runs launchd, which is what makes a user agent installable."""
        return host.is_macos() and which("launchctl") is not None

    def __init__(self, root: Path, agents: Path | None = None, shell: Shell = locally) -> None:
        """agents: the LaunchAgents directory, the user's own when None.

        shell: runs one command and answers with its status and output.
        """
        super().__init__(root)
        self.agents = agents or Path.home() / "Library" / "LaunchAgents"
        self.shell = shell

    def install(self, every: Every) -> Settling:
        """Write the agent and load it afresh, so changing the period is the same command."""
        command = self._pass("an agent")
        status, output = self.shell((*_LOOPBACK, "true"))
        if status:
            raise MissionError(
                f"`ssh localhost` failed ({complaint(output)}), and the pass runs there to reach "
                "the lake; allow Remote Login with Full Disk Access and a key login to this Mac"
            )
        self.agents.mkdir(parents=True, exist_ok=True)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        agent = {
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
        }
        self.plist.write_bytes(plistlib.dumps(agent))
        self.shell(("launchctl", "bootout", self._target))
        status, output = self.shell(("launchctl", "bootstrap", self._domain, str(self.plist)))
        if status:
            refusal = complaint(output)
            raise MissionError(f"launchd refused {self.plist.name}: {refusal}")
        return self.state()

    def remove(self) -> Settling:
        """Unload the agent and delete its plist, whether or not either is there."""
        self.shell(("launchctl", "bootout", self._target))
        self.plist.unlink(missing_ok=True)
        return self.state()

    def state(self) -> Settling:
        """What the plist on disk and launchd say together."""
        if not self.plist.is_file():
            return self._absent()
        agent = plistlib.loads(self.plist.read_bytes())
        every = LaunchdAgent._spelled(int(agent.get("StartInterval", 0)))
        active = self.shell(("launchctl", "print", self._target))[0] == 0
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
            root=str(agent.get("WorkingDirectory", "")),
            every=every,
            last_run=last_run,
            last_success=success,
            log=log,
            detail=(
                f"{self.label} {state}, one pass every {every}, read with `{log}`, "
                f"last log activity {last_run or 'never'}, "
                f"last successful pass {success or 'unknown'}"
            ),
            fix="" if active else f"launchctl bootstrap {self._domain} {self.plist}",
        )
