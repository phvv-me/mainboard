# The entries as systemd user units, the service manager a Linux workstation already has.

import re
import shlex
from getpass import getuser
from os import environ
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from ..errors import MissionError
from ..execution import host
from .base import Every, Settler, Settling, Shell, Unit, complaint, locally

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The timer properties a state read asks the user manager for.
_SHOWN = ("--property=ActiveState", "--property=LastTriggerUSec")
# How soon systemd starts the resident unit again after it exits, as launchd's throttle does.
_RESTART_SECONDS = 5


class Systemd:
    """This user's systemd manager, as a unit is written, armed, asked about and removed.

    A user unit needs no root and outlives every terminal, but the user manager is torn down when
    its user's last session ends unless that user lingers, which is why `lingering` is asked.

    units: the user unit directory, the XDG one when None.
    shell: runs one command and answers with its status and output.
    """

    @staticmethod
    def available() -> bool:
        """Whether this machine runs systemd, which is what makes a user unit installable."""
        return host.is_linux() and which("systemctl") is not None

    def __init__(self, units: Path | None = None, shell: Shell = locally) -> None:
        self.units = (
            units
            or Path(environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user"
        )
        self.shell = shell

    def active(self, name: str) -> bool:
        """Whether the user manager reports unit `name` active."""
        return self.show(name, "--property=ActiveState").get("ActiveState") == "active"

    def disable(self, name: str, *also: str) -> None:
        """Stop and disarm `name`, delete its file and `also`'s, and reload, whatever exists.

        The disable is asked before the files go, because a unit systemd no longer has a file
        for is one it cannot be told to stop.
        """
        self.shell(("systemctl", "--user", "disable", "--now", name))
        for unit in (name, *also):
            (self.units / unit).unlink(missing_ok=True)
        self.shell(("systemctl", "--user", "daemon-reload"))

    def enable(self, name: str, units: Mapping[str, str]) -> None:
        """Write `units` (file name to text), reload, and enable and start `name`.

        Idempotent: the files are rewritten and the enable is repeatable, so a change is the
        same command.
        """
        self.units.mkdir(parents=True, exist_ok=True)
        for unit, text in units.items():
            (self.units / unit).write_text(text, encoding="utf-8", newline="\n")
        self.shell(("systemctl", "--user", "daemon-reload"))
        status, output = self.shell(("systemctl", "--user", "enable", "--now", name))
        if status:
            raise MissionError(f"systemd refused {name}: {complaint(output)}")

    def lingering(self) -> bool:
        """Whether this user lingers, which is what carries a user unit across a reboot."""
        status, output = self.shell(("loginctl", "show-user", getuser(), "--property=Linger"))
        return not status and output.strip().endswith("=yes")

    def show(self, name: str, *properties: str) -> dict[str, str]:
        """The `properties` of unit `name` the user manager reports, empty without an answer."""
        status, output = self.shell(("systemctl", "--user", "show", name, *properties))
        if status:
            return {}
        pairs = (line.partition("=") for line in output.splitlines() if "=" in line)
        return {key: value.strip() for key, _, value in pairs}

    def stop(self, name: str, *also: str) -> None:
        """Stop `name` and `also` and leave their files, so a later `enable` arms them again."""
        status, output = self.shell(("systemctl", "--user", "stop", name, *also))
        if status:
            raise MissionError(f"systemd refused to stop {name}: {complaint(output)}")


class SystemdUser(Settler):
    """The pass as a systemd user timer, the periodic runner a Linux workstation already has.

    The linger state rides in every answer rather than waiting for a reboot to lose a night of
    settling. The unit names carry an eight-hex-digit stamp of this settler's root, which keeps
    several workspaces' timers apart.
    """

    @property
    def service(self) -> Path:
        """The unit that runs one pass."""
        return self.systemd.units / f"{self.tool}-monitor-{self.stamp}.service"

    @property
    def timer(self) -> Path:
        """The unit that runs the service on a period."""
        return self.systemd.units / f"{self.tool}-monitor-{self.stamp}.timer"

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
    def available() -> bool:
        return Systemd.available()

    def __init__(self, root: Path, units: Path | None = None, shell: Shell = locally) -> None:
        """units: the user unit directory, the XDG one when None.

        shell: runs one command and answers with its status and output.
        """
        super().__init__(root)
        self.systemd = Systemd(units, shell)

    def _reported(self) -> tuple[bool, str]:
        """Whether the user manager reports the timer active, and when its service last ran."""
        shown = self.systemd.show(self.timer.name, *_SHOWN)
        triggered = shown.get("LastTriggerUSec", "")
        last_run = "" if triggered in ("", "n/a") else triggered
        return shown.get("ActiveState") == "active", last_run

    def _service(self) -> str:
        """The unit for one pass, run from this settler's root, what it says kept by the journal.

        What a pass settles is recorded in the workspace lake; its own chatter is the service
        manager's to keep, where `journalctl` reads it, rather than one more file to grow.
        """
        return "\n".join(
            (
                "[Unit]",
                f"Description={self.tool} durable job settling for {self.root}",
                "",
                "[Service]",
                "Type=oneshot",
                f"WorkingDirectory={self.root}",
                f"ExecStart={' '.join(self._pass('a timer'))}",
                "",
            )
        )

    def _timer(self, every: Every) -> str:
        """The unit text that runs the service one period after arming and after every pass."""
        return "\n".join(
            (
                "[Unit]",
                f"Description={self.tool} durable job settling every {every.written}",
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

    def install(self, every: Every) -> Settling:
        """Write both units, reload the user manager and arm the timer, the same at any period."""
        self.systemd.enable(
            self.timer.name,
            {self.service.name: self._service(), self.timer.name: self._timer(every)},
        )
        return self.state()

    def remove(self) -> Settling:
        """Disarm the timer and delete both units, whether or not this machine has them."""
        self.systemd.disable(self.timer.name, self.service.name)
        return self.state()

    def start(self) -> str:
        """Enable and start the installed timer, which also re-arms it for the next login."""
        if not self.timer.is_file():
            raise self._uninstalled()
        every = self._setting(self.timer, "OnUnitActiveSec")
        if self.systemd.active(self.timer.name):
            return f"{self.timer.name} is already active, one pass every {every}; nothing to start"
        self.systemd.enable(self.timer.name, {})
        return f"started {self.timer.name}: one pass every {every}, the first one period from now"

    def state(self) -> Settling:
        """What the installed units and the user manager say together.

        The period, log and workspace are read back out of the units on disk, so the row says
        what actually runs, including a timer an earlier version of this tool wrote.
        """
        if not self.timer.is_file():
            return self._absent()
        active, last_run = self._reported()
        every = self._setting(self.timer, "OnUnitActiveSec")
        log = f"journalctl --user -u {self.service.name}" if self.service.is_file() else ""
        lingering = self.systemd.lingering()
        state = "active" if active else "installed but not armed"
        linger = "" if lingering else ", and a reboot stops it until this user lingers"
        success = self.last_success
        # The one command that makes an installed timer settle jobs for good. Arming comes before
        # lingering: a timer nothing armed does not run at all, while an armed one that does not
        # linger runs until the next reboot.
        if not active:
            fix = self._arming
        else:
            fix = "" if lingering else f"loginctl enable-linger {getuser()}"
        return Settling(
            installed=True,
            active=active,
            root=self._setting(self.service, "WorkingDirectory"),
            every=every,
            last_run=last_run,
            last_success=success,
            log=log,
            detail=(
                f"{self.timer.name} {state}, one pass every {every}, read with `{log}`, "
                f"last service trigger {last_run or 'never'}, "
                f"last successful pass {success or 'unknown'}{linger}"
            ),
            fix=fix,
        )

    def stop(self) -> str:
        """Stop the timer and any pass it is running, leaving both units installed."""
        if not self.systemd.active(self.timer.name):
            return f"{self.timer.name} is already stopped; nothing to stop"
        self.systemd.stop(self.timer.name, self.service.name)
        return (
            f"stopped {self.timer.name}: no pass runs until `{self._arming}` or the next login, "
            "its units stay"
        )


class SystemdUnit(Unit):
    """`mb serve`'s resident user service: systemd starts it at boot and again whenever it exits.

    The unit's cgroup is the orphan sweep launchd lacks, since a stop or crash of the unit kills
    every process in it, and a user unit reads any path its user can, so no ssh hop is needed.
    Boot start needs this user to linger, which root alone can grant.
    """

    @property
    def name(self) -> str:
        return f"{self.tool}-serve-{self.stamp}.service"

    @staticmethod
    def _escaped(command: Sequence[str]) -> str:
        """`command` as an `ExecStart=` line: shell quoting, then systemd's `%` and `$` doubled."""
        return shlex.join(command).replace("%", "%%").replace("$", "$$")

    @staticmethod
    def available() -> bool:
        return Systemd.available()

    def __init__(self, root: Path, units: Path | None = None, shell: Shell = locally) -> None:
        """units: the user unit directory, the XDG one when None.

        shell: runs one command and answers with its status and output.
        """
        super().__init__(root)
        self.systemd = Systemd(units, shell)

    def _text(self, command: Sequence[str], *, stop_seconds: int, log: Path) -> str:
        """The unit text: restarted always, stopped within `stop_seconds`, umask 077."""
        return "\n".join(
            (
                "[Unit]",
                f"Description={self.tool} serve for {self.root}",
                "",
                "[Service]",
                "Type=simple",
                f"WorkingDirectory={self.root}",
                f"ExecStart={SystemdUnit._escaped(command)}",
                "Restart=always",
                f"RestartSec={_RESTART_SECONDS}",
                f"TimeoutStopSec={stop_seconds}",
                "UMask=0077",
                "StandardOutput=null",
                f"StandardError=append:{log}",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            )
        )

    def install(self, command: Sequence[str], *, stop_seconds: int, log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        self.systemd.enable(
            self.name, {self.name: self._text(command, stop_seconds=stop_seconds, log=log)}
        )

    def installed(self) -> bool:
        return (self.systemd.units / self.name).is_file()

    def loaded(self) -> bool:
        return self.systemd.active(self.name)

    def remove(self) -> None:
        self.systemd.disable(self.name)
