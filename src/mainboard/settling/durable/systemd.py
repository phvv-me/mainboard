# The pass as a systemd user timer, the periodic runner a Linux workstation already has.

import re
from getpass import getuser
from os import environ
from pathlib import Path
from shutil import which

from ...core.errors import MissionError
from ...core.execution import host
from .base import Every, Settler, Settling, Shell, complaint, locally

# The timer properties a state read asks the user manager for.
_SHOWN = ("--property=ActiveState", "--property=LastTriggerUSec")


class SystemdUser(Settler):
    """The pass as a systemd user timer, the periodic runner a Linux workstation already has.

    A user timer needs no root and outlives every terminal, but a user manager is torn down when
    its last session ends unless that user lingers, so the linger state rides in every answer
    rather than waiting for a reboot to lose a night of settling. The unit names carry an
    eight-hex-digit stamp of this settler's root, which keeps several workspaces' timers apart.
    """

    @property
    def service(self) -> Path:
        """The unit that runs one pass."""
        return self.units / f"{self.tool}-monitor-{self.stamp}.service"

    @property
    def timer(self) -> Path:
        """The unit that runs the service on a period."""
        return self.units / f"{self.tool}-monitor-{self.stamp}.timer"

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
        """Whether this machine runs systemd, which is what makes a user timer installable."""
        return host.is_linux() and which("systemctl") is not None

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

    def _lingering(self) -> bool:
        """Whether this user lingers, which is what carries a user timer across a reboot."""
        status, output = self.shell(("loginctl", "show-user", getuser(), "--property=Linger"))
        return not status and output.strip().endswith("=yes")

    def _repair(self, *, active: bool, lingering: bool) -> str:
        """The one command that makes an installed timer settle jobs for good.

        Arming comes before lingering, since a timer nothing armed does not run at all while an
        armed one that does not linger runs until the next reboot.
        """
        if not active:
            return f"systemctl --user enable --now {self.timer.name}"
        return "" if lingering else f"loginctl enable-linger {getuser()}"

    def _reported(self) -> tuple[bool, str]:
        """Whether the user manager reports the timer active, and when its service last ran."""
        shown = self._shown()
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

    def _shown(self) -> dict[str, str]:
        """The timer properties the user manager reports, empty when it will not answer."""
        status, output = self.shell(("systemctl", "--user", "show", self.timer.name, *_SHOWN))
        if status:
            return {}
        pairs = (line.partition("=") for line in output.splitlines() if "=" in line)
        return {key: value.strip() for key, _, value in pairs}

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
        """Write both units, reload the user manager and arm the timer.

        Idempotent: the units are rewritten from the root and `every` and the enable is repeatable,
        so changing the period is the same command.
        """
        self.units.mkdir(parents=True, exist_ok=True)
        self.service.write_text(self._service(), encoding="utf-8", newline="\n")
        self.timer.write_text(self._timer(every), encoding="utf-8", newline="\n")
        self.shell(("systemctl", "--user", "daemon-reload"))
        status, output = self.shell(("systemctl", "--user", "enable", "--now", self.timer.name))
        if status:
            raise MissionError(f"systemd refused {self.timer.name}: {complaint(output)}")
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
            return self._absent()
        active, last_run = self._reported()
        every = self._setting(self.timer, "OnUnitActiveSec")
        log = f"journalctl --user -u {self.service.name}" if self.service.is_file() else ""
        lingering = self._lingering()
        state = "active" if active else "installed but not armed"
        linger = "" if lingering else ", and a reboot stops it until this user lingers"
        success = self.last_success
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
            fix=self._repair(active=active, lingering=lingering),
        )
