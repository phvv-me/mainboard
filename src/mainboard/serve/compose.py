"""The process-compose YAML `mb serve` runs, generated from the registry and never edited by hand.

No secret is ever written into it. process-compose itself starts from an empty environment holding
only `Compose.allowed`, every child inherits that, and an entry's own `environment` is refused when
it names a variable the workspace `.env` defines: that file holds the keys that rent, push and pay
(18 of its 28 names on 2026-10-28), and the shell activation exports all of it, which is why no
child starts inside an activation.

Nothing in the file is expanded or rendered: environment expansion is off for the whole file and
templates for each child, so the commands run exactly as they are written here.
"""

from getpass import getuser
from pathlib import Path
from shlex import join
from shutil import which
from tempfile import gettempdir
from typing import TYPE_CHECKING, Self

import yaml

from ..core.constants.names import SYSTEM_PATH
from ..core.errors import MissionError
from .constants import TOOL, Service
from .enums import Kind

if TYPE_CHECKING:
    from collections.abc import Sequence, Set

    from ..engines.compile.ecosystems.archive import Binaries
    from ..manifest.render.interpolate import Json
    from .keeper import Layout

# A resident child is started again this many seconds after it exits.
_BACKOFF = 2
# Each child's log rotates past this many megabytes, keeping this many compressed copies.
_ROTATED_MB = 4
_BACKUPS = 3
# What the unit's stop allows beyond the children's own timeouts, spent one after another.
_MARGIN = 10


class Programs:
    """Where each child's program is, by absolute path, since a child's PATH is the system's.

    binaries: the workspace's executables: its pinned installs, then its environment, then PATH.
    tool: the installed tool a command names as `TOOL`, None where none is installed.
    """

    @classmethod
    def here(cls, binaries: Binaries) -> Self:
        """This machine's programs, the tool found on PATH as the settler's pass finds it."""
        found = which(TOOL)
        return cls(binaries, tool=Path(found) if found else None)

    def __init__(self, binaries: Binaries, *, tool: Path | None) -> None:
        self.binaries = binaries
        self.tool = tool

    def locate(self, program: str) -> Path:
        """The absolute path of `program`: as written when absolute, else found here or refused."""
        if Path(program).is_absolute():
            return Path(program)
        if program != TOOL:
            return self.binaries.check(program).path
        if self.tool is None:
            raise MissionError(f"no {TOOL} on PATH for a served child to run; install the tool")
        return self.tool


class Compose:
    """The YAML of one served workspace: every child not off, under the allowlist.

    services: the declared children; an off one is left out of the file.
    root: the workspace, every child's working directory.
    layout: where the YAML and the children's logs go.
    programs: resolves each command's program to its absolute path.
    dotenv: the names the workspace `.env` defines, refused in any child's environment.
    """

    @property
    def running(self) -> list[Service]:
        """The children the YAML holds: every one not off."""
        return [service for service in self.services if not service.off]

    @staticmethod
    def allowed() -> dict[str, str]:
        """The whole environment process-compose starts with, which every child inherits."""
        user = getuser()
        return {
            "HOME": str(Path.home()),
            "LANG": "en_US.UTF-8",
            "LOGNAME": user,
            "PATH": SYSTEM_PATH,
            "TMPDIR": gettempdir(),
            "USER": user,
        }

    def __init__(
        self,
        services: Sequence[Service],
        *,
        root: Path,
        layout: Layout,
        programs: Programs,
        dotenv: Set[str],
    ) -> None:
        self.services = services
        self.root = root
        self.layout = layout
        self.programs = programs
        self.dotenv = dotenv

    def _child(self, service: Service) -> dict[str, Json]:
        """One child's YAML: its command by absolute path, its log, its stop and its schedule.

        A pass runs under `proc timeout`, since the supervisor bounds no run.
        """
        program, *arguments = service.command
        command = [str(self.programs.locate(program)), *arguments]
        if service.kind is Kind.PERIODIC:
            bound = [str(self.programs.locate(TOOL)), "proc", "timeout", str(service.timeout)]
            command = [*bound, *command]
        child: dict[str, Json] = {
            "command": f"exec {join(command)}",
            "is_template_disabled": True,
            "working_dir": str(self.root),
            "environment": [f"{name}={value}" for name, value in self._environment(service)],
            "log_location": str(self.layout.logs / f"{service.name}.log"),
            "log_configuration": {
                "rotation": {"max_size_mb": _ROTATED_MB, "max_backups": _BACKUPS, "compress": True}
            },
            "shutdown": {"signal": 15, "timeout_seconds": service.stop},
        }
        if service.kind is Kind.PERIODIC:
            child["schedule"] = {
                "interval": service.every,
                "run_on_start": True,
                "max_concurrent": 1,
            }
        else:
            child["availability"] = {"restart": "always", "backoff_seconds": _BACKOFF}
        return child

    def _environment(self, service: Service) -> tuple[tuple[str, str], ...]:
        """`service`'s own variables, refused for a `.env` name or one shadowing the allowlist."""
        names = {name for name, _ in service.environment}
        if leaked := sorted(names & self.dotenv):
            raise MissionError(
                f"{service.name} declares {', '.join(leaked)}, which the workspace .env defines; "
                "a served child is given the path of a secret's 0600 file, never the secret"
            )
        if shadowed := sorted(names & Compose.allowed().keys()):
            raise MissionError(f"{service.name} redefines {', '.join(shadowed)} of the allowlist")
        return service.environment

    def document(self) -> dict[str, Json]:
        """The whole YAML as a mapping, strict so process-compose refuses a key it doesn't know."""
        return {
            "version": "0.5",
            "is_strict": True,
            "disable_env_expansion": True,
            "log_level": "info",
            "processes": {service.name: self._child(service) for service in self.running},
        }

    def stop_seconds(self) -> int:
        """How long the whole tree may take to stop: the children stop one after another."""
        return sum(service.stop for service in self.running) + _MARGIN

    def write(self) -> None:
        """Write the YAML where the supervisor reads it, 0600 like everything in the state."""
        self.layout.logs.mkdir(mode=0o700, exist_ok=True)
        text = yaml.safe_dump(self.document(), sort_keys=False)
        self.layout.compose.write_text(text, encoding="utf-8", newline="\n")
        self.layout.compose.chmod(0o600)
