import platform
from contextlib import nullcontext
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from plumbum import local

from ....core import MissionError
from .process import Process

if TYPE_CHECKING:
    from collections.abc import Callable

    from plumbum.commands.base import BaseCommand


# A Windows conda package ships `npm` (a POSIX script `CreateProcess` refuses) beside `npm.cmd`,
# and plumbum would take the script. Only `PATHEXT` spellings run, `.cmd`/`.bat` via cmd.exe.
_SCRIPT_LAUNCHERS = {".cmd", ".bat"}


def windows_launcher(name: str) -> BaseCommand:
    """The command that runs `name` on Windows: its `PATHEXT` spelling, under cmd.exe if a script.

    name: `npm`, or a spelling already carrying its extension (`C:/Python/python.exe`), which is
        looked up as it stands.
    """
    extensions = local.env.get("PATHEXT", ".EXE;.CMD;.BAT").lower().split(";")
    suffix = Path(name).suffix.lower()
    spellings = [name] if suffix and suffix in extensions else [name + ext for ext in extensions]
    for directory in local.env.path:
        for spelling in spellings:
            candidate = Path(str(directory)) / spelling
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() in _SCRIPT_LAUNCHERS:
                return local["cmd.exe"]["/d", "/c", str(candidate)]
            return local[str(candidate)]
    raise MissionError(f"{name} is not on PATH")


class Tool:
    """A package-manager backend that builds one command line; `Process` spawns it.

    Subclasses set `name` and override `scope` (args pinning the workspace) and `available`.
    """

    name: str = ""

    def __call__(self, verb: str, *args: str, **flags: bool | str | None) -> None:
        """Run in the foreground, a no-op if unavailable, `MissionError` on failure.

        `flags` become CLI args (see `flags`) placed before the positional `args`.
        """
        if not self.available():
            return
        if not self.within_cwd(Process.foreground, verb, *args, **flags):
            raise MissionError(f"`{self.name} {verb}` failed (see its output above)")

    @cached_property
    def command(self) -> BaseCommand:
        """The resolved local command, looked up lazily.

        The name is required only here, since a backend running through another tool overrides
        `__call__` and names no binary.
        """
        if not self.name:
            raise MissionError(f"{type(self).__name__} names no command of its own to run")
        if platform.system() == "Windows":
            return windows_launcher(self.name)
        return local[self.name]

    @staticmethod
    def flags(**options: bool | str | None) -> list[str]:
        """Keyword options as CLI args: `feature_x=True` -> `--feature-x`, `k=v` -> `--k v`.

        `False`, `None` and `""` are dropped.
        """
        out: list[str] = []
        for key, value in options.items():
            if value in (None, False, ""):
                continue
            out.append(f"--{key.replace('_', '-')}")
            if value is not True:
                out.append(str(value))
        return out

    def available(self) -> bool:
        return True

    def cwd(self) -> Path | None:
        """Directory to run in, for tools that target a workspace by location."""
        return None

    def exit_code(self, verb: str, *args: str, **flags: bool | str | None) -> int:
        """Run in the foreground, returning the exact exit code (`0` if unavailable)."""
        if not self.available():
            return 0
        return self.within_cwd(Process.passthrough, verb, *args, **flags)

    def defer(self, verb: str, *args: str, **flags: bool | str | None) -> None:
        """Start a command that must outlive this process, doing nothing when unavailable."""
        if self.available():
            self.within_cwd(Process.detached, verb, *args, **flags)

    def scope(self) -> tuple[str, ...]:
        """Args after the verb pinning the command to this workspace."""
        return ()

    def within_cwd[T](
        self,
        action: Callable[[BaseCommand], T],
        verb: str,
        *args: str,
        **flags: bool | str | None,
    ) -> T:
        """Build `verb + scope + flags + args` and run `action` on it inside `cwd`."""
        command = self.command[(verb, *self.scope(), *self.flags(**flags), *args)]
        directory = self.cwd()
        with nullcontext() if directory is None else local.cwd(str(directory)):
            return action(command)
