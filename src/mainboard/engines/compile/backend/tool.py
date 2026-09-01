from functools import cached_property
from typing import TYPE_CHECKING

from plumbum import local

from ....core import MissionError
from .process import Process

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from plumbum.commands.base import BaseCommand


class Tool:
    """A package-manager backend: build one command line and run it in the foreground.

    Subclasses set ``name`` to the binary they run and override ``scope`` (args that pin the
    command to the workspace) and ``available`` (a guard). Building the argv is this class's
    job and spawning the child is :class:`Process`'s, so a backend only ever describes what to
    run.
    """

    name: str = ""

    def __call__(self, verb: str, *args: str, **flags: bool | str | None) -> None:
        """Run the backend in the foreground. A no-op if unavailable, `MissionError` on failure.

        Keyword ``flags`` translate to CLI args (`resolve=True` -> `--resolve`, `feature=env`
        -> `--feature env`), inserted before the positional ``args``. Raising on failure keeps
        a failed solve or install from being reported as green success by the caller.
        """
        if not self.available():
            return
        if not self.within_cwd(Process.foreground, verb, *args, **flags):
            raise MissionError(f"`{self.name} {verb}` failed (see its output above)")

    @cached_property
    def command(self) -> BaseCommand:
        """The resolved local command, looked up lazily so importing doesn't require it.

        A backend that runs through another tool overrides :meth:`__call__` and never names a
        binary, so the name is required here, at the one boundary that needs it, rather than of
        every subclass.
        """
        if not self.name:
            raise MissionError(f"{type(self).__name__} names no command of its own to run")
        return local[self.name]

    @staticmethod
    def flags(**options: bool | str | None) -> list[str]:
        """Turn keyword options into CLI args (`_`->`-`), dropping `False`/`None`/`""`.

        A `True` becomes a bare `--flag`, and any other value becomes `--flag value`.
        """
        out: list[str] = []
        for key, value in options.items():
            if value is None or value is False or value == "":
                continue
            out.append(f"--{key.replace('_', '-')}")
            if value is not True:
                out.append(str(value))
        return out

    def available(self) -> bool:
        """Whether the command should run at all."""
        return True

    def cwd(self) -> Path | None:
        """Directory to run in, for tools that target a workspace by location, not a flag."""
        return None

    def exit_code(self, verb: str, *args: str, **flags: bool | str | None) -> int:
        """Run in the foreground and return the command's exact exit code (``0`` if unavailable).

        The code-preserving sibling of :meth:`__call__`, for a transparent passthrough where a
        failing command must exit non-zero.
        """
        if not self.available():
            return 0
        return self.within_cwd(Process.passthrough, verb, *args, **flags)

    def defer(self, verb: str, *args: str, **flags: bool | str | None) -> None:
        """Start a command that must outlive this process, doing nothing when unavailable."""
        if self.available():
            self.within_cwd(Process.detached, verb, *args, **flags)

    def scope(self) -> tuple[str, ...]:
        """Args injected after the verb to pin the command to this workspace (default none)."""
        return ()

    def within_cwd[T](
        self,
        action: Callable[[BaseCommand], T],
        verb: str,
        *args: str,
        **flags: bool | str | None,
    ) -> T:
        """Build ``verb + scope + flags + args`` and run ``action`` on it inside ``cwd``."""
        command = self.command[(verb, *self.scope(), *self.flags(**flags), *args)]
        directory = self.cwd()
        if directory is None:
            return action(command)
        with local.cwd(str(directory)):
            return action(command)
