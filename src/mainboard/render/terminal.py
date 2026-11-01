"""Who reads the output, a person or an agent, decided once and handed to everything that prints.

A person gets rich output: colors, tables and boxes. An agent gets plain rows: no ESC characters,
glyphs, boxes or hints. Results, exit codes and words are the same for both.
"""

import os
import sys
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Self, TextIO

from cyclopts.help import DefaultFormatter, PlainFormatter
from patos import FrozenModel
from rich.console import Console

from ..core.errors import MissionError
from ..core.project import Project
from .enums import Audience

if TYPE_CHECKING:
    from cyclopts.exceptions import CycloptsError


class Streams(FrozenModel):
    """Which standard streams of a process are terminals.

    stdin: standard input is one, so a person there can answer.
    stdout: standard output is one.
    """

    stdin: bool = False
    stdout: bool = False


class Terminal(FrozenModel):
    """The terminal one run prints to, built once in `main` and read by every printer.

    audience: who reads.
    interactive: a person at a terminal who can answer a question, so an agent never hangs.
    marked: an agent harness started the process, whatever `MB_AGENT` says.
    tty: whether standard output is a terminal.
    """

    audience: Audience
    interactive: bool = False
    marked: bool = False
    tty: bool = False

    @classmethod
    def current(cls) -> Terminal:
        """The terminal `main` installed, else the one this process has now."""
        try:
            return _INSTALLED.get()
        except LookupError:
            return cls.here()

    @classmethod
    def here(cls) -> Self:
        """The terminal of this process: its environment and its standard streams."""
        return cls.of(os.environ, Streams(stdin=_is_tty(sys.stdin), stdout=_is_tty(sys.stdout)))

    @classmethod
    def of(cls, environ: Mapping[str, str], streams: Streams) -> Self:
        """The terminal of a process with `environ` and `streams`."""
        audience = Audience.of(environ, tty=streams.stdout)
        return cls(
            audience=audience,
            interactive=audience is Audience.HUMAN and streams.stdin,
            marked=Audience.marked(environ),
            tty=streams.stdout,
        )

    @contextmanager
    def installed(self) -> Generator[Self]:
        """Make this the terminal `current` answers for the block."""
        token = _INSTALLED.set(self)
        try:
            yield self
        finally:
            _INSTALLED.reset(token)

    def _console(self, *, stderr: bool, width: int | None) -> Console:
        """A rich console for this audience: an agent's has no color whatever the environment says.

        A person's follows rich's own rules, so `NO_COLOR` turns color off and `FORCE_COLOR` turns
        it on without a terminal. What is printed is data, so a `[word]` in it is never read as
        markup and nothing in it is highlighted.
        """
        return Console(
            color_system="auto" if self.audience is Audience.HUMAN else None,
            highlight=False,
            markup=False,
            stderr=stderr,
            width=width,
        )

    def console(self, *, width: int | None = None) -> Console:
        """The console on standard output, `width` columns wide, the terminal's own when None."""
        return self._console(stderr=False, width=width)

    def error_console(self, *, width: int | None = None) -> Console:
        """The console on standard error, `width` columns wide, the terminal's own when None."""
        return self._console(stderr=True, width=width)

    def error_formatter(self) -> Callable[[CycloptsError], str] | None:
        """A command-line error as a plain `error:` row for an agent, None for a person's panel."""
        return _row if self.audience is Audience.AGENT else None

    def formatter(self) -> DefaultFormatter | PlainFormatter:
        """The help formatter: rich panels for a person, plain text for an agent."""
        return PlainFormatter() if self.audience is Audience.AGENT else DefaultFormatter()

    def refuse_agents(self, command: str) -> None:
        """Refuse an owner-only command unless a person at a terminal, with no harness, types it.

        `MB_AGENT=0` changes only how output looks: a harness marker, or a standard input that is
        not a terminal, still refuses. This is a guard, not a boundary: a harness that sets no
        marker and hands its commands a terminal still passes, so the agent configs deny these
        commands too.

        command: the words the command is typed as, after the tool's name.

        Raises MissionError when an agent, or no person at a terminal, runs it.
        """
        if self.audience is Audience.AGENT or self.marked or not self.interactive:
            name = Project().name
            raise MissionError(
                f"`{name} {command}` is the owner's act and refuses unless the owner types it at "
                "a terminal, never through an agent; run it from your own terminal"
            )

    def styled(self, *, stderr: bool = False) -> bool:
        """Whether what is printed to standard output (or error) carries color."""
        return self._console(stderr=stderr, width=None).color_system is not None


_INSTALLED: ContextVar[Terminal] = ContextVar("terminal")


def _row(fault: CycloptsError) -> str:
    """A command-line error as the `error:` row every other refusal prints."""
    return f"error: {fault}"


def _is_tty(stream: TextIO) -> bool:
    """Whether `stream` is a terminal; a closed one (`<&-` leaves `sys.stdin` None) is not."""
    try:
        return stream.isatty()
    except AttributeError, ValueError:
        return False
