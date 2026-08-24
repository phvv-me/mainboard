import codecs
import sys
from concurrent.futures import ThreadPoolExecutor
from subprocess import PIPE
from typing import TYPE_CHECKING, TextIO, cast

from ....core import MissionError
from .result import CommandResult

if TYPE_CHECKING:
    from io import BufferedReader

    from plumbum.commands.base import BaseCommand


class Process:
    """The one seam that spawns a child process, in each of the shapes a backend needs.

    plumbum is untyped, so every launch funnels through here and comes back as a typed
    :class:`CommandResult`. Which shape a caller picks is a decision about the terminal:
    :meth:`stream` tees output while retaining a copy for diagnostics, :meth:`handover` gives
    the child the caller's own tty because an interactive program draws its own screen, and
    :meth:`output` captures a query whose text is the answer.
    """

    @staticmethod
    def capture(command: BaseCommand, *, timeout: float | None = None) -> CommandResult:
        """Capture a query whose failure is itself an answer, under an optional deadline.

        The shape a probe wants rather than the shape a step wants. A lock that holds no such
        environment, or a tool this workspace never installed, is something the caller reports
        as a finding, so nothing is replayed to the terminal and nothing is raised. ``timeout``
        bounds a probe that would otherwise hang, and plumbum kills the child and raises
        ``ProcessTimedOut`` when it expires, which the caller reports as its own finding too.
        """
        returncode, stdout, stderr = command.run(retcode=None, timeout=timeout)
        return CommandResult(int(returncode), str(stdout), str(stderr))

    @staticmethod
    def handover(command: BaseCommand) -> int:
        """Give ``command`` the caller's own terminal on all three streams, returning its code.

        An interactive program draws its own screen: a shell, a REPL or a pager puts the
        terminal into raw mode and expects every keystroke and every escape sequence to travel
        over the tty. :meth:`stream` gives it pipes instead so it can retain a copy of the
        output, which leaves such a program typing into a terminal nothing ever redraws. Those
        callers come here and trade the retained copy for a working terminal.
        """
        return command.popen(stdin=None, stdout=None, stderr=None).wait()

    @staticmethod
    def output(command: BaseCommand, operation: str) -> str:
        """Capture a query command, replaying its output before a user-facing failure."""
        returncode, stdout, stderr = command.run(retcode=None)
        result = CommandResult(int(returncode), str(stdout), str(stderr))
        if not result.succeeded:
            result.replay()
            raise MissionError(f"`{operation}` failed (see its output above)")
        return result.stdout

    @staticmethod
    def relay(stream: BufferedReader, destination: TextIO, encoding: str) -> str:
        """Copy available pipe bytes to ``destination`` while retaining decoded text.

        ``BufferedReader.read(size)`` may wait for all ``size`` bytes while a long-lived child
        remains open. ``read1`` returns after one underlying pipe read instead, so short
        protocol messages reach their client immediately.
        """
        decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
        chunks: list[str] = []
        while chunk := stream.read1(4096):
            text = decoder.decode(chunk)
            chunks.append(text)
            destination.write(text)
            destination.flush()
        if tail := decoder.decode(b"", final=True):
            chunks.append(tail)
            destination.write(tail)
            destination.flush()
        return "".join(chunks)

    @classmethod
    def foreground(cls, command: BaseCommand) -> bool:
        """Run ``command`` attached to the terminal, returning whether it succeeded."""
        return cls.stream(command).succeeded

    @classmethod
    def passthrough(cls, command: BaseCommand) -> int:
        """Run ``command`` attached to the terminal, returning its exact exit code.

        Unlike :meth:`foreground` (a success bool), this preserves the code so a transparent
        caller exits with whatever the wrapped command exited instead of collapsing every
        failure to ``1``.
        """
        return cls.stream(command).returncode

    @classmethod
    def stream(cls, command: BaseCommand) -> CommandResult:
        """Run ``command`` while teeing and retaining both output streams."""
        process = command.popen(stdin=None, stdout=PIPE, stderr=PIPE)
        # `PIPE` on both streams guarantees Popen hands back real pipes, never `None`.
        stdout_pipe = cast("BufferedReader", process.stdout)
        stderr_pipe = cast("BufferedReader", process.stderr)
        encoding = sys.getfilesystemencoding()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mainboard-compile") as pool:
            stdout = pool.submit(cls.relay, stdout_pipe, sys.stdout, encoding)
            stderr = pool.submit(cls.relay, stderr_pipe, sys.stderr, encoding)
            returncode = process.wait()
        return CommandResult(returncode, stdout.result(), stderr.result())
