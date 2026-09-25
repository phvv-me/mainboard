import codecs
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from subprocess import DEVNULL, PIPE
from typing import TYPE_CHECKING, TextIO, cast

from ....core import MissionError
from .result import CommandResult

if TYPE_CHECKING:
    from io import BufferedReader

    from plumbum.commands.base import BaseCommand

# DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, spelled out since POSIX subprocess lacks them.
_WINDOWS_DETACHED_FLAGS = 0x00000200 | 0x00000008


class Process:
    """The one seam that spawns a child process, typed over untyped plumbum.

    The shape is a decision about the terminal: `stream` tees and retains output, `handover`
    gives the child the caller's tty, `output` captures a query whose text is the answer.
    """

    @staticmethod
    def capture(command: BaseCommand, *, timeout: float | None = None) -> CommandResult:
        """Capture a probe whose failure is itself a finding: nothing is replayed or raised.

        On `timeout` plumbum kills the child and raises `ProcessTimedOut`, also a finding.
        """
        returncode, stdout, stderr = command.run(retcode=None, timeout=timeout)
        return CommandResult(int(returncode), str(stdout), str(stderr))

    @staticmethod
    def handover(command: BaseCommand) -> int:
        """Give `command` the caller's own terminal on all three streams, returning its code.

        A shell, REPL or pager puts the tty into raw mode and draws its own screen, which the
        pipes of `stream` would leave unredrawn.
        """
        return command.popen(stdin=None, stdout=None, stderr=None).wait()

    @staticmethod
    def detached(command: BaseCommand) -> None:
        """Start work that must outlive and release the calling executable, in its own session.

        Windows cannot replace a running executable, so a self-update launches Pixi with no
        inherited handles and returns, letting this process exit before Pixi's uv child replaces
        the tool directory.
        """
        if platform.system() == "Windows":
            command.popen(
                stdin=DEVNULL,
                stdout=DEVNULL,
                stderr=DEVNULL,
                creationflags=_WINDOWS_DETACHED_FLAGS,
            )
            return
        command.popen(stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)

    @staticmethod
    def output(command: BaseCommand, operation: str) -> str:
        """Capture a query command, replaying its output before a user-facing failure."""
        result = Process.capture(command)
        if not result.succeeded:
            result.replay()
            raise MissionError(f"`{operation}` failed (see its output above)")
        return result.stdout

    @staticmethod
    def relay(stream: BufferedReader, destination: TextIO, encoding: str) -> str:
        """Copy available pipe bytes to `destination` while retaining decoded text.

        `read1`, unlike `read(size)`, returns after one pipe read, so a long-lived child's short
        protocol messages reach their client immediately.
        """
        decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
        chunks: list[str] = []
        while chunk := stream.read1(4096):
            text = decoder.decode(chunk)
            chunks.append(text)
            Process._display(destination, text)
        if tail := decoder.decode(b"", final=True):
            chunks.append(tail)
            Process._display(destination, tail)
        return "".join(chunks)

    @staticmethod
    def _display(destination: TextIO, text: str) -> None:
        """Write text through the terminal's own encoding without losing captured evidence."""
        if encoding := destination.encoding:
            text = text.encode(encoding, errors="replace").decode(encoding)
        destination.write(text)
        destination.flush()

    @classmethod
    def foreground(cls, command: BaseCommand) -> bool:
        """Run `command` attached to the terminal, returning whether it succeeded."""
        return cls.stream(command).succeeded

    @classmethod
    def passthrough(cls, command: BaseCommand) -> int:
        """Run `command` attached to the terminal, returning its exact exit code."""
        return cls.stream(command).returncode

    @classmethod
    def stream(cls, command: BaseCommand) -> CommandResult:
        """Run `command` while teeing and retaining both output streams."""
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
