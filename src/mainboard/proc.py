# The process chores no portable binary does on every center: `mainboard proc`.
#
# The default environment standardizes the everyday commands, but three common operations have
# no binary that behaves the same under PowerShell, zsh and bash. `timeout` is GNU coreutils and
# absent from macOS and Windows; killing a process with everything it started is `pkill -P` or
# `kill -- -pgid` on one system and `taskkill /T` on another; and waiting for a file or a port is
# a shell loop around `sleep`, `test` and `nc` that no Windows shell runs. Each is one method here,
# built on the process-tree primitive the dispatch transport already relies on.

import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from .core.errors import MissionError
from .dispatch.transport import terminate_process_tree

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# The exit status GNU `timeout` reports for a command it had to stop, kept so scripts that
# branch on it keep working unchanged.
TIMED_OUT = 124

# How long a stopped tree gets to exit on its own before it is killed outright.
_GRACE = 5.0

# How often a wait looks again.
_POLL = 0.25


class Processes:
    """Kill trees, bound commands and wait for conditions, the same on every operating system.

    clock: the monotonic clock deadlines are read from.
    sleep: pauses between looks.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clock = clock
        self.sleep = sleep

    def kill(self, pids: Sequence[int], *, force: bool = False) -> list[int]:
        """Stop each process and everything it started, children first, answering who was gone.

        pids: the processes to stop.
        force: kill at once instead of asking each process to terminate.
        """
        gone = []
        for pid in pids:
            try:
                terminate_process_tree(pid, force=force)
            except psutil.NoSuchProcess:
                gone.append(pid)
        return gone

    def timeout(self, seconds: float, command: Sequence[str]) -> int:
        """Run `command` with this terminal's stdio, stopping its whole tree after `seconds`.

        Answers the command's own exit status, or `TIMED_OUT` when it had to be stopped. The
        tree gets a grace period to exit after it is asked to, then is killed, so a job that
        ignores the request still ends.

        seconds: the hard limit.
        command: the program and its arguments, run without a shell.
        """
        if not command:
            raise MissionError("proc timeout needs a command to run")
        try:
            process = subprocess.Popen(list(command))  # ruff:ignore[subprocess-without-shell-equals-true]  reason=the caller's own argv, run without a shell since=2026-09-25
        except OSError as fault:
            raise MissionError(f"cannot start {command[0]}: {fault}") from fault
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process.pid)
        try:
            process.wait(timeout=_GRACE)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process.pid, force=True)
            process.wait()
        return TIMED_OUT

    def wait(
        self, *, file: Path | None = None, port: str = "", pid: int = 0, seconds: float = 0.0
    ) -> bool:
        """Block until every named condition holds, answering whether they did in time.

        file: a path that must exist.
        port: a `host:port` that must accept a TCP connection.
        pid: a process that must have exited.
        seconds: how long to wait at most, 0 for as long as it takes.
        """
        if file is None and not port and not pid:
            raise MissionError("proc wait needs --file, --port or --pid")
        deadline = self.clock() + seconds if seconds else float("inf")
        while not self._holds(file, port, pid):
            if self.clock() >= deadline:
                return False
            self.sleep(_POLL)
        return True

    def _holds(self, file: Path | None, port: str, pid: int) -> bool:
        """Whether every named condition holds right now."""
        return (
            (file is None or file.exists())
            and (not port or _accepts(port))
            and (not pid or not psutil.pid_exists(pid))
        )


def _accepts(port: str) -> bool:
    """Whether `host:port` accepts a TCP connection within a moment."""
    host, _, number = port.rpartition(":")
    if not number.isdigit():
        raise MissionError(f"a port is written host:port, not {port!r}")
    try:
        with socket.create_connection((host or "localhost", int(number)), timeout=_POLL):
            return True
    except OSError:
        return False
