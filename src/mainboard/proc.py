# `mainboard proc`: the process chores no binary does the same on macOS and Linux. `timeout` is
# GNU coreutils, absent from macOS; killing a process tree is `pkill -P` or `kill -- -pgid`
# depending on how it was started; and waiting for a file or a port is a shell loop around
# `sleep`, `test` and `nc`.

import socket
import subprocess
import time
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from .core.errors import MissionError
from .dispatch.transport.core import terminate_process_tree

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import IO


class Exit(IntEnum):
    """The exit statuses a command did not choose, spelled as the shell spells them.

    TIMED_OUT: GNU `timeout`'s status for a command it had to stop, so scripts branching on it
        still work.
    MISSING: the shell's status for a command it could not find.
    """

    TIMED_OUT = 124
    MISSING = 127


# Seconds a stopped tree gets to exit on its own before it is killed outright.
_GRACE = 5.0

# Seconds between a wait's looks.
_POLL = 0.25


class Processes:
    """Kill trees, bound commands and wait for conditions, the same on every operating system."""

    @staticmethod
    def _accepts(port: str) -> bool:
        """Whether `host:port` accepts a TCP connection within a moment."""
        host, _, number = port.rpartition(":")
        if not number.isdigit():
            raise MissionError(f"a port is written host:port, not {port!r}")
        try:
            connection = socket.create_connection(
                (host or "localhost", int(number)), timeout=_POLL
            )
        except OSError:
            return False
        connection.close()
        return True

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clock = clock
        self.sleep = sleep

    def _holds(self, file: Path | None, port: str, pid: int) -> bool:
        return (
            (file is None or file.exists())
            and (not port or Processes._accepts(port))
            and (not pid or not psutil.pid_exists(pid))
        )

    def kill(self, pids: Sequence[int], *, force: bool = False) -> list[int]:
        """Stop each process and everything it started, children first, answering who was gone.

        force: kill at once instead of asking each process to terminate.
        """
        gone = []
        for pid in pids:
            try:
                terminate_process_tree(pid, force=force)
            except psutil.NoSuchProcess:
                gone.append(pid)
        return gone

    def matching(self, pattern: str) -> list[dict[str, str | int | float]]:
        """Every process whose command line contains `pattern`, started by this tool or not.

        An empty pattern lists this user's processes only; a pattern searches everyone's, so a
        job somebody started by hand on a shared host is found too. This process is left out.
        """
        current = psutil.Process()
        user = current.username()
        found = []
        attributes = ["pid", "ppid", "username", "cpu_percent", "memory_info", "create_time"]
        for process in psutil.process_iter([*attributes, "cmdline", "name"]):
            info = process.info
            line = " ".join(info["cmdline"] or []) or info["name"] or ""
            if process.pid == current.pid or (pattern and pattern not in line):
                continue
            if not pattern and info["username"] != user:
                continue
            memory = info["memory_info"]
            found.append(
                {
                    "pid": process.pid,
                    "ppid": info["ppid"] or 0,
                    "user": info["username"] or "",
                    "cpu_pct": info["cpu_percent"] or 0.0,
                    "rss_mb": round(memory.rss / 2**20) if memory else 0,
                    "age_s": round(time.time() - (info["create_time"] or time.time())),
                    "command": line[:200],
                }
            )
        return found

    def timeout(
        self,
        seconds: float,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        output: IO[bytes] | None = None,
    ) -> int:
        """Run `command` with this terminal's stdio, stopping its whole tree after `seconds`.

        Answers the command's own exit status, or `Exit.TIMED_OUT` when it had to be stopped. The
        tree gets `_GRACE` to exit after it is asked to, then is killed, so a job that ignores
        the request still ends. `command` runs without a shell; with `output`, both its streams
        go there instead of the terminal.
        """
        if not command:
            raise MissionError("proc timeout needs a command to run")
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT if output else None,
            )  # ruff:ignore[subprocess-without-shell-equals-true]  reason=the caller's own argv, run without a shell since=2026-09-25
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
        return Exit.TIMED_OUT

    def wait(
        self, *, file: Path | None = None, port: str = "", pid: int = 0, seconds: float = 0.0
    ) -> bool:
        """Block until every named condition holds, answering whether they did in time.

        file: must exist.
        port: a `host:port` that must accept a TCP connection.
        pid: must have exited.
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
