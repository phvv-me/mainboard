from pathlib import Path
from typing import TYPE_CHECKING

from ...core.errors import MissionError
from ...log import logger
from ...proc import Processes

if TYPE_CHECKING:
    pass


# The values a compact record leaves out: they say nothing an absent field does not.

from ..base import Verbs
from ..options import Output

# The verbs' default output and batch declaration.
_COMPACT = Output()


class ProcVerbs(Verbs):
    """The `procs` verbs."""

    def proc_kill(
        self,
        pids: list[int] | None = None,
        *,
        match: str = "",
        on: str = "local",
        force: bool = False,
    ) -> int:
        """Stop each process and everything it started, children first, on any system.

        What `pkill -P`, `kill -- -pgid` and `taskkill /T` each do on one system, for any
        process, a job somebody started by hand included. Exits 1 when a process was gone.

        Args:
            pids: the processes to stop.
            match: also stop every process whose command line contains this (see `proc list`).
            on: the host alias, `local` for this machine; a host runs its own copy.
            force: kill at once instead of asking each process to terminate first.
        """
        if on != "local":
            flags = [*(["--match", match] if match else []), *(["--force"] if force else [])]
            self.session.board(on).interact(
                self.session.project.name, "proc", "kill", *map(str, pids or ()), *flags
            )
        chosen = (
            [*(pids or ()), *(int(row["pid"]) for row in Processes().matching(match))]
            if match
            else list(pids or ())
        )
        if not chosen:
            raise MissionError("name a process: a pid, or --match <text> (see `proc list`)")
        gone = Processes().kill(chosen, force=force)
        for pid in gone:
            logger.info(f"no process {pid}")
        return 1 if gone else 0

    def proc_list(
        self, pattern: str = "", *, on: str = "local", output: Output = _COMPACT
    ) -> None:
        """List processes, started by this tool or not: pid, parent, user, cpu, memory, age.

        Args:
            pattern: a substring of the command line (`train.py`, `vllm`); your own processes
                when empty, everyone's when given.
            on: the host alias, `local` for this machine; a host answers with its own copy.
        """
        if on != "local":
            self.session.board(on).interact(self.session.project.name, "proc", "list", pattern)
        output.print_rows(Processes().matching(pattern), title="proc")

    def proc_timeout(self, seconds: float, *command: str) -> int:
        """Run a command with a hard limit, stopping its whole tree when the limit passes.

        The portable `timeout`: the command runs with this terminal's stdio and without a shell,
        and exits with its own status, or 124 when it had to be stopped, as GNU `timeout` does.
        A tree that ignores the request to stop is killed after a short grace.

        Args:
            command: the program and its arguments, from the first token after the limit.
        """
        return Processes().timeout(seconds, command)

    def proc_wait(
        self, *, file: Path | None = None, port: str = "", pid: int = 0, timeout: float = 0.0
    ) -> int:
        """Block until a file exists, a port accepts, and a process has exited, whichever named.

        The loop around `sleep`, `test` and `nc`, as one verb. Exits 0 once every named
        condition holds and 1 when the timeout passed first.

        Args:
            port: a `host:port` that must accept a TCP connection.
            pid: a process that must have exited.
            timeout: seconds to wait at most, 0 for as long as it takes.
        """
        return 0 if Processes().wait(file=file, port=port, pid=pid, seconds=timeout) else 1
