# How the center asks a target's agent one thing. The target runs whatever Python its shell
# answers to with a one-line bootstrap, and everything else travels on standard input in one
# framed stream: the agent's own source, the request as one JSON line, then any payload. Nothing
# is installed on the far side and nothing is quoted for its shell but the bootstrap, which is
# plain enough for sh, cmd.exe and PowerShell alike.

import io
import json
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=ssh argv built from typed fields, not untrusted input since=2026-09-25
import threading
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import IO, TYPE_CHECKING, Protocol

import psutil

from ...core.errors import MissionError
from ..transport import HostUnreachable, SshTransport, is_transport_failure, terminate_process_tree
from . import program

if TYPE_CHECKING:
    from collections.abc import Buffer, Callable

    from .program import Json, Request

# Reads the framed source off standard input and runs it, leaving the rest of the stream to it.
# No `$`, backquote or inner double quote, so every shell a target might log in to passes it on.
BOOTSTRAP = ";".join(
    (
        "import sys",
        "b=sys.stdin.buffer",
        "exec(compile(b.read(int(b.readline())),'mainboard-agent','exec'))",
    )
)

# The agent's source exactly as it ships, read once.
_SOURCE = Path(program.__file__).read_bytes()

# How much of what the agent says on stderr is kept for the one line a failure names.
_TAIL = 8192


class AgentRefused(MissionError):
    """The agent ran and declined, or failed, and said why on its last line."""


class Process(Protocol):
    """What a running agent is to the exchange: three pipes, a pid and an exit status."""

    stdin: IO[bytes] | None
    stdout: IO[bytes] | None
    stderr: IO[bytes] | None
    pid: int
    returncode: int | None

    def wait(self, timeout: float | None = None) -> int:
        """Block until the agent exits or `timeout` passes, raising `TimeoutExpired` then."""
        ...


class Link(ABC):
    """How one target is reached: the process that runs a command line there, and its end.

    host: the name every failure is reported under.
    """

    def __init__(self, host: str) -> None:
        self.host = host

    @abstractmethod
    def spawn(self, command: str) -> Process:
        """Start `command` on the target with all three standard streams piped."""

    @abstractmethod
    def end(self, process: Process) -> None:
        """Stop `process` and everything it started, then reap it."""


class SshLink(Link):
    """A target reached by OpenSSH under one bounded policy.

    host: the alias, or the declared alias of a rental whose policy carries its endpoint.
    ssh: the liveness and endpoint policy every connection rides.
    """

    def __init__(self, host: str, ssh: SshTransport | None = None) -> None:
        super().__init__(host)
        self.ssh = ssh or SshTransport()

    def spawn(self, command: str) -> Process:
        """One ssh process in its own session, so a stall can end its whole process group."""
        argv = ["ssh", *self.ssh.options, self.ssh.destination(self.host), command]
        try:
            return subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=ssh argv built from typed fields since=2026-09-25
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            raise HostUnreachable(f"ssh to {self.host!r} could not start: {error}") from error

    def end(self, process: Process) -> None:
        """Kill the ssh process tree, ProxyJump children included."""
        with suppress(psutil.Error):
            terminate_process_tree(process.pid, force=True)
        process.wait()


class Agent:
    """The standard-library agent on one target, asked one request at a time.

    link: how the target is reached.
    python: the interpreter command the target's login shell runs the agent with.
    patience: seconds an exchange may pass without a byte moving either way before the target
        counts as stalled. The exchange itself is unbounded, since a slow uplink is only slow.
    """

    def __init__(self, link: Link, *, python: str = "python3", patience: float = 60.0) -> None:
        self.link = link
        self.python = python
        self.patience = patience

    @property
    def host(self) -> str:
        """The target's name, as failures report it."""
        return self.link.host

    def ask(
        self, request: Request, *, payload: Callable[[Sink], None] | None = None
    ) -> list[Json]:
        """Send `request`, stream `payload` behind it, and answer the records the agent wrote.

        payload: writes the bytes that follow the request, a tar stream for a mirror; its own
            exception wins over whatever the agent then says about the stream ending early.
        """
        process = self.link.spawn(f'{self.python} -c "{BOOTSTRAP}"')
        exchange = _Exchange(process, request, payload)
        exchange.start()
        while not exchange.settled(timeout=min(1.0, self.patience)):
            if exchange.idle > self.patience:
                self.link.end(process)
                exchange.join()
                raise HostUnreachable(
                    f"agent on {self.host!r} moved nothing for {self.patience:g}s"
                )
        exchange.join()
        if exchange.fault is not None:
            raise exchange.fault
        code = process.returncode or 0
        if code == 0:
            return exchange.records
        said = exchange.said
        if is_transport_failure(code, said):
            raise HostUnreachable(f"agent on {self.host!r} unreachable: {_last(said, code)}")
        raise AgentRefused(_last(said, code))


class Sink(io.BufferedIOBase):
    """The stdin a payload writes to, noting the moment of every write for the stall watch."""

    def __init__(self, stream: IO[bytes], exchange: _Exchange) -> None:
        super().__init__()
        self.stream = stream
        self.exchange = exchange

    def write(self, data: Buffer, /) -> int:
        """Send `data` on to the agent."""
        sent = bytes(data)
        self.stream.write(sent)
        self.exchange.touch()
        return len(sent)


class _Exchange:
    """One request in flight: a writer feeding stdin, readers draining stdout and stderr."""

    def __init__(
        self,
        process: Process,
        request: Request,
        payload: Callable[[Sink], None] | None,
    ) -> None:
        self.process = process
        self.request = request
        self.payload = payload
        self.records: list[Json] = []
        self.tail = bytearray()
        self.fault: BaseException | None = None
        self.moved = monotonic()
        self.threads = [
            threading.Thread(target=target, daemon=True)
            for target in (self.__write, self.__read, self.__listen)
        ]

    @property
    def idle(self) -> float:
        """Seconds since a byte last moved either way."""
        return monotonic() - self.moved

    @property
    def said(self) -> str:
        """What the agent wrote on stderr, the tail of it."""
        return bytes(self.tail).decode("utf-8", errors="replace")

    def touch(self) -> None:
        """Note that a byte just moved."""
        self.moved = monotonic()

    def start(self) -> None:
        """Start the writer and both readers."""
        for thread in self.threads:
            thread.start()

    def settled(self, *, timeout: float) -> bool:
        """Whether the agent exited within `timeout` seconds."""
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def join(self) -> None:
        """Wait for the writer and both readers to finish with their pipes."""
        for thread in self.threads:
            thread.join()

    def __write(self) -> None:
        """Frame the source, send the request, stream the payload, then close stdin."""
        stdin = self.process.stdin
        if stdin is None:
            return
        try:
            stdin.write(b"%d\n" % len(_SOURCE) + _SOURCE)
            stdin.write(json.dumps(self.request).encode("utf-8") + b"\n")
            if self.payload is not None:
                self.payload(Sink(stdin, self))
        except BrokenPipeError:
            pass
        except BaseException as fault:
            self.fault = fault
        finally:
            with suppress(BrokenPipeError):
                stdin.close()

    def __read(self) -> None:
        """Collect the agent's JSON records as they arrive."""
        stdout = self.process.stdout
        if stdout is None:
            return
        for line in stdout:
            self.touch()
            self.records.append(json.loads(line))

    def __listen(self) -> None:
        """Keep the tail of the agent's stderr."""
        stderr = self.process.stderr
        if stderr is None:
            return
        for chunk in iter(lambda: stderr.read(4096), b""):
            self.touch()
            self.tail += chunk
            del self.tail[:-_TAIL]


def _last(said: str, code: int) -> str:
    """The last thing the agent said, or its exit status when it said nothing."""
    lines = [line for line in said.strip().splitlines() if line.strip()]
    return lines[-1].removeprefix("mainboard: ") if lines else f"exit {code}"
