# How the center asks a target's agent one thing. The target runs mainboard's own Python there
# (`Dialect.python`) with a one-line bootstrap, and everything else travels on standard input in
# one framed stream: the agent's own source, the request as one JSON line, then any payload.
# Nothing is quoted for the target's shell but the bootstrap.

import inspect
import io
import json
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=ssh argv built from typed fields, not untrusted input since=2026-09-25
import threading
from contextlib import suppress
from time import monotonic
from types import ModuleType
from typing import IO, TYPE_CHECKING, ClassVar

import psutil

from ...core.errors import MissionError
from ..transport.core import (
    HostUnreachable,
    SshTransport,
    is_transport_failure,
    terminate_process_tree,
)
from .program import constants, exceptions, snapshot, transfer, walking

if TYPE_CHECKING:
    from collections.abc import Buffer, Callable

    from .program import Request
    from .program.walking import Json


def _bundled(*modules: ModuleType) -> bytes:
    """The agent as one source, then `transfer.main` run on the rest of the stream.

    Each module is registered under one package, in the order given (each importing only those
    before it), so their relative imports resolve on a target as they do here.
    """
    lines = [
        "import sys,types",
        "p=types.ModuleType('mainboard_agent');p.__path__=[];sys.modules[p.__name__]=p",
        *(
            f"m=types.ModuleType(p.__name__+'.{module.__name__.rpartition('.')[2]}');"
            "m.__package__=p.__name__;sys.modules[m.__name__]=m;"
            f"exec(compile({inspect.getsource(module)!r},m.__name__,'exec'),m.__dict__)"
            for module in modules
        ),
        "sys.exit(sys.modules[p.__name__+'.transfer'].main("
        "sys.stdin.buffer,stdout=sys.stdout.buffer,stderr=sys.stderr))",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


# The agent as it ships, read once.
_SOURCE = _bundled(constants, exceptions, walking, snapshot, transfer)

# How much of what the agent says on stderr is kept for the one line a failure names.
_TAIL = 8192


class AgentRefused(MissionError):
    """The agent ran and declined, or failed, and said why on its last line."""


class SshLink:
    """A target reached by OpenSSH under one bounded policy.

    host: the alias, or the declared alias of a rental whose policy carries its endpoint.
    ssh: the liveness and endpoint policy every connection rides.
    """

    def __init__(self, host: str, ssh: SshTransport | None = None) -> None:
        self.host = host
        self.ssh = ssh or SshTransport()

    def end(self, process: subprocess.Popen[bytes]) -> None:
        """Kill the ssh process tree, ProxyJump children included."""
        with suppress(psutil.Error):
            terminate_process_tree(process.pid, force=True)
        process.wait()

    def spawn(self, command: str) -> subprocess.Popen[bytes]:
        """One ssh process in its own session, so a stall can end its whole process group."""
        argv = [*self.ssh.command(self.host), command]
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


class Agent:
    """The standard-library agent on one target, asked one request at a time.

    link: how the target is reached.
    python: the command starting mainboard's own Python on the target (`Dialect.python`).
    patience: seconds an exchange may pass without a byte moving either way before the target
        counts as stalled. The exchange itself is unbounded, since a slow uplink is only slow.
    """

    # Reads the framed source off standard input and runs it, leaving the rest of the stream to it.
    # No `$`, backquote or inner double quote, so every shell a target might log in to passes it
    # on.
    BOOTSTRAP: ClassVar[str] = (
        "import sys;b=sys.stdin.buffer;"
        "exec(compile(b.read(int(b.readline())),'mainboard-agent','exec'))"
    )

    @property
    def host(self) -> str:
        return self.link.host

    def __answered(self, process: subprocess.Popen[bytes], exchange: _Exchange) -> list[Json]:
        """The records of an exchange that ended, or why it failed.

        It failed by its payload's own fault, a transport that dropped, or the agent's refusal.
        """
        if exchange.fault is not None:
            raise exchange.fault
        code = process.returncode or 0
        if code == 0:
            return exchange.records
        if is_transport_failure(code, exchange.said):
            raise HostUnreachable(
                f"agent on {self.host!r} unreachable: {_last(exchange.said, code)}"
            )
        raise AgentRefused(_last(exchange.said, code))

    def __init__(self, link: SshLink, *, python: str, patience: float = 60.0) -> None:
        self.link = link
        self.python = python
        self.patience = patience

    def __waited(self, process: subprocess.Popen[bytes], exchange: _Exchange) -> None:
        """Wait for the agent to exit, ending it as stalled once nothing moved for `patience`."""
        while not exchange.settled(timeout=min(1.0, self.patience)):
            if exchange.idle > self.patience:
                self.link.end(process)
                exchange.join()
                raise HostUnreachable(
                    f"agent on {self.host!r} moved nothing for {self.patience:g}s"
                )
        exchange.join()

    def ask(
        self, request: Request, *, payload: Callable[[Sink], None] | None = None
    ) -> list[Json]:
        """Send `request`, stream `payload` behind it, and answer the records the agent wrote.

        payload: writes the bytes that follow the request, a tar stream for a mirror; its own
            exception wins over whatever the agent then says about the stream ending early.
        """
        process = self.link.spawn(f'{self.python} -c "{Agent.BOOTSTRAP}"')
        exchange = _Exchange(process, request, payload)
        exchange.start()
        self.__waited(process, exchange)
        return self.__answered(process, exchange)


class Sink(io.BufferedIOBase):
    """The stdin a payload writes to, noting the moment of every write for the stall watch."""

    def __init__(self, stream: IO[bytes], exchange: _Exchange) -> None:
        super().__init__()
        self.stream = stream
        self.exchange = exchange

    def write(self, data: Buffer, /) -> int:
        sent = bytes(data)
        self.stream.write(sent)
        self.exchange.touch()
        return len(sent)


class _Exchange:
    """One request in flight: a writer feeding stdin, readers draining stdout and stderr."""

    @property
    def idle(self) -> float:
        """Seconds since a byte last moved either way."""
        return monotonic() - self.moved

    @property
    def said(self) -> str:
        """The tail of what the agent wrote on stderr."""
        return bytes(self.tail).decode("utf-8", errors="replace")

    def __init__(
        self,
        process: subprocess.Popen[bytes],
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

    def __listen(self) -> None:
        """Keep the tail of the agent's stderr."""
        stderr = self.process.stderr
        if stderr is None:
            return
        for chunk in iter(lambda: stderr.read(4096), b""):
            self.touch()
            self.tail += chunk
            del self.tail[:-_TAIL]

    def __read(self) -> None:
        """Collect the agent's JSON records."""
        stdout = self.process.stdout
        if stdout is None:
            return
        for line in stdout:
            self.touch()
            # A blank line is the agent's pulse while it works: movement, not a record.
            if line.strip():
                self.records.append(json.loads(line))

    def __send(self, stdin: IO[bytes]) -> None:
        """Frame the source, send the request, then stream the payload."""
        stdin.write(b"%d\n" % len(_SOURCE) + _SOURCE)
        stdin.write(json.dumps(self.request).encode("utf-8") + b"\n")
        if self.payload is not None:
            self.payload(Sink(stdin, self))

    def __write(self) -> None:
        """Frame the source, send the request, stream the payload, then close stdin."""
        stdin = self.process.stdin
        if stdin is None:
            return
        try:
            self.__send(stdin)
        except BrokenPipeError:
            # The agent ended before reading it all; its exit status and stderr say why.
            return
        except BaseException as fault:
            self.fault = fault
        finally:
            with suppress(BrokenPipeError):
                stdin.close()

    def join(self) -> None:
        for thread in self.threads:
            thread.join()

    def settled(self, *, timeout: float) -> bool:
        """Whether the agent exited within `timeout` seconds."""
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def touch(self) -> None:
        """Note that a byte just moved."""
        self.moved = monotonic()


def _last(said: str, code: int) -> str:
    """The last thing the agent said, or its exit status when it said nothing."""
    lines = [line for line in said.strip().splitlines() if line.strip()]
    return lines[-1].removeprefix("mainboard: ") if lines else f"exit {code}"
