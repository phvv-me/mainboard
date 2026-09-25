# What keeps a test suite off every real machine but the one it runs on.
#
# A suite that reaches a real host does not fail: ssh times out or answers "could not resolve",
# the code under test reads that as an unreachable host, and the test goes green having proved
# nothing, or goes red only on the one runner whose DNS or ssh agent answered differently. The
# ad-hoc answer was an `ssh` stub first on PATH, which covered one tool on POSIX alone and let
# the test pass. The seal covers every spawn Python makes, on every platform, and every socket
# that leaves loopback, and whatever it refuses fails the test even when the code under test
# swallowed the refusal.
#
# Two layers, because a remote tool is reached two ways. Python's own spawns all funnel through
# `Popen._execute_child`, whichever module imported `Popen` and however it spelled the command,
# so wrapping that one method sees plumbum's ssh sessions and a bare `subprocess.run` alike. A
# tool that starts ssh itself (rsync's `-e`, git over ssh) never passes back through Python, so a
# directory of refusing stand-ins goes first on PATH as well, POSIX only since Windows resolves a
# program by `.exe` and never runs a script stand-in.

import os
import socket
import subprocess
from ipaddress import ip_address
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Concatenate, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pytest

# The programs that reach another machine: the ssh family and every batch scheduler's client.
REMOTE_TOOLS = frozenset(
    {
        "ssh",
        "scp",
        "sftp",
        "qsub",
        "qstat",
        "qdel",
        "pbsnodes",
        "sbatch",
        "squeue",
        "scancel",
        "sinfo",
    }
)

# The one private method every `Popen` starts its child through, on POSIX and Windows alike, and
# absent from the type stubs for being private, which is why it is reached by name.
SPAWN = "_execute_child"

# Where every stand-in appends the command line it was given, the one channel a process spawned
# by another process has back to the test that caused it.
_LOG_VAR = "MAINBOARD_SEALED_LOG"

_STAND_IN = f"""#!/bin/sh
echo "$(basename "$0") $*" >> "${_LOG_VAR}"
echo "sealed: a test reached for $(basename "$0"), which never leaves this machine" >&2
exit 255
"""

# The names a connection may still reach: this machine under every spelling it answers to.
_LOCAL_NAMES = frozenset({"", "localhost", "0.0.0.0"})


# Everything `Popen` accepts as the command: one line, or an argv whose first word runs.
type Command = (
    str
    | bytes
    | os.PathLike[str]
    | os.PathLike[bytes]
    | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes]]
)


class RemoteReached(RuntimeError):
    """A test reached for another machine, which the seal refuses."""


def program(args: Command) -> str:
    """The program `args` starts, its bare lowercase name without a Windows extension.

    args: what `Popen` was handed, a command line string or an argv whose first word runs.
    """
    match args:
        case str() | bytes():
            first = next(iter(os.fsdecode(args).split()), "")
        case os.PathLike():
            first = os.fsdecode(args)
        case _:
            first = os.fsdecode(next(iter(args), ""))
    name = PurePath(first.replace("\\", "/")).name.lower()
    return name.removesuffix(".exe")


def is_local(address: tuple[str, int] | tuple[str, int, int, int] | str | bytes) -> bool:
    """Whether a socket `address` stays on this machine: a Unix path, loopback or the wildcard.

    address: what `socket.connect` was handed for any family.
    """
    if not isinstance(address, tuple):
        return True
    host = str(address[0]).lower()
    if host in _LOCAL_NAMES:
        return True
    try:
        return ip_address(host.partition("%")[0]).is_loopback
    except ValueError:
        return False


class Seal:
    """Every attempt one test made at another machine, refused as it happened.

    stand_ins: the directory holding a refusing stand-in per remote tool, first on PATH.
    """

    def __init__(self, stand_ins: Path) -> None:
        self.stand_ins = stand_ins
        self.attempts: list[str] = []

    @classmethod
    def made(cls, directory: Path) -> Self:
        """A seal whose stand-ins are written into `directory`, its log beside them."""
        directory.mkdir(parents=True, exist_ok=True)
        for tool in REMOTE_TOOLS:
            stand_in = directory / tool
            stand_in.write_text(_STAND_IN, encoding="utf-8", newline="\n")
            stand_in.chmod(0o755)
        return cls(directory)

    @property
    def log(self) -> Path:
        """The file every stand-in appends its command line to."""
        return self.stand_ins / "reached.log"

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Refuse remote spawns and connections for as long as `monkeypatch` holds."""
        self.log.write_text("", encoding="utf-8")
        monkeypatch.setenv(_LOG_VAR, str(self.log))
        monkeypatch.setenv("PATH", os.pathsep.join([str(self.stand_ins), os.environ["PATH"]]))
        spawn = getattr(subprocess.Popen, SPAWN)
        monkeypatch.setattr(subprocess.Popen, SPAWN, self._spawning(spawn))
        for name in ("connect", "connect_ex"):
            connect = getattr(socket.socket, name)
            monkeypatch.setattr(socket.socket, name, self._connecting(connect))

    def breaches(self) -> list[str]:
        """Every attempt so far, the ones Python refused and the ones a stand-in logged."""
        logged = self.log.read_text(encoding="utf-8").splitlines()
        return [*self.attempts, *(f"spawned {line}" for line in logged)]

    def _refuse(self, attempt: str) -> RemoteReached:
        self.attempts.append(attempt)
        return RemoteReached(f"sealed: a test {attempt}; fake the transport seam instead")

    def _spawning[**Rest](
        self, spawn: Callable[Concatenate[subprocess.Popen[bytes], Command, Rest], None]
    ) -> Callable[Concatenate[subprocess.Popen[bytes], Command, Rest], None]:
        def guarded(
            popen: subprocess.Popen[bytes],
            args: Command,
            /,
            *rest: Rest.args,
            **named: Rest.kwargs,
        ) -> None:
            if (name := program(args)) in REMOTE_TOOLS:
                raise self._refuse(f"spawned {name}")
            spawn(popen, args, *rest, **named)

        return guarded

    def _connecting(
        self, connect: Callable[[socket.socket, tuple[str, int]], int | None]
    ) -> Callable[[socket.socket, tuple[str, int]], int | None]:
        def guarded(sock: socket.socket, address: tuple[str, int]) -> int | None:
            if not is_local(address):
                raise self._refuse(f"connected to {address[0]}:{address[1]}")
            return connect(sock, address)

        return guarded
