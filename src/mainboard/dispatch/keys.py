# The ssh client, agent and shared connections this tool keeps, so a host is reached without a
# prompt.
#
# A key with a passphrase is silenced by an agent holding it. A client and an agent must come from
# one build, since each Windows build reaches its agent differently: Microsoft's through the agent
# service's named pipe, Git's or MSYS2's through an emulated socket file that Microsoft's client
# opens as a plain file and writes its requests into, so its `ssh-add` answers "invalid format".
# Every agent tool here is therefore the sibling of the one `ssh` this tool runs. A machine whose
# user already runs an agent (`SSH_AUTH_SOCK` set) keeps it.
#
# A host that asks for more than a key on every login (miyabi's one-time code) is silenced only by
# sharing one login, which `unlock` opens by hand and keeps for days. Ordinary multiplexing passes
# file descriptors over the control socket, which no Windows build can; proxy mode (`-O proxy`,
# OpenSSH 7.4) speaks the SSH protocol over that socket instead and works on Git's and MSYS2's
# builds. Microsoft's build has no multiplexing at all and fails outright when asked, so on Windows
# this tool runs Git's or MSYS2's ssh whenever one is installed.

import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the ssh client's own agent tools with fixed argv since=2026-09-28
from functools import cache
from pathlib import Path
from shutil import which

from ..core.errors import MissionError
from ..core.host import WINDOWS

# The sockets this tool owns, beside the user's ssh config: its agent's and one per shared login.
CONTROL = Path.home() / ".ssh" / "cm"

# Where a POSIX-socket agent listens, in the forward-slash form Git's ssh reads on Windows too.
SOCKET = (CONTROL / "agent.sock").as_posix()

# Where Microsoft's agent service listens.
SERVICE_PIPE = r"\\.\pipe\openssh-ssh-agent"

# How long a shared login outlives its last session, as the Mac's own config keeps miyabi's.
PERSIST = "72h"

# `ssh-add -l`'s answers: 0 lists keys, 1 holds none; anything else means no agent listens.
_LISTENING = frozenset({0, 1})


@cache
def client() -> Path:
    """The ssh this tool runs: on Windows a Git or MSYS2 build ahead of Microsoft's, whose missing
    multiplexing would leave a host that asks for a one-time code unreachable."""
    found = which("ssh")
    candidates = [*(_posix_builds() if WINDOWS else ()), *([Path(found)] if found else [])]
    if not candidates:
        raise MissionError("no ssh on PATH; install an OpenSSH client")
    return candidates[0]


def microsoft() -> bool:
    """Whether the ssh this tool runs is Microsoft's Windows build rather than Git's or MSYS2's."""
    return WINDOWS and "usr" not in client().parts


def address() -> str:
    """Where the agent this tool uses listens, as its ssh spells it."""
    return SERVICE_PIPE if microsoft() else SOCKET


def shared(host: str) -> tuple[str, ...]:
    """The options that route an ssh to `host` through the login `unlock` opened for it, none
    when no such login answers."""
    return (*_control(host), "-O", "proxy") if _live(host) else ()


def serving(socket: str | None = None) -> bool:
    """Whether an agent answers on `socket`, the tool's own address when none is given."""
    socket = socket or address()
    if socket == SOCKET and not Path(socket).exists():
        return False
    listed = subprocess.run(
        [_tool("ssh-add"), "-l"],
        env={**os.environ, "SSH_AUTH_SOCK": socket},
        capture_output=True,
        timeout=10,
        check=False,
    )
    return listed.returncode in _LISTENING


@cache
def adopt() -> None:
    """Point every ssh this process starts at the tool's agent, unless the user runs their own.

    Asked by whatever is about to start ssh, once per process, so a command that never reaches
    another machine never looks for an agent.
    """
    if not os.environ.get("SSH_AUTH_SOCK") and serving():
        os.environ["SSH_AUTH_SOCK"] = address()


def started() -> str:
    """The tool's agent, started when none answers; its address."""
    if serving():
        return address()
    if microsoft():
        raise MissionError(
            "Windows' ssh-agent service is not running; in an administrator PowerShell run "
            "`Set-Service ssh-agent -StartupType Automatic; Start-Service ssh-agent`"
        )
    Path(SOCKET).unlink(missing_ok=True)
    CONTROL.mkdir(parents=True, exist_ok=True)
    subprocess.run([_tool("ssh-agent"), "-a", SOCKET], capture_output=True, timeout=10, check=True)
    if not serving():
        raise MissionError(f"ssh-agent did not start listening on {SOCKET}")
    return SOCKET


def identities(host: str) -> list[str]:
    """The key files ssh would offer `host`, as its own config resolves them, those that exist."""
    resolved = subprocess.run(
        [str(client()), "-G", host],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=True,
    ).stdout
    files = (
        Path(line.split(maxsplit=1)[1]).expanduser()
        for line in resolved.splitlines()
        if line.lower().startswith("identityfile ")
    )
    return [str(path) for path in files if path.is_file()]


def unlock(host: str) -> int:
    """Make `host` answer without a prompt: add its keys to the tool's agent, asking for each
    passphrase once, and when it still asks for more, log in once by hand and keep that login for
    every later ssh; the last check's exit status."""
    environ = {**os.environ, "SSH_AUTH_SOCK": started()}
    if not _silent(host, environ):
        return 0
    keys = identities(host)
    if not keys:
        raise MissionError(f"ssh names no key file for {host}; add an IdentityFile to its config")
    for key in keys:
        subprocess.run([_tool("ssh-add"), key], env=environ, check=False)
    if not _silent(host, environ):
        return 0
    CONTROL.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(client()),
            *_control(host),
            *("-o", "ControlMaster=yes", "-o", f"ControlPersist={PERSIST}", "-fN", host),
        ],
        env=environ,
        check=False,
    )
    _live.cache_clear()
    return _silent(host, environ)


def _posix_builds() -> list[Path]:
    """Git's and MSYS2's ssh on this Windows machine, those on PATH first, then Git's bundled one
    (Git puts only its `cmd` folder on PATH)."""
    folders = [Path(folder) for folder in os.environ.get("PATH", "").split(os.pathsep) if folder]
    git = which("git")
    if git:
        folders += [parent / "usr" / "bin" for parent in Path(git).parents]
    builds = (folder / "ssh.exe" for folder in folders if "usr" in folder.parts)
    return [build for build in builds if build.is_file()]


def _control(host: str) -> tuple[str, str]:
    """The option naming `host`'s shared-login socket."""
    return ("-o", f"ControlPath=~/.ssh/cm/mb-{host}")


@cache
def _live(host: str) -> bool:
    """Whether a shared login to `host` answers, clearing the socket one that died left behind."""
    socket = CONTROL / f"mb-{host}"
    if not socket.exists():
        return False
    checked = subprocess.run(
        [str(client()), *_control(host), "-O", "check", host],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if checked.returncode:
        socket.unlink(missing_ok=True)
    return not checked.returncode


def _tool(name: str) -> str:
    """The agent tool `name` from the directory of the ssh this tool runs."""
    return str(client().with_name(name + client().suffix))


def _silent(host: str, environ: dict[str, str]) -> int:
    """`host`'s answer to a login that may not prompt, 0 when it let one in."""
    return subprocess.run(
        [
            str(client()),
            *shared(host),
            *("-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "true"),
        ],
        env=environ,
        capture_output=True,
        check=False,
    ).returncode
