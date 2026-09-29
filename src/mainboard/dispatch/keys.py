# The ssh-agent this tool keeps, so a host whose key has a passphrase is reached without a prompt.
#
# Sharing one connection (`ControlMaster`) needs a socket that passes file descriptors, which
# neither Git's nor Microsoft's Windows ssh has: every ssh this tool starts is its own login. What
# makes those logins silent is an agent holding the key. Microsoft's agent service speaks a named
# pipe the ssh on PATH (Git's) cannot use, and a Git Bash agent dies with its terminal, so this
# keeps one agent on a fixed socket beside the user's ssh config, started on demand, outliving
# every process, and points every ssh the tool starts at it. A machine whose user already runs an
# agent (`SSH_AUTH_SOCK` set) keeps it.

import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the ssh client's own agent tools with fixed argv since=2026-09-28
from functools import cache
from pathlib import Path
from shutil import which

from ..core.errors import MissionError

# Where the agent listens, in the forward-slash form Git's ssh reads on Windows too.
SOCKET = (Path.home() / ".ssh" / "mb-agent.sock").as_posix()

# `ssh-add -l`'s answers: 0 lists keys, 1 holds none; anything else means no agent listens.
_LISTENING = frozenset({0, 1})


def serving(socket: str = SOCKET) -> bool:
    """Whether an agent answers on `socket`."""
    if not Path(socket).exists() or which("ssh-add") is None:
        return False
    listed = subprocess.run(
        ["ssh-add", "-l"],
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
        os.environ["SSH_AUTH_SOCK"] = SOCKET


def started() -> str:
    """The tool's agent, started when none answers; its socket."""
    if serving():
        return SOCKET
    Path(SOCKET).unlink(missing_ok=True)
    Path(SOCKET).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh-agent", "-a", SOCKET], capture_output=True, timeout=10, check=True)
    if not serving():
        raise MissionError(f"ssh-agent did not start listening on {SOCKET}")
    return SOCKET


def identities(host: str) -> list[str]:
    """The key files ssh would offer `host`, as its own config resolves them, those that exist."""
    resolved = subprocess.run(
        ["ssh", "-G", host],
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
    """Add `host`'s keys to the tool's agent, asking for each passphrase once, then prove the
    host answers without a prompt; the last command's exit status."""
    environ = {**os.environ, "SSH_AUTH_SOCK": started()}
    if not _silent(host, environ):
        return 0
    keys = identities(host)
    if not keys:
        raise MissionError(f"ssh names no key file for {host}; add an IdentityFile to its config")
    # A key left locked matters only if the host still refuses, which the last check says.
    for key in keys:
        subprocess.run(["ssh-add", key], env=environ, check=False)
    return _silent(host, environ)


def _silent(host: str, environ: dict[str, str]) -> int:
    """`host`'s answer to a login that may not prompt, 0 when it let one in."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "true"],
        env=environ,
        capture_output=True,
        check=False,
    ).returncode
