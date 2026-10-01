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
#
# A key's passphrase is asked for once and kept in the system's keystore (Credential Manager,
# the Keychain, the Secret Service), as the Mac's `UseKeychain` keeps it, so an agent started
# after a reboot is loaded again without asking. `ssh-add` reads it through an askpass script
# from its own environment, never from an argv.

import os
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the ssh client's own agent tools with fixed argv since=2026-09-28
from functools import cache
from getpass import getpass
from pathlib import Path
from shutil import which
from time import sleep

import keyring
from keyring.errors import KeyringError

from ..core.errors import MissionError
from ..core.host import WINDOWS
from ..core.project import Project
from ..log import logger

# The sockets this tool owns, beside the user's ssh config: its agent's and one per shared login.
CONTROL = Path.home() / ".ssh" / "cm"

# Where a POSIX-socket agent listens, in the forward-slash form Git's ssh reads on Windows too.
SOCKET = (CONTROL / "agent.sock").as_posix()

# Where Microsoft's agent service listens.
SERVICE_PIPE = r"\\.\pipe\openssh-ssh-agent"

# `ssh-add -l`'s answers: 0 lists keys, 1 holds none; anything else means no agent listens.
_LISTENING = frozenset({0, 1})

# The keystore service a key's passphrase is filed under, the key's path as the account; the
# keys kept there, one path per line, which a new agent is loaded with; and the script answering
# `ssh-add` from the variable `_PASSPHRASE` it is started with. `ssh-add` asks again after a wrong
# passphrase until it hears an empty one, so the script answers once, marking `_ASKED`, and
# empty after that.
_KEYSTORE = f"{Project().name} ssh key"
_KEPT = CONTROL / "kept"
_ASKPASS = CONTROL / "askpass.sh"
_PASSPHRASE = "MB_SSH_PASSPHRASE"
_ASKED = "MB_SSH_ASKED"
_ANSWER = f"""#!/bin/sh
[ -e "${_ASKED}" ] && exit 0
: > "${_ASKED}"
printf '%s\\n' "${_PASSPHRASE}"
"""

# What a keystore raises when this session has none to offer: keyring's own refusals and, on
# Windows, the credential API's error, raised unwrapped from a logon session that holds no
# credential store (error 1312 in a login over ssh, 2026-10-01).
_REFUSED: tuple[type[Exception], ...] = (KeyringError,)
if WINDOWS:
    from win32ctypes.pywin32.pywintypes import error as _CredentialError

    _REFUSED = (KeyringError, _CredentialError)

# Windows ends a process with the console window it was started from, and Git's ssh does not
# detach from it as a POSIX daemon does: closing the window `unlock` ran in took miyabi-g's
# shared login down twice on 2026-10-01, and the agent with it. The agent and every shared login
# start with no console and outside the window's job, as `_daemon` does.
_BREAKAWAY = subprocess.CREATE_BREAKAWAY_FROM_JOB if WINDOWS else 0
_DETACHED = (
    subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | _BREAKAWAY
    if WINDOWS
    else 0
)

# The script a detached master asks through: it leaves the prompt in the folder `_RELAY` names
# and waits for the answer `unlock` writes there after asking in the terminal.
_RELAYER = CONTROL / "relay.sh"
_RELAY = "MB_SSH_RELAY"
_RELAY_SCRIPT = f"""#!/bin/sh
printf '%s' "$1" > "${_RELAY}/prompt.part" && mv "${_RELAY}/prompt.part" "${_RELAY}/prompt"
while [ ! -e "${_RELAY}/answer" ]; do sleep 0.2; done
cat "${_RELAY}/answer"
rm -f "${_RELAY}/answer"
"""


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
    when no such login answers; they go ahead of every other option, since ssh keeps the first
    value an option is given.

    A client in proxy mode sends no keepalives of its own: they go unanswered through the shared
    login and end the client's session after one interval of silence (miyabi-g, 2026-10-01),
    while the master keeps the login itself alive.
    """
    riding = (*_control(host), "-O", "proxy", "-o", "ServerAliveInterval=0")
    return riding if _live(host) else ()


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
    """Point every ssh this process starts at the tool's agent, unless the user runs their own,
    starting it when the keystore keeps keys to load it with.

    Asked by whatever is about to start ssh, once per process, so a command that never reaches
    another machine never looks for an agent.
    """
    if not os.environ.get("SSH_AUTH_SOCK") and (serving() or (_kept() and started())):
        os.environ["SSH_AUTH_SOCK"] = address()


def started() -> str:
    """The tool's agent, started when none answers and loaded with every key the keystore
    keeps; its address."""
    if serving():
        return address()
    if microsoft():
        raise MissionError(
            "Windows' ssh-agent service is not running; in an administrator PowerShell run "
            "`Set-Service ssh-agent -StartupType Automatic; Start-Service ssh-agent`"
        )
    Path(SOCKET).unlink(missing_ok=True)
    CONTROL.mkdir(parents=True, exist_ok=True)
    _daemon([_tool("ssh-agent"), "-D", "-a", SOCKET], dict(os.environ), CONTROL / "agent.log")
    for _ in range(50):
        if serving():
            break
        sleep(0.1)
    else:
        raise MissionError(f"ssh-agent did not start listening on {SOCKET}")
    for key in _kept():
        passphrase = _keystore(key)
        if passphrase is None or not _add(key, passphrase, SOCKET):
            logger.warning("the keystore no longer opens {}; `unlock` asks for it again", key)
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
    """Make `host` answer without a prompt: add its keys to the tool's agent, asking for a
    passphrase only when the keystore holds none that opens the key, and when it still asks for
    more, log in once by hand and keep that login for every later ssh; the last check's exit
    status."""
    socket = started()
    environ = {**os.environ, "SSH_AUTH_SOCK": socket}
    if not _silent(host, environ):
        return 0
    keys = identities(host)
    if not keys:
        raise MissionError(f"ssh names no key file for {host}; add an IdentityFile to its config")
    for key in keys:
        _remember(key, socket)
    if not _silent(host, environ):
        return 0
    _login(host, environ)
    return _silent(host, environ)


def _login(host: str, environ: dict[str, str]) -> None:
    """Open `host`'s shared login as a daemon, relaying each question it asks to this terminal.

    The master has no console to ask in, so `ssh` hands every prompt (a passphrase, a one-time
    code) to the `_RELAY` script, which leaves it in a folder this process watches and waits for
    the answer typed here. It runs until the site ends the login or `ssh -O exit` does.
    """
    relay = CONTROL / f"relay-{os.getpid()}"
    relay.mkdir(parents=True, exist_ok=True)
    _RELAYER.write_text(_RELAY_SCRIPT, newline="\n")
    _RELAYER.chmod(0o700)
    master = _daemon(
        [str(client()), *_control(host), "-o", "ControlMaster=yes", "-N", host],
        {
            **environ,
            "SSH_ASKPASS": _RELAYER.as_posix(),
            "SSH_ASKPASS_REQUIRE": "force",
            _RELAY: relay.as_posix(),
        },
        CONTROL / f"mb-{host}.log",
    )
    try:
        while master.poll() is None and not _answering(host):
            try:
                prompt = (relay / "prompt").read_text(encoding="utf-8")
            except FileNotFoundError:
                sleep(0.2)
                continue
            (relay / "prompt").unlink()
            (relay / "answer.part").write_text(getpass(prompt) + "\n", newline="\n")
            (relay / "answer.part").replace(relay / "answer")
    finally:
        shutil.rmtree(relay, ignore_errors=True)


def _answering(host: str) -> bool:
    """Whether `host`'s shared login answers now, read afresh rather than from the cache."""
    _live.cache_clear()
    return _live(host)


def _daemon(argv: list[str], environ: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    """`argv` started as a daemon, its output appended to `log`: on Windows with no console and,
    where the window's job allows it, outside that job; elsewhere in a session of its own."""
    with log.open("ab") as sink:
        try:
            return subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=fixed argv of the ssh client's own tools since=2026-10-01
                argv,
                env=environ,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=sink,
                creationflags=_DETACHED,
                start_new_session=not WINDOWS,
            )
        except PermissionError:
            return subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=fixed argv of the ssh client's own tools since=2026-10-01
                argv,
                env=environ,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=sink,
                creationflags=_DETACHED & ~_BREAKAWAY,
            )


def _remember(key: str, socket: str) -> None:
    """Add `key` to the agent at `socket` from the keystore, asking for its passphrase only when
    the keystore holds none that opens it, then keeping the one that did.

    Where nothing can be filed, `ssh-add` asks by itself: Windows' agent service keeps what it is
    given across reboots on its own, and a session with no keystore (a Windows login over ssh
    holds no credentials) has nowhere to keep it.
    """
    if microsoft():
        return _asked(key, socket)
    try:
        kept = keyring.get_password(_KEYSTORE, key)
    except _REFUSED as refusal:
        logger.warning(
            "this session has no system keystore ({}); ssh-add asks for {}", refusal, key
        )
        return _asked(key, socket)
    if kept is not None and _add(key, kept, socket):
        return None
    passphrase = getpass(f"passphrase for {key} (kept in the system keystore): ")
    if not _add(key, passphrase, socket):
        raise MissionError(f"ssh-add refused {key}; was the passphrase right?")
    try:
        keyring.set_password(_KEYSTORE, key, passphrase)
    except _REFUSED as refusal:
        logger.warning(
            "the system keystore refused {}, so a new agent asks again: {}", key, refusal
        )
        return None
    _KEPT.write_text("".join(f"{path}\n" for path in sorted({*_kept(), key})), newline="\n")
    return None


def _asked(key: str, socket: str) -> None:
    """`key` added to the agent at `socket` by `ssh-add`, which asks for its passphrase itself."""
    subprocess.run(
        [_tool("ssh-add"), key], env={**os.environ, "SSH_AUTH_SOCK": socket}, check=False
    )


def _kept() -> list[str]:
    """The keys whose passphrases the keystore holds."""
    try:
        return _KEPT.read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return []


def _keystore(key: str) -> str | None:
    """`key`'s passphrase as the keystore keeps it, None when it keeps none or cannot be read."""
    try:
        return keyring.get_password(_KEYSTORE, key)
    except _REFUSED as refusal:
        logger.warning("the system keystore could not be read for {}: {}", key, refusal)
        return None


def _add(key: str, passphrase: str, socket: str) -> bool:
    """Whether the agent at `socket` took `key`, its passphrase answered once by `_ASKPASS`."""
    _ASKPASS.write_text(_ANSWER, newline="\n")
    _ASKPASS.chmod(0o700)
    asked = CONTROL / f"asked-{os.getpid()}"
    asked.unlink(missing_ok=True)
    try:
        added = subprocess.run(
            [_tool("ssh-add"), key],
            env={
                **os.environ,
                "SSH_AUTH_SOCK": socket,
                "SSH_ASKPASS": _ASKPASS.as_posix(),
                "SSH_ASKPASS_REQUIRE": "force",
                _PASSPHRASE: passphrase,
                _ASKED: asked.as_posix(),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
    finally:
        asked.unlink(missing_ok=True)
    return added.returncode == 0


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
    """Whether a shared login to `host` answers, clearing the socket one that died left behind.

    Only a socket nothing listens on is cleared (a new master would refuse to replace it), never
    one whose check failed otherwise, which would cut a living login off from every later ssh.
    """
    socket = CONTROL / f"mb-{host}"
    if not socket.exists():
        return False
    checked = subprocess.run(
        [str(client()), *_control(host), "-O", "check", host],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if "connection refused" in checked.stderr.lower():
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
