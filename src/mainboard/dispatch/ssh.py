# The ssh client this tool runs. Everything else about reaching a host belongs to the user's ssh
# config: keys and their passphrases (an agent; on macOS `UseKeychain` keeps a passphrase in the
# Keychain), and shared logins (`ControlMaster`, `ControlPersist`). A host that asks for a one-time
# code on every login (miyabi-g) is reached by logging in once with `ssh <host>`; every later ssh,
# this tool's included, rides that login until it persists no longer.

from functools import cache
from pathlib import Path
from shutil import which

from ..core.errors import MissionError


@cache
def client() -> Path:
    """The ssh on PATH."""
    found = which("ssh")
    if found is None:
        raise MissionError("no ssh on PATH; install an OpenSSH client")
    return Path(found)
