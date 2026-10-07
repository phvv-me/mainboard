# The ssh client this tool runs. Everything else about reaching a host belongs to the user's ssh
# config: keys and their passphrases (an agent; on macOS `UseKeychain` keeps a passphrase in the
# Keychain), and shared logins (`ControlMaster`, `ControlPersist`). A host that asks for a one-time
# code on every login (miyabi-g) is reached by logging in once with `ssh <host>`; every later ssh,
# this tool's included, rides that login until it persists no longer.
#
# On Windows, Microsoft's build has no multiplexing and fails outright on a config that asks for
# it, so this tool runs Git's or MSYS2's ssh whenever one is installed.

import os
from functools import cache
from pathlib import Path
from shutil import which

from ..core.errors import MissionError
from ..core.host import WINDOWS


@cache
def client() -> Path:
    """The ssh this tool runs: on Windows a Git or MSYS2 build ahead of Microsoft's."""
    found = which("ssh")
    candidates = [*(_posix_builds() if WINDOWS else ()), *([Path(found)] if found else [])]
    if not candidates:
        raise MissionError("no ssh on PATH; install an OpenSSH client")
    return candidates[0]


def microsoft() -> bool:
    """Whether the ssh this tool runs is Microsoft's Windows build rather than Git's or MSYS2's."""
    return WINDOWS and "usr" not in client().parts


def _posix_builds() -> list[Path]:
    """Git's and MSYS2's ssh on this Windows machine, those on PATH first, then Git's bundled one
    (Git puts only its `cmd` folder on PATH)."""
    folders = [Path(folder) for folder in os.environ.get("PATH", "").split(os.pathsep) if folder]
    git = which("git")
    if git:
        folders += [parent / "usr" / "bin" for parent in Path(git).parents]
    builds = (folder / "ssh.exe" for folder in folders if "usr" in folder.parts)
    return [build for build in builds if build.is_file()]
