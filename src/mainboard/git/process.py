import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

from ..core.errors import MissionError

# Settings every call carries. A CRLF conversion warning is advice about a checkout on another
# platform, printed once per file on every `add`, and it buried the one line that mattered in a
# tree this size. Nothing here is read as colour or as a pager, so neither is ever switched on.
# Git never recurses into submodules on its own: the tree walks every repository itself, in the
# order a pointer needs, and git's own recursion would fetch, check out or push the foreign ones
# too, and fail a parent's fetch over a submodule remote that cannot serve one pointer.
_QUIET = (
    "core.safecrlf=false",
    "color.ui=never",
    "core.pager=cat",
    "submodule.recurse=false",
    "fetch.recurseSubmodules=false",
    "push.recurseSubmodules=no",
)

# No prompt, since a verb walking thirty repositories would hang on the first credential it
# lacked. No optional locks, since `status` otherwise rewrites the index of a tree it only reads,
# and a second git working there at the same moment finds `index.lock` and fails.
_ENVIRONMENT = {"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}

# A fetch or a push waits on a network and on a remote that may never answer, so it is bounded.
# Ten minutes is past any push this tree has made, including an LFS upload.
_NETWORK_SECONDS = 600.0

# The exit code `timeout(1)` uses, so a bounded call that ran out reads the way it does in a shell.
_TIMED_OUT = 124

# GitHub over HTTPS asks for a credential, and the one tool on every center that already holds a
# GitHub login is `gh`. Its helper is appended after whatever this machine configures, so a
# working keychain still answers first and `gh` is only asked when nothing else could.
_GITHUB_HELPER = "credential.https://github.com.helper"


class Git:
    """Git run in one working tree, captured and never prompting."""

    @staticmethod
    def _credential() -> tuple[str, ...]:
        """The `gh` credential helper appended for a GitHub call, nothing when `gh` is absent."""
        if (found := shutil.which("gh")) is None:
            return ()
        return ("-c", f'{_GITHUB_HELPER}=!"{Path(found).as_posix()}" auth git-credential')

    def __init__(self, path: Path) -> None:
        self.path = path

    def line(self, *args: str) -> str:
        """The one line a query that has to succeed prints, trimmed."""
        return self.out(*args).strip()

    def ok(self, *args: str) -> bool:
        """Whether a yes-or-no query answered yes, which git says with exit code zero."""
        return self.run(*args).returncode == 0

    def out(self, *args: str, stdin: str = "") -> str:
        """The output of a call that has to succeed, refused with git's own words when not."""
        result = self.run(*args, stdin=stdin)
        if result.returncode:
            raise MissionError(f"git {' '.join(args)} failed in {self.path}: {said(result)}")
        return result.stdout

    def run(
        self, *args: str, stdin: str = "", network: bool = False, errors: str = "replace"
    ) -> subprocess.CompletedProcess[str]:
        """Run `git args` here, whatever it exits with.

        stdin: text fed to the command, for a `--pathspec-from-file=-`.
        network: bound the call and offer the `gh` credential, for a fetch, push or clone.
        errors: how bytes that are no UTF-8 decode, `surrogateescape` to keep a path whole.
        """
        argv = [_executable(), "-C", str(self.path)]
        argv += [word for setting in _QUIET for word in ("-c", setting)]
        if network:
            argv.extend(Git._credential())
        argv.extend(args)
        try:
            return subprocess.run(
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors=errors,
                env={**os.environ, **_ENVIRONMENT},
                timeout=_NETWORK_SECONDS if network else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            silent = f"git {args[0]} gave no answer in {_NETWORK_SECONDS:.0f} s"
            return subprocess.CompletedProcess(argv, _TIMED_OUT, "", silent)


def said(result: subprocess.CompletedProcess[str]) -> str:
    """What git said went wrong: the remote's last word if it spoke, else git's last line.

    A refused push ends on `failed to push some refs`, which is true and says nothing; the
    reason is what the remote printed above it, prefixed `remote:`.
    """
    lines = [line.strip() for line in (result.stderr + result.stdout).splitlines()]
    remote = [line.removeprefix("remote:").strip() for line in lines if line.startswith("remote:")]
    spoken = [line for line in remote if line] or [line for line in lines if line]
    return spoken[-1] if spoken else f"exit {result.returncode}"


@cache
def _executable() -> str:
    """The git on PATH, refused by name when there is none."""
    if (found := shutil.which("git")) is None:
        raise MissionError("git is not on PATH; install it before operating the repository tree")
    return found
