# How a caller's argv becomes the one command string a target runs, and the refusal that stops a
# malformed one before it reaches a meter.
#
# Every lane interpolates that string into a shell: an ssh host's activated `bash -lc` line, vast's
# `bash -c` container entrypoint, hpc-ai's initScript, modal's sandbox `bash -c`. `shlex.join` is
# right for a program and its arguments and wrong for a shell line quoted into one token: it turns
# `cd work && python train.py` into one word, every lane looks for a program by that name, and the
# run exits 127 having done nothing. On owned hardware that costs a scheduler round trip; on a
# rental it costs the whole rental, since the meter starts at boot and never learns the command
# never ran (one campaign lost a rental this way, 2026-08-25).
#
# So, with no flag, a lone token carrying shell syntax is wrapped as `bash -c <token>` (the fix a
# caller used to type by hand), and every line is vetted for what a shell would refuse anyway.

import os
import re
import shlex
import subprocess
from functools import cache
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import MissionError
from ..core.host import WINDOWS
from ..core.project import Project

if TYPE_CHECKING:
    from collections.abc import Sequence

# What only a shell can act on. A token carrying one of these means something a program name and
# its arguments cannot mean, which is what tells a quoted-up shell line apart from plain argv.
# Globs and braces are deliberately absent: a glob is a legitimate argument to pass through
# unexpanded, so treating one as shell syntax would wrap commands that work today.
_SHELL_SYNTAX = frozenset("|&;<>()$`\n")

# A Windows drive path inside an argument, the only shape a rewritten POSIX path takes.
_DRIVE = re.compile(r"[A-Za-z]:[\\/]")

# How long a shell's own `cygpath` may take to name its root.
_CYGPATH_SECONDS = 10.0


def needs_shell(token: str) -> bool:
    """Whether the argv `token` is a shell program in its own right rather than one plain word."""
    return any(character in _SHELL_SYNTAX for character in token)


def joined(tokens: Sequence[str]) -> str:
    """`tokens` as the one command line a target runs, a lone shell program wrapped for a shell.

    A program and its arguments arrive as several tokens and are shell-quoted, so an argument
    that merely contains a semicolon (`python -c 'a; b'`) stays text. A quoted shell line arrives
    as one token. The token count is the only thing that can tell the two apart, since nothing
    here knows which programs the far side has.
    """
    _unconverted(tokens)
    if len(tokens) == 1 and needs_shell(tokens[0]):
        return shlex.join(["bash", "-c", vetted(tokens[0])])
    return vetted(shlex.join(tokens))


def _unconverted(tokens: Sequence[str]) -> None:
    """Refuse an argument the calling shell rewrote from a POSIX path into a Windows one.

    Git Bash and MSYS2 turn every argument that looks like an absolute POSIX path into a path
    under their own install folder before a native program sees it, so `--out-dir
    /home/crimson/cache/y` reached a Linux host as `C:/Program Files/Git/home/crimson/cache/y`
    and an hour-long GPU job wrote its results under that literal name (2026-09-30).
    """
    roots = _shell_roots() if any(_DRIVE.search(token) for token in tokens) else ()
    for token, root in product(tokens, roots):
        before, found, meant = token.replace("\\", "/").partition(root)
        if found:
            raise MissionError(
                f"the argument {token!r} is not what was typed: this shell rewrote "
                f"`{before}/{meant}` into a path under its own folder {root} before "
                f"{Project().name} saw it. Run the command again with `MSYS_NO_PATHCONV=1` in "
                "front (Git Bash; `MSYS2_ARG_CONV_EXCL='*'` in MSYS2), which hands every "
                "argument on as typed."
            )


@cache
def _shell_roots() -> tuple[str, ...]:
    """The Windows folders the MSYS shells on this process's PATH place `/` at, `C:/Program
    Files/Git/` for Git Bash; none off Windows, outside such a shell, and when the caller
    already switched its path conversion off.

    Every `cygpath` on PATH answers for its own shell, since the first is not always the one
    that rewrote the arguments: under `mb run` the environment's own MSYS tools lead PATH.
    """
    off = "MSYS_NO_PATHCONV" in os.environ or os.environ.get("MSYS2_ARG_CONV_EXCL") == "*"
    if not WINDOWS or "MSYSTEM" not in os.environ or off:
        return ()
    folders = os.environ.get("PATH", "").split(os.pathsep)
    tools = (Path(folder, "cygpath.exe") for folder in folders if folder)
    roots = (_root(tool) for tool in dict.fromkeys(tools) if tool.is_file())
    return tuple(dict.fromkeys(root for root in roots if root))


def _root(cygpath: Path) -> str:
    """Where the MSYS shell owning `cygpath` places `/`, slash-ended; empty when it cannot say."""
    try:
        asked = subprocess.run(
            [str(cygpath), "-m", "/"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_CYGPATH_SECONDS,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return ""
    return f"{asked.stdout.strip().rstrip('/')}/"


def vetted(line: str) -> str:
    """`line` back, refusing here what the far side's shell would refuse after the money is spent.

    Only an empty command and a quote that never closes, faults a shell itself raises on, so
    nothing runnable is turned away; otherwise each surfaces as a billed instance whose command
    exited without running.
    """
    if not line.strip():
        raise MissionError("nothing to run: the command is empty")
    try:
        shlex.split(line)
    except ValueError as unbalanced:
        raise MissionError(
            f"the command {line!r} is not a runnable shell line ({unbalanced}); "
            "close the quote, or pass the line as `-- bash -c '<line>'`"
        ) from None
    return line
