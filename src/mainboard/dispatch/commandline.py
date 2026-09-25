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

import shlex
from typing import TYPE_CHECKING

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Sequence

# What only a shell can act on. A token carrying one of these means something a program name and
# its arguments cannot mean, which is what tells a quoted-up shell line apart from plain argv.
# Globs and braces are deliberately absent: a glob is a legitimate argument to pass through
# unexpanded, so treating one as shell syntax would wrap commands that work today.
_SHELL_SYNTAX = frozenset("|&;<>()$`\n")


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
    if len(tokens) == 1 and needs_shell(tokens[0]):
        return shlex.join(["bash", "-c", vetted(tokens[0])])
    return vetted(shlex.join(tokens))


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
