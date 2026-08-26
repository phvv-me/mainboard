# How a caller's argv becomes the one command string a target actually runs, and the refusal that
# stops a malformed one before it reaches a meter.
#
# Every lane downstream of here interpolates that string into a shell. An ssh host runs it inside
# the activated `bash -lc` line, vast runs it as the `bash -c` argument of its container
# entrypoint, hpc-ai writes it into an initScript, and modal hands it to `bash -c` in a sandbox.
# The string is therefore a shell program, and joining argv with `shlex.join` is exactly right for
# a program name and its own arguments and exactly wrong for a shell line someone quoted into a
# single token. The quoting turns `cd work && python train.py` into one word, every lane then
# looks for a program by that name, and the run exits 127 having done nothing. On owned hardware
# that costs a scheduler round trip. On a rented instance it costs the whole rental, because the
# meter starts when the machine boots and never learns that the command never ran (one campaign
# lost a rental this way, 2026-08-25).
#
# Two things happen here, and neither is a flag. A lone token carrying shell syntax is wrapped as
# `bash -c <token>`, which is the very fix a caller used to have to type by hand, so shell syntax
# arrives on the far side as shell syntax. And whatever line comes out is vetted for what a shell
# would refuse anyway, an empty command or an unbalanced quote, which costs nothing here and costs
# a rental once it has been dispatched.

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
    """Whether `token` is a shell program in its own right rather than one plain word.

    token: a single argv token as the caller typed it.
    """
    return any(character in _SHELL_SYNTAX for character in token)


def joined(tokens: Sequence[str]) -> str:
    """`tokens` as the one command line a target runs, a lone shell program wrapped for a shell.

    A program and its arguments arrive as several tokens and are shell-quoted, so an argument
    that merely contains a semicolon (`python -c 'a; b'`) keeps it as text. A shell line arrives
    as one token, since that is what quoting it on the command line produces, and is handed to
    `bash -c` instead of being quoted into a single unrunnable word. The token count is what tells
    the two apart, and it is the only thing that can, since nothing here knows which programs the
    far side has.

    tokens: the argv the caller passed, everything after `--`.
    """
    if len(tokens) == 1 and needs_shell(tokens[0]):
        return shlex.join(["bash", "-c", vetted(tokens[0])])
    return vetted(shlex.join(tokens))


def vetted(line: str) -> str:
    """`line` back, refusing here what the far side's shell would refuse after the money is spent.

    Only faults a shell itself would raise on, so nothing runnable is ever turned away: an empty
    command, and a quote that never closes. Both are free to find and both otherwise surface as a
    started, billed instance whose command exited without running.

    line: the assembled command line.
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
