# Where a verb's own options end and the command it hands on begins.
#
# `run`, `submit` and `interact` take another program's argv, and they used to need `--` in front
# of it: `mainboard run pytest --noconftest` read `--noconftest` as an option of this tool and
# refused it, so every agent session learned to type the delimiter or failed once first. The rule
# `uv run` and `docker run` follow is the one a reader already expects: this tool's options come
# first, and the first token that is not one of them starts the command, which from there on is
# passed through verbatim, delimiters and flags included.
#
# The delimiter is placed rather than the parser loosened. Letting the command parameter swallow
# leading hyphens folded an option this tool does not know into the user's command instead of
# refusing it, and four jobs failed on a remote host minutes later that way (2026-08-25). So the
# scan below walks only the options the verb actually declares, each with the number of values it
# takes, and stops at the first token that is neither: a word is the command and gets its `--`,
# while an unknown option is left in place for the parser to refuse by name.

from inspect import Parameter, signature
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cyclopts import App

# What ends this tool's options by hand, and what the placement inserts.
DELIMITER = "--"


class Delimiter:
    """Places the `--` a trailing-command verb implies, so its command never has to spell one."""

    def __init__(self, app: App) -> None:
        """app: the root application whose verbs the argv names."""
        self.app = app

    def placed(self, tokens: Sequence[str]) -> list[str]:
        """`tokens` with `--` before the first command token of a trailing-command verb.

        Anything else comes back unchanged: a verb with no trailing command, an argv that already
        delimits itself, one that names no command at all, and one whose options include a name
        the verb does not declare, which the parser then refuses with that name.

        tokens: the argv after the program name.
        """
        _, apps, rest = self.app.parse_commands(tokens)
        widths = _widths(apps[-1])
        if widths is None:
            return list(tokens)
        verb = list(tokens[: len(tokens) - len(rest)])
        return [*verb, *_delimited(rest, widths)]


def _widths(verb: App) -> Mapping[str, int] | None:
    """Every option name `verb` declares and how many values it takes, None without a command.

    A verb takes a trailing command exactly when its function collects a variadic positional,
    which is how `run`, `submit`, `interact` and `help` spell their tails. A negative flag takes
    no value whatever its positive spelling takes.
    """
    command = verb.default_command
    if command is None:
        return None
    kinds = [parameter.kind for parameter in signature(command).parameters.values()]
    if Parameter.VAR_POSITIONAL not in kinds:
        return None
    widths = dict.fromkeys((*verb.help_flags, *verb.version_flags), 0)
    for argument in verb.assemble_argument_collection():
        if argument.field_info.kind is Parameter.VAR_POSITIONAL:
            continue
        taken, _ = argument.token_count()
        positive = set(argument.parameter.name or ())
        widths.update({name: taken if name in positive else 0 for name in argument.names})
    return widths


def _delimited(rest: Sequence[str], widths: Mapping[str, int]) -> list[str]:
    """`rest` with the delimiter in front of its first command token, when it has one.

    rest: the verb's own tokens, options first.
    widths: the verb's option names and the values each takes.
    """
    at = 0
    while at < len(rest):
        token = rest[at]
        if token == DELIMITER:
            break
        if not token.startswith("-") or token == "-":
            return [*rest[:at], DELIMITER, *rest[at:]]
        name, inline, _ = token.partition("=")
        if name not in widths:
            break
        at += 1 if inline else 1 + widths[name]
    return list(rest)
