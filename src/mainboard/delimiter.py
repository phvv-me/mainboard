# Where a verb's own options end and the command it hands on begins.
#
# `run`, `submit` and `shell` take another program's argv. Following `uv run` and `docker run`,
# this tool's options come first and the first token that is not one of them starts the command,
# passed through verbatim from there, so `mainboard run pytest --noconftest` needs no `--`.
#
# The delimiter is placed rather than the parser loosened. Letting the command parameter swallow
# leading hyphens folded an option this tool does not know into the user's command instead of
# refusing it, and four jobs failed on a remote host minutes later that way (2026-08-25). So the
# scan walks only the options the verb declares, each with the number of values it takes, and
# stops at the first token that is neither: a word is the command and gets its `--`, while an
# unknown option is left in place for the parser to refuse by name.
#
# A `--` further on is the command's own (`proc timeout 900 mb job submit --on gold -- python
# x.py`, `git log -- path`) and travels with it, except in the one spelling where it ends this
# verb's options written after a job target: `job submit x.py::t --on gold -- --fresh a`.

from inspect import Parameter, signature
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cyclopts import App

# What ends this tool's options by hand, and what the placement inserts.
DELIMITER = "--"

# The variadic positionals read verbatim, another program's argv or a help query. Any other, such
# as `lint`'s paths, is an ordinary positional the parser lets options follow.
TAILS = frozenset({"command", "query"})


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
        """
        _, apps, rest = self.app.parse_commands(tokens)
        tail = _tail(apps[-1])
        if tail is None:
            return list(tokens)
        verb = list(tokens[: len(tokens) - len(rest)])
        return [*verb, *_delimited(rest, tail)]


class _Tail(NamedTuple):
    """What a trailing-command verb reads for itself ahead of the command it hands on.

    widths: every option name it declares and how many values each takes.
    leading: how many positional arguments come before the command, `proc timeout`'s seconds.
    """

    widths: Mapping[str, int]
    leading: int


def _tail(verb: App) -> _Tail | None:
    """What `verb` reads ahead of its trailing command, None for a verb that takes none.

    A verb takes a trailing command exactly when its function collects a variadic positional
    named in `TAILS`, which is how `run`, `submit`, `shell`, `proc timeout` and `help` spell
    their tails. A negative flag takes no value whatever its positive spelling takes.
    """
    command = verb.default_command
    if command is None:
        return None
    kinds = [
        (parameter.kind, parameter.name) for parameter in signature(command).parameters.values()
    ]
    if not any(kind is Parameter.VAR_POSITIONAL and name in TAILS for kind, name in kinds):
        return None
    widths = dict.fromkeys((*verb.help_flags, *verb.version_flags), 0)
    for argument in verb.assemble_argument_collection():
        if argument.field_info.kind is Parameter.VAR_POSITIONAL:
            continue
        taken, _ = argument.token_count()
        positive = set(argument.parameter.name or ())
        widths.update({name: taken if name in positive else 0 for name in argument.names})
    leading = [kind for kind, _ in kinds].index(Parameter.VAR_POSITIONAL)
    return _Tail(widths, leading)


def _delimited(rest: Sequence[str], tail: _Tail) -> list[str]:
    """`rest` with the delimiter in front of its first command token, when it has one.

    rest: the verb's own tokens, options and leading arguments first.
    """
    at = 0
    leading = tail.leading
    while at < len(rest):
        token = rest[at]
        if token == DELIMITER:
            break
        if _option(token):
            name, inline, _ = token.partition("=")
            if name not in tail.widths:
                break
            at += 1 if inline else 1 + tail.widths[name]
        elif leading:
            leading -= 1
            at += 1
        elif _delimits_itself(rest[at:], tail.widths):
            break
        else:
            return [*rest[:at], DELIMITER, *rest[at:]]
    return list(rest)


def _option(token: str) -> bool:
    return token.startswith("-") and token != "-"


def _delimits_itself(command: Sequence[str], widths: Mapping[str, int]) -> bool:
    """Whether a `--` further on in `command` is this verb's own: `target --on host -- args`.

    Only when the tokens before it hold options of this verb and no other. Otherwise the `--`
    belongs to the command being handed on, which keeps it with its own flags: `proc timeout 900
    mb job submit --on gold -- python train.py` once died on `--on`, read as this tool's.
    """
    if DELIMITER not in command:
        return False
    own = command[: command.index(DELIMITER)]
    options = [token.partition("=")[0] for token in own if _option(token)]
    return bool(options) and all(name in widths for name in options)
