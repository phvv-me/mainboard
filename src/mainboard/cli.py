__lazy_modules__ = [
    "mainboard.apple",
    "mainboard.core.errors",
    "mainboard.core.execution.modules",
    "mainboard.core.project",
    "mainboard.dispatch.schedulers",
    "mainboard.log",
    "mainboard.profile",
    "mainboard.render",
    "mainboard.verbs.constants",
    "mainboard.verbs.help",
    "mainboard.verbs.node",
]

import os
import sys
from contextlib import ExitStack
from inspect import Parameter as Signed
from inspect import signature
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, NoReturn

from .apple import Apple
from .core.errors import MissionError, NoWorkspace
from .core.execution.modules import importable, submodules
from .core.project import Project
from .dispatch.schedulers import HostUnreachable
from .log import configure, logger
from .profile import Feature, Profiler
from .render import Audience, Terminal
from .verbs.constants import TREE
from .verbs.help import Root
from .verbs.node import Typed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cyclopts import App


# Entry points and the standard-library agents are left out of a profile: importing them runs
# them.
_UNPROFILED = ("__main__", "agent.program", "center.remote", ".profile")

# The exit status of a command line naming no command, cyclopts' own for a usage error.
_USAGE = 2

# What joins a job target's file and the name inside it, `file.py::name`.
_SEPARATOR = "::"

# What ends this tool's options by hand, and what the placement inserts.
_DELIMITER = "--"

# The variadic positionals read verbatim, another program's argv or a help query. Any other, such
# as `lint`'s paths, is an ordinary positional the parser lets options follow.
_TAILS = frozenset({"command", "query"})


def main() -> None:
    """Console entry point, `MissionError` printed to stderr without a traceback, exit 1.

    An alias is spelled as the command it stands for, a person told the new spelling in one dim
    line, and a command line naming no command is answered from `TREE` with the path it moved to
    or the nearest commands before any verb module is imported. A trailing-command verb gets the
    `--` its command implies, so nothing typed after the command is ever read as this tool's.
    The verbs a person or an agent types log short plain lines; the ones a dispatched job runs
    keep the JSON its log readers parse.
    """
    terminal = Terminal.here()
    _prepare(terminal)
    argv = _spelled(terminal, sys.argv[1:])
    if refusal := Typed(node=TREE).refusal(argv):
        logger.error(refusal)
        raise SystemExit(_USAGE)
    try:
        _run(terminal, argv)
    # An exit passes untouched: matched against the tuple below, it would import the dispatch
    # stack for `HostUnreachable`, which took `--help` from 0.17 s to 0.47 s on Python 3.15.
    except SystemExit:
        raise
    except BrokenPipeError:
        _closed()
    except (MissionError, NoWorkspace, HostUnreachable) as error:
        _refused(error)


def _closed() -> NoReturn:
    """End quietly on a reader that went away, before Python's shutdown flush tries again."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        os.dup2(sink.fileno(), sys.stdout.fileno())
    raise SystemExit(0) from None


def _prepare(terminal: Terminal) -> None:
    """Ready this process to print and to build.

    Streams never fail on a character, the Xcode in use is selected, every verb but the one
    a dispatched job runs logs short plain lines, and logs are colored for a person only.
    """
    # A job's log carries whatever it printed (a progress bar's block characters), which a pipe
    # in a narrower encoding cannot print; one such character killed `job logs` mid-print
    # (2026-09-29).
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(errors="replace")
    Apple.select_xcode()
    line = not Project().variable("LOG_FORMAT").read() and sys.argv[1:2] != ["execute"]
    configure(output="line" if line else "", colors=terminal.styled(stderr=True))


def _profiled(report: Path, app: App, argv: Sequence[str]) -> None:
    """Run `argv` through `app` under this tool's own profiler, its span report in `report`.

    Every one of its modules is instrumented, so `MB_PROFILE=prof.txt mb job submit ...` names the
    slow function of any verb, on any machine, with nothing else installed. A module of an optional
    extra that is not installed is left out and named in the log.
    """
    package = __name__.partition(".")[0]
    modules = importable(
        name
        for name in submodules(package)
        if name != package and not any(skip in name for skip in _UNPROFILED)
    )
    profiler = Profiler(features=Feature.SPANS, gpus=(), auto=modules)
    with ExitStack() as finish:
        finish.callback(_reported, profiler, report)
        with profiler:
            app(argv)


def _refused(error: Exception) -> NoReturn:
    """End on a refusal with its one sentence, never a traceback.

    A host that will not answer is an answer, not a crash: it used to end a verb as a traceback
    forty lines long with the one sentence that mattered last.
    """
    logger.error(str(error))
    raise SystemExit(1) from None


def _reported(profiler: Profiler, report: Path) -> None:
    """Write the profiler's span report to `report`, and say where."""
    report.write_text(profiler.result().report(), encoding="utf-8", newline="\n")
    logger.info(f"profile written to {report}")


def _run(terminal: Terminal, typed: Sequence[str]) -> None:
    """Run the verb `typed` names, under the profiler when `MB_PROFILE` asks.

    An owner-scoped verb refuses an agent first, and only the verb module `typed` names is
    imported. The meta app runs the line, so a root option answers before any command word.
    """
    if owned := Typed(node=TREE).owned(typed):
        terminal.refuse_agents(owned)
    with terminal.installed():
        app = Root.cli(TREE, typed)
        argv = Delimiter(app).placed(typed)
        if profiled := Project().variable("PROFILE").read():
            _profiled(Path(profiled), app.meta, argv)
        else:
            app.meta(argv)


def _spelled(terminal: Terminal, typed: Sequence[str]) -> list[str]:
    """`typed` with a leading alias spelled as its path; a person is told so in one dim line."""
    argv = TREE.rewritten(typed)
    if argv != list(typed) and terminal.audience is Audience.HUMAN:
        said = [" ".join((TREE.name, *words)) for words in (typed, argv)]
        terminal.error_console().print(f"`{said[0]}` is spelled `{said[1]}` now", style="dim")
    return argv


class Delimiter:
    """Places the `--` a trailing-command verb implies, so its command never has to spell one.

    `run`, `submit` and `shell` take another program's argv. Following `uv run` and `docker run`,
    this tool's options come first and the first token that is not one of them starts the
    command, passed through verbatim from there, so `mainboard run pytest --noconftest` needs no
    `--`.

    The delimiter is placed rather than the parser loosened. Letting the command parameter swallow
    leading hyphens folded an option this tool does not know into the user's command instead of
    refusing it, and four jobs failed on a remote host minutes later that way (2026-08-25). So the
    scan walks only the options the verb declares, each with the number of values it takes, and
    stops at the first token that is neither: a word is the command and gets its `--`, while an
    unknown option is left in place for the parser to refuse by name.

    A `--` further on is the command's own (`proc timeout 900 mb job submit --on gold -- python
    x.py`, `git log -- path`) and travels with it, except in the one spelling where it ends this
    verb's options written after a job target: `job submit x.py::t --on gold -- --fresh a`.
    """

    @staticmethod
    def _command_at(rest: Sequence[str], tail: _Tail) -> int | None:
        """Where the command in `rest` starts, None when it delimits itself or names none.

        It is None too at an option the verb does not declare, which the parser then refuses.
        """
        position, leading = 0, tail.leading
        while position < len(rest) and rest[position] != _DELIMITER:
            token = rest[position]
            option = _is_option(token)
            if not option and not leading:
                command = rest[position:]
                return None if Delimiter._is_delimits_itself(command, tail.widths) else position
            width = Delimiter._skipped(token, tail.widths) if option else 1
            if width is None:
                return None
            leading -= not option
            position += width
        return None

    @staticmethod
    def _delimited(rest: Sequence[str], tail: _Tail) -> list[str]:
        """`rest` with the delimiter in front of its first command token, when it has one.

        rest: the verb's own tokens, options and leading arguments first.
        """
        start = Delimiter._command_at(rest, tail)
        return list(rest) if start is None else [*rest[:start], _DELIMITER, *rest[start:]]

    @staticmethod
    def _is_delimits_itself(command: Sequence[str], widths: Mapping[str, int]) -> bool:
        """Whether a `--` further on in `command` is this verb's own: `target --on host -- args`.

        Only when the command is a job target (`file.py` or `file.py::name`) and the tokens before
        the `--` hold options of this verb and no other. Otherwise the `--` belongs to the command
        being handed on, which keeps it with its own flags: `proc timeout 900 mb job submit --on
        gold -- python train.py` once died on `--on`, read as this tool's, and `run tool --help --
        x` showed this tool's help instead of the tool's.
        """
        target = command[0].partition(_SEPARATOR)[0]
        if _DELIMITER not in command or not target.endswith(".py"):
            return False
        own = command[: command.index(_DELIMITER)]
        options = [token.partition("=")[0] for token in own if _is_option(token)]
        return bool(options) and all(name in widths for name in options)

    @staticmethod
    def _skipped(option: str, widths: Mapping[str, int]) -> int | None:
        """The tokens `option` spans with its values, None for one the verb does not declare."""
        name, inline, _ = option.partition("=")
        if name not in widths:
            return None
        return 1 if inline else 1 + widths[name]

    @staticmethod
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
            (parameter.kind, parameter.name)
            for parameter in signature(command).parameters.values()
        ]
        if not any(kind is Signed.VAR_POSITIONAL and name in _TAILS for kind, name in kinds):
            return None
        widths = dict.fromkeys((*verb.help_flags, *verb.version_flags), 0)
        for argument in verb.assemble_argument_collection():
            if argument.field_info.kind is Signed.VAR_POSITIONAL:
                continue
            taken, _ = argument.token_count()
            positive = set(argument.parameter.name or ())
            widths.update({name: taken if name in positive else 0 for name in argument.names})
        leading = [kind for kind, _ in kinds].index(Signed.VAR_POSITIONAL)
        return _Tail(widths, leading)

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
        tail = Delimiter._tail(apps[-1])
        if tail is None:
            return list(tokens)
        verb = list(tokens[: len(tokens) - len(rest)])
        return [*verb, *Delimiter._delimited(rest, tail)]


class _Tail(NamedTuple):
    """What a trailing-command verb reads for itself ahead of the command it hands on.

    widths: every option name it declares and how many values each takes.
    leading: how many positional arguments come before the command, `proc timeout`'s seconds.
    """

    widths: Mapping[str, int]
    leading: int


def _is_option(word: str) -> bool:
    return word.startswith("-") and word != "-"
