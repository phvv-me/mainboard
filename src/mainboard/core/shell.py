import os
import shlex
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from string.templatelib import Interpolation, Template
from typing import TYPE_CHECKING, NoReturn

from plumbum import FG, ProcessExecutionError

if TYPE_CHECKING:
    from plumbum.commands.base import BaseCommand


def foreground(command: BaseCommand) -> int:
    """Run a bound command with inherited stdio, returning its exit code.

    The one way this workspace runs a command it wants to watch rather than capture, local or
    over an open connection alike, since a bound command carries its own machine either way.
    """
    try:
        command & FG
    except ProcessExecutionError as error:
        return int(error.retcode or 1)
    else:
        return 0


def sh(template: Template) -> str:
    """A shell line from a t-string, every interpolation passed through `shlex.quote`.

    `sh(t"cd {root} && {command}")` keeps a hostile path or argument inside its word, and a plain
    string is a `TypeError`, so unquoted composition is unrepresentable at the call site.
    """
    return _render(template, lambda item: shlex.quote(str(item.value)))


def script(template: Template) -> str:
    """A shell fragment from a t-string, interpolations landed verbatim.

    For composing trusted, already-quoted fragments (a `sh` result, a rendered activation snippet)
    into a larger line without quoting them twice.
    """
    return _render(template, lambda item: str(item.value))


def _render(template: Template, convert: Callable[[Interpolation], str]) -> str:
    if not isinstance(template, Template):
        raise TypeError(
            f"expected a t-string, got {type(template).__name__}; "
            'write t"..." so interpolations stay quotable'
        )
    return "".join(convert(item) if isinstance(item, Interpolation) else item for item in template)


def become(program: str, argv: list[str], env: Mapping[str, str] | None = None) -> NoReturn:
    """Replace this process with `program` and `argv`, the terminal and its signals included."""
    if env is None:
        os.execvp(program, argv)
    os.execvpe(program, argv, dict(env))


# Shells that need `-i` to read their interactive startup when not started by a terminal.
_POSIX_SHELLS = frozenset({"zsh", "bash", "fish", "sh", "dash", "ksh"})


def interactive_shell() -> list[str]:
    """The user's own interactive shell as an argv: `$SHELL` where this machine runs it, else
    `sh`."""
    named = os.environ.get("SHELL", "")
    if named and (found := shutil.which(named) or shutil.which(Path(named).name)):
        return [found, "-i"] if Path(found).stem in _POSIX_SHELLS else [found]
    return ["/bin/sh", "-i"]
