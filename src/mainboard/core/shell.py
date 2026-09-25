import shlex
from collections.abc import Callable
from string.templatelib import Interpolation, Template
from typing import TYPE_CHECKING

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
