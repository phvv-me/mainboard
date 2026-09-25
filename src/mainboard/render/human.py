import os
import sys
from contextlib import contextmanager, redirect_stdout
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from .values import columns_of

_UNBOUNDED = 1 << 16
_STDOUT_FD = 1
_STDERR_FD = 2

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .values import Row


def render_table(
    rows: Sequence[Row], *, fields: Sequence[str] | None = None, title: str = ""
) -> None:
    """Print `rows` as a rich table, one line per record.

    Markup is off because a cell like `[dev.python.deps]` is data that rich would read as a
    style tag and render empty; highlighting stays on, since it never hides a value.

    fields: the column names to keep, every key from the first row when None.
    """
    columns = columns_of(rows, fields)
    table = Table(title=title or None)
    for column in columns:
        # A handle or digest is only useful whole, so fold rather than cut it to `e775…`.
        table.add_column(column, overflow="fold")
    for row in rows:
        cells = (row.get(column) for column in columns)
        table.add_row(*("" if cell is None else str(cell) for cell in cells))
    console = Console(markup=False)
    if not console.is_terminal:
        # Off a terminal rich assumes eighty columns and shreds a wide table; a pipe has no
        # width, so the table takes its own, measured on a console too wide to clip it.
        natural = Console(markup=False, width=_UNBOUNDED).measure(table).maximum
        console = Console(markup=False, width=natural)
    console.print(table)


@contextmanager
def progress(description: str) -> Iterator[Callable[[str], None]]:
    """A stderr progress reporter around a block of unknown duration, yielding the stage setter.

    A terminal gets a transient spinner. Off a terminal `Console.status` stays silent until the
    end, which reads as a hang in a log, so each stage prints as its own line instead. Stdout is
    `diverted` for the whole block.
    """
    console = Console(stderr=True, markup=False)
    with diverted():
        if not console.is_terminal:
            console.print(description)
            yield console.print
            return
        with console.status(description) as status:
            yield status.update


@contextmanager
def diverted() -> Iterator[None]:
    """Send everything written to stdout inside the block to stderr instead.

    Stdout carries only the document a verb prints when done, so `--json` parses whole. The
    chatter of SDKs, transfers and libraries is diverted at the descriptor as well as at
    `sys.stdout`, so child processes are silenced on stdout too.
    """
    sys.stdout.flush()
    held = os.dup(_STDOUT_FD)
    os.dup2(_STDERR_FD, _STDOUT_FD)
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(held, _STDOUT_FD)
        os.close(held)


def install_traceback() -> None:
    """Install rich's traceback handler for readable uncaught errors, the CLI error boundary."""
    install_rich_traceback(show_locals=False)
