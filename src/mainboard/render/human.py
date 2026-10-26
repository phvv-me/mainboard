import os
import sys
import time
from contextlib import ExitStack, contextmanager, redirect_stdout
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from ..log import logger
from .values import columns_of

_UNBOUNDED = 1 << 16
# A block shorter than this prints no closing total.
_WORTH_A_TOTAL = 2.0
_STDOUT_FD = 1
_STDERR_FD = 2

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

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
    with console.capture() as captured:
        console.print(table)
    logger.result(captured.get(), end="")


@contextmanager
def progress(description: str) -> Generator[Callable[[str], None]]:
    """A stderr progress reporter around a block of unknown duration, yielding the stage setter.

    Each stage prints as its own line, led by the seconds since the block began, and a block
    that took a while ends with its total, so the slow stage of a setup or a dispatch is named in
    every log without anyone profiling it (a pin spent a minute in regex unnoticed that way).
    Stdout is `diverted` for the whole block.
    """
    start = time.monotonic()

    def stage(line: str) -> None:
        logger.info(f"{time.monotonic() - start:6.1f}s {line}")

    with diverted():
        stage(description)
        yield stage
    if (total := time.monotonic() - start) >= _WORTH_A_TOTAL:
        logger.info(f"{total:6.1f}s done: {description}")


@contextmanager
def diverted() -> Generator[None]:
    """Send everything written to stdout inside the block to stderr instead.

    Stdout carries only the document a verb prints when done, so `--json` parses whole. The
    chatter of SDKs, transfers and libraries is diverted at the descriptor as well as at
    `sys.stdout`, so child processes are silenced on stdout too.
    """
    sys.stdout.flush()
    held = os.dup(_STDOUT_FD)
    os.dup2(_STDERR_FD, _STDOUT_FD)
    with ExitStack() as stack:
        stack.callback(_restore, held)
        stack.enter_context(redirect_stdout(sys.stderr))
        yield


def _restore(held: int) -> None:
    """Point stdout back at the descriptor `held` saved, flushing what the block wrote."""
    sys.stdout.flush()
    os.dup2(held, _STDOUT_FD)
    os.close(held)
