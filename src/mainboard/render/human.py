from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from .values import columns_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .values import Row


def render_table(
    rows: Sequence[Row], *, fields: Sequence[str] | None = None, title: str = ""
) -> None:
    """Print `rows` as a rich table, one line per record, columns projected to `fields`.

    Every cell is data rather than prose, so rich's console markup is off: a value in square
    brackets is a manifest table heading like `[dev.python.deps]`, and rich reads that as a
    style tag and renders the cell empty. Highlighting stays on, since colouring a version or a
    path changes how a value looks and never whether it is shown.

    rows: the records to render, each a flat field-name to value mapping.
    fields: the column names to keep, every key from the first row when None.
    title: the table's heading, untitled when empty.
    """
    columns = columns_of(rows, fields)
    table = Table(title=title or None)
    for column in columns:
        table.add_column(column)
    for row in rows:
        cells = (row.get(column) for column in columns)
        table.add_row(*("" if cell is None else str(cell) for cell in cells))
    Console(markup=False).print(table)


@contextmanager
def progress(description: str) -> Iterator[Callable[[str], None]]:
    """A stderr progress reporter around a block of unknown duration.

    Yields the label setter, so a block that reaches several stages says which one it is on
    instead of standing still under one description; a caller with nothing to report ignores it.
    An interactive terminal gets the transient spinner this always was. Anywhere else, a log
    file, a background job, a CI runner, `Console.status` answers every `.update()` with silence
    and renders once at the very end, so a multi-stage onboarding piped to a log stood
    indistinguishable from a hang until the caller killed it; there each stage prints as its own
    line instead, since a stage worth naming to a live viewer is a stage worth naming in the log
    nobody is watching live.

    description: the label shown first, and beside the spinner on a terminal.
    """
    console = Console(stderr=True, markup=False)
    if not console.is_terminal:
        console.print(description)
        yield console.print
        return
    with console.status(description) as status:
        yield status.update


def install_traceback() -> None:
    """Install rich's traceback handler for readable uncaught errors, the CLI error boundary."""
    install_rich_traceback(show_locals=False)
