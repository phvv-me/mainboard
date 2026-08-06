"""A swept study rendered as a small multiple.

A surface over several axes has no single plot, so the view is one panel per facet with the
swept axis running along each row. Read across a row to see how a measure moves with one axis
and down the panels to see whether another axis changes that, which is the question a single
line cannot answer and the reason a sweep exists at all.

Rendered in the terminal because that is where the rest of mainboard reports and because a
sweep is read while it is being reasoned about, not filed. A sparkline carries the shape and the
numbers beside it carry the magnitude, which together are what a chart would have said.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rich.console import RenderableType

    from .study import Point, Row

# Eight levels is what a sparkline can say without pretending to be a chart.
_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float]) -> str:
    """Return one line showing the shape of `values`, scaled to their own range.

    Scaled to the row rather than to the whole surface, because the question a row answers is
    how a measure moves along one axis. Comparing magnitudes between rows is what the printed
    numbers are for.
    """
    usable = [value for value in values if value > 0]
    if not usable:
        return " " * len(values)
    low, high = min(usable), max(usable)
    span = high - low
    return "".join(
        " "
        if value <= 0
        else _BLOCKS[0]
        if span == 0
        else _BLOCKS[min(int((value - low) / span * (len(_BLOCKS) - 1)), len(_BLOCKS) - 1)]
        for value in values
    )


def facet[P: Point](
    rows: Sequence[Row[P]],
    *,
    along: Callable[[P], object],
    within: Callable[[P], object],
    measure: Callable[[Row[P]], float],
    axis: str = "",
    unit: str = "MB/s",
) -> RenderableType:
    """Group `rows` into one line per `within` value, swept along `along`.

    along: the axis running across a row, whose values become the columns.
    within: what distinguishes one row of the multiple from another.
    measure: what to read off each row, usually a rate.
    axis: what `along` varies, named rather than guessed, since the caller passes a lambda and a
        panel titled after an anonymous function tells the reader nothing.
    """
    grouped: dict[object, dict[object, float]] = defaultdict(dict)
    positions: list[object] = []
    for row in rows:
        position = along(row.point)
        if position not in positions:
            positions.append(position)
        grouped[within(row.point)][position] = measure(row)

    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("")
    table.add_column("shape")
    for position in positions:
        table.add_column(str(position), justify="right")
    for name, series in grouped.items():
        values = [series.get(position, 0.0) for position in positions]
        cells = [f"{value:.0f}" if value else "-" for value in values]
        table.add_row(str(name), sparkline(values), *cells)
    return Panel(table, title=f"{unit} by {axis}" if axis else unit, expand=False)


def show_surface[P: Point](
    rows: Sequence[Row[P]],
    *,
    along: Callable[[P], object],
    within: Callable[[P], object],
    measure: Callable[[Row[P]], float],
    axis: str = "",
    unit: str = "MB/s",
    color: bool = True,
) -> None:
    """Print one small multiple for `rows`."""
    Console(no_color=not color).print(
        Group(facet(rows, along=along, within=within, measure=measure, axis=axis, unit=unit))
    )
