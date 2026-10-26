"""A heatmap panel: one grid of cells, colored by a value column, under its color bar."""

from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, NullFormatter

from .table.core import loose

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Colormap
    from matplotlib.figure import SubFigure
    from pydantic import JsonValue

    from ..manifest import Panel
    from .table.columns import Columns, Value

# The most distinct cell values a heatmap's color bar marks one by one.
_MARKED_CELLS = 8


class Heatmap:
    """The one layer of a Heatmap panel, drawn by hand since no Seaborn mark draws it."""

    @staticmethod
    def _annotated(axis: Axes, cells: _Cells, *, colors: Callable[[float], str]) -> None:
        """Write each cell's text at its place, dark or light against the cell's own color."""
        if not cells.texts:
            return
        for x, y, text in zip(cells.across, cells.up, cells.texts, strict=True):
            column, line = cells.xs.index(x), cells.ys.index(y)
            loose(axis.text)(
                column,
                line,
                str(text),
                ha="center",
                va="center",
                fontsize=cells.fontsize,
                color=colors(cells.grid[line, column]),
            )

    @staticmethod
    def _cells(selected: Columns, variables: Mapping[str, str]) -> _Cells:
        """The heatmap's one layer as a grid of cells at the distinct x and y values."""
        columns = [variables[key] for key in ("x", "y", "color") if key in variables]
        if len(columns) != 3:
            raise ValueError("Heatmap needs x, y and color columns")
        selected.select(columns).checked()
        if not set(columns) <= selected.numbers:
            raise ValueError("Heatmap x, y and color columns must be numeric")
        if selected.distinct(columns[:2]) != selected.height:
            raise ValueError("heatmap cells repeat; aggregate each cell in SQL first")
        across, up, values = (selected.floats(column) for column in columns)
        texts = selected[variables["text"]] if "text" in variables else []
        return _Cells(across=across, up=up, values=values, texts=texts)

    @staticmethod
    def _colorbar(target: SubFigure, mesh: ScalarMappable, *, axis: Axes, cells: _Cells) -> None:
        """The heatmap's color bar, marked at the cell values when there are few of them."""
        colorbar = target.colorbar(mesh, ax=axis, fraction=0.05, pad=0.03)
        colorbar.set_label(cells.label)
        distinct = sorted(set(cells.masked.compressed().tolist()))
        if len(distinct) <= _MARKED_CELLS:
            colorbar.set_ticks(distinct)
        bar_fmt = cells.bar_fmt
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: bar_fmt.format(value))
        )
        colorbar.ax.yaxis.set_minor_formatter(NullFormatter())

    @staticmethod
    def _ticked(axis: Axes, cells: _Cells) -> None:
        """Tick the grid's columns and lines at their values, with no grid lines over the cells."""
        axis.set_xticks(range(len(cells.xs)), [cells.fmt.format(value) for value in cells.xs])
        axis.set_yticks(range(len(cells.ys)), [cells.fmt.format(value) for value in cells.ys])
        axis.grid(visible=False)

    def __init__(self, panel: Panel) -> None:
        self.panel = panel

    def draw(self, target: SubFigure, selected: Columns, variables: Mapping[str, str]) -> None:
        """One grid of cells at the distinct x and y values, colored by a value column.

        A Heatmap panel holds exactly one layer. `x` and `y` are numeric columns whose distinct
        values become the ordered categories, `color` the numeric cell value, and an optional
        `text` the cell annotation; cells without a row stay blank. Layer kws: `cmap` (a
        Matplotlib colormap name), `log` (a logarithmic color scale), `vmin` and `vmax` (the
        color scale's ends, the cell values' by default), `fontsize` for the annotations,
        `label` for the color bar, `fmt` for the tick labels and `cbar_fmt` for the color bar's,
        which sits at the cell values when there are at most eight of them.
        """
        if len(self.panel.layers) != 1:
            raise ValueError("a Heatmap panel holds exactly one layer")
        cells = Heatmap._cells(selected, variables)
        kws = cells.styled(dict(self.panel.layers[0].kws))
        axis = target.subplots()
        mesh = loose(axis.imshow)(
            cells.masked,
            cmap=cells.colormap,
            norm=cells.norm,
            aspect="auto",
            origin="lower",
            **kws,
        )
        Heatmap._ticked(axis, cells)
        Heatmap._annotated(axis, cells, colors=cells.ink)
        Heatmap._colorbar(target, mesh, axis=axis, cells=cells)
        axis.set(**self.panel.axis)
        loose(axis.tick_params)(**self.panel.ticks)


class _Cells:
    """A heatmap's cells and how its layer styles them.

    across, up, values: each row's x, y and cell value; texts: each row's annotation, if any.
    xs, ys: the distinct x and y values in order, the grid's columns and lines.
    """

    def __init__(
        self,
        *,
        across: list[float],
        up: list[float],
        values: list[float],
        texts: Sequence[Value],
    ) -> None:
        self.across, self.up, self.texts = across, up, texts
        self.xs, self.ys = sorted(set(across)), sorted(set(up))
        self.grid = np.full((len(self.ys), len(self.xs)), np.nan)
        for x, y, cell in zip(across, up, values, strict=True):
            self.grid[self.ys.index(y), self.xs.index(x)] = cell
        self.masked = np.ma.masked_invalid(self.grid)
        self.fmt = self.bar_fmt = "{:g}"
        self.label = ""
        self.fontsize: JsonValue = mpl.rcParams["font.size"]
        self.colormap: Colormap = mpl.colormaps["Purples"]
        self.norm: Normalize = Normalize()

    def ink(self, value: float) -> str:
        """The annotation color readable on a cell of `value`."""
        red, green, blue, _ = self.colormap(self.norm(value))
        return "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"

    def styled(self, kws: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Take this heatmap's own options out of a layer's `kws`, answering the rest.

        The rest go to `imshow` as they are.
        """
        self.fmt = str(kws.pop("fmt", "{:g}"))
        self.bar_fmt = str(kws.pop("cbar_fmt", self.fmt))
        self.fontsize = kws.pop("fontsize", self.fontsize)
        self.label = str(kws.pop("label", ""))
        self.colormap = mpl.colormaps[str(kws.pop("cmap", "Purples"))]
        low = _bound(kws.pop("vmin", None), self.masked.min())
        high = _bound(kws.pop("vmax", None), self.masked.max())
        logarithmic = bool(kws.pop("log", False))
        self.norm = LogNorm(low, high) if logarithmic else Normalize(low, high)
        return kws


def _bound(value: JsonValue, default: float) -> float:
    """One end of a heatmap's color scale: `value` when the layer gave one, else `default`."""
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"a heatmap's color scale ends at numbers, not {value!r}")
    return float(value)
