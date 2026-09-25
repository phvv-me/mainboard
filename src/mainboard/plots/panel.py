"""Native panel composition; semantic legends remain separate from measurement layers."""

from typing import TYPE_CHECKING, cast, get_args

import matplotlib as mpl
import numpy as np
import polars as pl
import seaborn.objects as so
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from ..manifest.schema.figures.layer import Layer
from .table import Plot

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.figure import FigureBase, SubFigure
    from matplotlib.image import AxesImage
    from matplotlib.legend import Legend
    from pydantic import JsonValue

    from ..manifest import Panel
    from ..manifest.schema.plot import PlotStyle

# Every manifest mark but Heatmap, which the panel draws itself, is a Seaborn mark of that name.
_MARKS: dict[str, Callable[..., so.Mark]] = {
    name: getattr(so, name)
    for name in get_args(Layer.model_fields["mark"].annotation)
    if name != "Heatmap"
}
_MOVES: dict[str, Callable[..., so.Move]] = {"Dodge": so.Dodge, "Stack": so.Stack}
_SECONDARY = {"marker", "linestyle", "fill"}
_PLAIN_KEY: dict[str, JsonValue] = {"loc": "best", "frameon": False}


class PanelPlot(Plot):
    """One panel and its native layers, with no hidden data transformation."""

    def __init__(self, frame: pl.DataFrame, panel: Panel, style: PlotStyle) -> None:
        super().__init__(frame, style)
        self.panel = panel

    def draw(self, target: SubFigure, tables: list[pl.DataFrame]) -> dict[str, Artist]:
        """Compile marks, apply axes, and return the shared color legend."""
        if any(layer.mark == "Heatmap" for layer in self.panel.layers):
            return self._heatmap(target, tables)
        parent = target.get_figure(root=True)
        assert parent is not None
        owners = (target, parent)
        existing = {legend for owner in owners for legend in owner.legends}
        self._drawing(tables).on(target).plot()
        self._axes(target)
        return self._legends(target, tables, self._detach(owners, existing))

    @property
    def _facets(self) -> dict[str, str]:
        """The row and col facets that name a column."""
        return {
            key: value
            for key, value in self.panel.facet.items()
            if key in {"row", "col"} and isinstance(value, str)
        }

    def _variables(self, layer: Layer) -> dict[str, str]:
        """Every column one layer binds: the panel's, its own, then the facets."""
        return self.panel.variables | layer.variables | self._facets

    def _mapped(self) -> dict[str, str]:
        """Every variable the panel binds, whether on the panel or on one of its layers."""
        mapped = dict(self.panel.variables)
        for layer in self.panel.layers:
            mapped |= layer.variables
        return mapped

    @staticmethod
    def _numeric(frame: pl.DataFrame) -> pl.DataFrame:
        """Plot decimal SQL literals as floats without changing the selected evidence."""
        return frame.with_columns(
            pl.col(name).cast(pl.Float64)
            for name, dtype in frame.schema.items()
            if isinstance(dtype, pl.Decimal)
        )

    def _drawing(self, tables: list[pl.DataFrame]) -> so.Plot:
        """Bind each layer's complete mappings before native Seaborn composition."""
        drawing = cast("Callable[..., so.Plot]", so.Plot)(
            self._numeric(self.frame).to_dict(as_series=False), **self.panel.variables
        ).theme(dict(mpl.rcParams.items()))
        for layer, selected in zip(self.panel.layers, tables, strict=True):
            self._validate(selected, layer)
            drawing = cast("Callable[..., so.Plot]", drawing.add)(
                _MARKS[layer.mark](**layer.kws),
                *[_MOVES[name](**kws) for name, kws in layer.moves.items()],
                data=self._numeric(selected).to_dict(as_series=False),
                **self._variables(layer),
            )
        return self._scales(drawing)

    def _validate(self, frame: pl.DataFrame, layer: Layer) -> None:
        """Missing data and absent interval bounds are never silently estimated."""
        variables = self._variables(layer)
        self._data(frame.select(list(dict.fromkeys(variables.values()))))
        self._bounds(frame, variables)
        if layer.mark in {"Range", "Band"} and not ({"xmin", "ymin"} & variables.keys()):
            raise ValueError("Range/Band requires explicit bounds from SQL, never estimation")
        if layer.mark in {"Bar", "Bars"}:
            self._bars(frame, variables)
        if self.style.colors and "color" in variables:
            self._named(frame[variables["color"]].cast(pl.String).unique())

    @staticmethod
    def _bounds(frame: pl.DataFrame, variables: Mapping[str, str]) -> None:
        """Supplied interval endpoints are numeric, ordered, and contain a numeric center."""
        for coordinate in ("x", "y"):
            low, high = f"{coordinate}min", f"{coordinate}max"
            if (low in variables) != (high in variables):
                raise ValueError("intervals require both lower and upper bound columns")
            if low not in variables:
                continue
            lower, upper = frame[variables[low]], frame[variables[high]]
            if not lower.dtype.is_numeric() or not upper.dtype.is_numeric():
                raise ValueError("interval bounds must be numeric")
            if (lower > upper).any():
                raise ValueError("interval lower bound exceeds upper bound")
            center = frame[variables.get(coordinate, variables[low])]
            if center.dtype.is_numeric() and ((lower > center) | (center > upper)).any():
                raise ValueError("interval bounds must contain the plotted value")

    def _bars(self, frame: pl.DataFrame, variables: Mapping[str, str]) -> None:
        """Bars require one row per category, semantic group, and facet."""
        orient = "y" if "y" in variables and not frame[variables["y"]].dtype.is_numeric() else "x"
        grouping = [
            column
            for key, column in variables.items()
            if key in {orient, "color", "group", "marker"}
        ]
        if (
            frame.n_unique(list(dict.fromkeys([*grouping, *self._facets.values()])))
            != frame.height
        ):
            raise ValueError("bar groups repeat; aggregate each group in SQL first")

    def _scales(self, drawing: so.Plot) -> so.Plot:
        """Keep color identity stable across subsets and reuse native nominal ordering."""
        panel = self.panel
        for variable, values in (
            ("color", self.style.colors),
            ("marker", self.style.markers),
            ("linestyle", self.style.linestyles),
        ):
            if values:
                ordered = list(panel.order.get(variable, values))
                drawing = drawing.scale(**{variable: so.Nominal(values=values, order=ordered)})
        for variable, order in panel.order.items():
            if variable not in {"color", "marker", "linestyle"}:
                drawing = drawing.scale(**{variable: so.Nominal(order=list(order))})
        if not self.style.colors:
            drawing = drawing.scale(color=so.Nominal(values=self.style.palette))
        for method, options in (
            (so.Plot.label, panel.label),
            (so.Plot.facet, panel.facet),
            (so.Plot.share, panel.share),
        ):
            if options:
                drawing = cast("Callable[..., so.Plot]", method)(drawing, **options)
        return drawing

    def _axes(self, target: SubFigure) -> None:
        """Apply native axis settings and the project's semantic display labels."""
        panel = self.panel
        for axis in target.axes:
            axis.set(**panel.axis)
            cast("Callable[..., None]", axis.tick_params)(**panel.ticks)
            # Matplotlib turns a tick label about its centre, which walks a slanted label off
            # the tick it names. Anchoring the corner keeps each one under its own category.
            rotation = panel.ticks.get("labelrotation")
            if isinstance(rotation, (int, float)) and rotation % 180:
                for label in axis.get_xticklabels():
                    label.set_horizontalalignment("right" if rotation % 360 < 180 else "left")
                    label.set_rotation_mode("anchor")
            # Explicit ticks name every label the author wants; a log axis adds none between.
            for ticks, coordinate in (("xticks", axis.xaxis), ("yticks", axis.yaxis)):
                if ticks in panel.axis:
                    coordinate.set_minor_formatter(NullFormatter())
            if panel.grid:
                cast("Callable[..., None]", axis.grid)(**panel.grid)
            title = axis.get_title()
            axis.set_title(self.style.labels.get(title, title))
        for axis in target.axes:
            for coordinate in (axis.xaxis, axis.yaxis):
                texts = [label.get_text() for label in coordinate.get_ticklabels()]
                if any(text in self.style.labels for text in texts):
                    coordinate.set_ticks(
                        coordinate.get_ticklocs(),
                        [self.style.labels.get(text, text) for text in texts],
                    )

    def _heatmap(self, target: SubFigure, tables: list[pl.DataFrame]) -> dict[str, Artist]:
        """One grid of cells at the distinct x and y values, colored by a value column.

        A Heatmap panel holds exactly one layer. `x` and `y` are numeric columns whose distinct
        values become the ordered categories, `color` the numeric cell value, and an optional
        `text` the cell annotation; cells without a row stay blank. Layer kws: `cmap` (a
        Matplotlib colormap name), `log` (a logarithmic color scale), `vmin` and `vmax` (the
        color scale's ends, the cell values' by default), `fontsize` for the annotations,
        `label` for the color bar, `fmt` for the tick labels and `cbar_fmt` for the color bar's,
        which sits at the cell values when there are at most eight of them.
        """
        panel = self.panel
        if len(panel.layers) != 1:
            raise ValueError("a Heatmap panel holds exactly one layer")
        (layer,), (frame,) = panel.layers, tables
        variables = self._variables(layer)
        frame = self._numeric(frame)
        columns = [variables[key] for key in ("x", "y", "color") if key in variables]
        if len(columns) != 3:
            raise ValueError("Heatmap needs x, y and color columns")
        self._data(frame.select(columns))
        if not all(frame[column].dtype.is_numeric() for column in columns):
            raise ValueError("Heatmap x, y and color columns must be numeric")
        x, y, color = columns
        if frame.n_unique([x, y]) != frame.height:
            raise ValueError("heatmap cells repeat; aggregate each cell in SQL first")
        xs = sorted(frame[x].unique().to_list())
        ys = sorted(frame[y].unique().to_list())
        grid = np.full((len(ys), len(xs)), np.nan)
        for row in frame.iter_rows(named=True):
            grid[ys.index(row[y]), xs.index(row[x])] = row[color]
        kws = dict(layer.kws)
        fmt = str(kws.pop("fmt", "{:g}"))
        bar_fmt = str(kws.pop("cbar_fmt", fmt))
        fontsize = kws.pop("fontsize", mpl.rcParams["font.size"])
        label = str(kws.pop("label", ""))
        logarithmic = bool(kws.pop("log", False))
        colormap = mpl.colormaps[str(kws.pop("cmap", "Purples"))]
        values = np.ma.masked_invalid(grid)
        low = float(cast("float", kws.pop("vmin", values.min())))
        high = float(cast("float", kws.pop("vmax", values.max())))
        norm = LogNorm(low, high) if logarithmic else Normalize(low, high)
        axis = target.subplots()
        mesh = cast("Callable[..., AxesImage]", axis.imshow)(
            values, cmap=colormap, norm=norm, aspect="auto", origin="lower", **kws
        )
        axis.set_xticks(range(len(xs)), [fmt.format(value) for value in xs])
        axis.set_yticks(range(len(ys)), [fmt.format(value) for value in ys])
        axis.grid(False)
        if "text" in variables:
            for row in frame.iter_rows(named=True):
                column, line = xs.index(row[x]), ys.index(row[y])
                red, green, blue, _ = colormap(norm(grid[line, column]))
                axis.text(
                    column,
                    line,
                    str(row[variables["text"]]),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black",
                )
        colorbar = target.colorbar(mesh, ax=axis, fraction=0.05, pad=0.03)
        colorbar.set_label(label)
        distinct = sorted(set(values.compressed().tolist()))
        if len(distinct) <= 8:
            colorbar.set_ticks(distinct)
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: bar_fmt.format(value))
        )
        colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
        axis.set(**panel.axis)
        cast("Callable[..., None]", axis.tick_params)(**panel.ticks)
        return {}

    @staticmethod
    def _detach(owners: Sequence[FigureBase], existing: set[Legend]) -> dict[str, Artist]:
        """Remove the legends Seaborn just drew and return their entries, label to handle."""
        created = [
            legend for owner in owners for legend in owner.legends if legend not in existing
        ]
        for owner in owners:
            owner.legends[:] = [legend for legend in owner.legends if legend in existing]
        return {
            text.get_text(): handle
            for legend in created
            for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True)
            if handle is not None
        }

    def _legends(
        self, target: SubFigure, tables: list[pl.DataFrame], native: dict[str, Artist]
    ) -> dict[str, Artist]:
        """Replace native combined guides with explicit shared and secondary keys."""
        panel = self.panel
        if panel.legend is False:
            return {}
        secondary = self._secondary(native)
        layered = bool(_SECONDARY & self._mapped().keys())
        # A mark only one layer carries has no key of its own, so the panel draws the whole
        # key itself, colors first and marks under them. A mark the panel carries keeps the
        # split it already had: marks here, colors in the figure's shared key.
        alone = layered and not (_SECONDARY & panel.variables.keys())
        if secondary and (isinstance(panel.legend, dict) or layered):
            colors = self._swatches(tables) if panel.key != "marks" else {}
            shapes = secondary if panel.key != "colors" else {}
            options = panel.legend if isinstance(panel.legend, dict) else _PLAIN_KEY
            self._key(target, (colors | shapes) if alone else secondary, options)
            if alone:
                return {}
        elif isinstance(panel.legend, dict) and native:
            # Seaborn hands back whichever artist drew each level last, so a reference line
            # layered over the series lends the key its own grey dash and the key stops
            # naming colors. Entries that are color values are redrawn in their own color;
            # anything else, a reference line among them, keeps the artist it came from.
            colored = self._swatches(tables)
            entries = {label: colored.get(label, handle) for label, handle in native.items()}
            self._key(target, entries, panel.legend)
            return {}
        # Color-only figure legends do not confuse batch markers with engine identity.
        return self._swatches(tables, **self.style.legend_marker) if self.style.colors else native

    def _key(
        self,
        target: SubFigure,
        entries: Mapping[str, Artist],
        options: Mapping[str, JsonValue],
    ) -> None:
        """Draw `entries` as the panel's own legend under the style's display labels."""
        cast("Callable[..., Legend]", target.axes[0].legend)(
            list(entries.values()),
            [self.style.labels.get(label, label) for label in entries],
            **options,
        )

    def _secondary(self, native: dict[str, Artist]) -> dict[str, Artist]:
        """Draw observed batch/line/fill keys independently of engine colors."""
        if not self.style.colors:
            return native
        mapped = self._mapped()
        # A native secondary key retains batch markers/fill while colors remain shared.
        secondary = {
            label: handle
            for label, handle in native.items()
            if label not in self.style.colors and label not in mapped.values()
        }
        ink = mpl.rcParams["text.color"]
        proxy = cast("Callable[..., Line2D]", Line2D)
        for variable in sorted(_SECONDARY & mapped.keys()):
            for value in self.frame[mapped[variable]].unique(maintain_order=True):
                label = str(value)
                secondary[label] = proxy(
                    [],
                    [],
                    color=ink,
                    marker=self.style.markers.get(label, "o"),
                    linestyle=self.style.linestyles.get(label, "none"),
                    markerfacecolor="none" if variable == "fill" and not value else ink,
                )
        return secondary

    def _swatches(self, tables: list[pl.DataFrame], **options: JsonValue) -> dict[str, Artist]:
        """One proxy line per style color the layers draw, in style order."""
        present: set[str] = set()
        for layer, selected in zip(self.panel.layers, tables, strict=True):
            if "color" in (variables := self._variables(layer)):
                present.update(selected[variables["color"]].cast(pl.String).unique())
        return {
            name: cast("Callable[..., Line2D]", Line2D)([], [], color=color, **options)
            for name, color in self.style.colors.items()
            if name in present
        }
