"""Native panel composition; semantic legends remain separate from measurement layers."""

from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import polars as pl
import seaborn.objects as so
from matplotlib.lines import Line2D

from .table import Plot

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Set

    from matplotlib.artist import Artist
    from matplotlib.figure import SubFigure
    from matplotlib.legend import Legend

    from ..manifest import Layer, Panel
    from ..manifest.schema.plot import PlotStyle

_MARKS: dict[str, Callable[..., so.Mark]] = {
    name: getattr(so, name)
    for name in (
        "Dot",
        "Dots",
        "Line",
        "Lines",
        "Path",
        "Paths",
        "Dash",
        "Range",
        "Bar",
        "Bars",
        "Area",
        "Band",
        "Text",
    )
}
_MOVES: dict[str, Callable[..., so.Move]] = {"Dodge": so.Dodge, "Stack": so.Stack}


class PanelPlot(Plot):
    """One panel and its native layers, with no hidden data transformation."""

    def __init__(self, frame: pl.DataFrame, panel: Panel, style: PlotStyle) -> None:
        super().__init__(frame, style)
        self.panel = panel

    def draw(self, target: SubFigure, tables: list[pl.DataFrame]) -> dict[str, Artist]:
        """Compile marks, apply axes, and return the shared color legend."""
        drawing = self._drawing(tables)
        parent = target.get_figure(root=True)
        assert parent is not None
        existing = set((*target.legends, *parent.legends))
        drawing.on(target).plot()
        self._axes(target)
        created = set((*target.legends, *parent.legends)) - existing
        return self._legends(target, tables, created)

    @staticmethod
    def _interval(frame: pl.DataFrame, variables: Mapping[str, str], coordinate: str) -> None:
        """Check numeric endpoints and an optional numeric center for one interval."""
        lower, upper = frame[variables[f"{coordinate}min"]], frame[variables[f"{coordinate}max"]]
        if not lower.dtype.is_numeric() or not upper.dtype.is_numeric():
            raise ValueError("interval bounds must be numeric")
        if (lower > upper).any():
            raise ValueError("interval lower bound exceeds upper bound")
        center = frame[variables.get(coordinate, variables[f"{coordinate}min"])]
        if center.dtype.is_numeric() and ((lower > center) | (center > upper)).any():
            raise ValueError("interval bounds must contain the plotted value")

    @staticmethod
    def _numeric(frame: pl.DataFrame) -> pl.DataFrame:
        """Plot decimal SQL literals as floats without changing the selected evidence."""
        return frame.with_columns(
            pl.col(name).cast(pl.Float64)
            for name, dtype in frame.schema.items()
            if isinstance(dtype, pl.Decimal)
        )

    def _axes(self, target: SubFigure) -> None:
        """Apply native axis settings and the project's semantic display labels."""
        panel = self.panel
        for axis in target.axes:
            axis.set(**panel.axis)
            ticks = cast("Callable[..., None]", axis.tick_params)
            ticks(**panel.ticks)
            if panel.grid:
                grid = cast("Callable[..., None]", axis.grid)
                grid(**panel.grid)
            title = axis.get_title()
            axis.set_title(self.style.labels.get(title, title))
        coordinates = [
            coordinate for axis in target.axes for coordinate in (axis.xaxis, axis.yaxis)
        ]
        for coordinate in coordinates:
            texts = [label.get_text() for label in coordinate.get_ticklabels()]
            if any(text in self.style.labels for text in texts):
                coordinate.set_ticks(
                    coordinate.get_ticklocs(),
                    [self.style.labels.get(text, text) for text in texts],
                )

    def _bars(self, frame: pl.DataFrame, variables: Mapping[str, str]) -> None:
        """Bars require one row per category, semantic group, and facet."""
        panel = self.panel
        orient = "y" if "y" in variables and not frame[variables["y"]].dtype.is_numeric() else "x"
        grouping = [
            column
            for key, column in variables.items()
            if key in {orient, "color", "group", "marker"}
        ]
        grouping += [
            value
            for key, value in panel.facet.items()
            if key in {"row", "col"} and isinstance(value, str)
        ]
        if frame.n_unique(list(dict.fromkeys(grouping))) != frame.height:
            raise ValueError("bar groups repeat; aggregate each group in SQL first")

    def _bounds(self, frame: pl.DataFrame, variables: Mapping[str, str]) -> None:
        """Require supplied interval endpoints to contain their plotted center."""
        for coordinate in ("x", "y"):
            low, high = f"{coordinate}min", f"{coordinate}max"
            if (low in variables) != (high in variables):
                raise ValueError("intervals require both lower and upper bound columns")
            if low not in variables:
                continue
            self._interval(frame, variables, coordinate)

    def _drawing(self, tables: list[pl.DataFrame]) -> so.Plot:
        """Bind each layer's complete mappings before native Seaborn composition."""
        panel = self.panel
        construct = cast("Callable[..., so.Plot]", so.Plot)
        drawing = construct(
            self._numeric(self.frame).to_dict(as_series=False), **panel.variables
        ).theme({key: value for key, value in mpl.rcParams.items()})
        for layer, selected in zip(panel.layers, tables, strict=True):
            self._validate(selected, layer)
            add = cast("Callable[..., so.Plot]", drawing.add)
            drawing = add(
                _MARKS[layer.mark](**layer.kws),
                *[_MOVES[name](**kws) for name, kws in layer.moves.items()],
                data=self._numeric(selected).to_dict(as_series=False),
                **(
                    panel.variables
                    | layer.variables
                    | {
                        key: value
                        for key, value in panel.facet.items()
                        if key in {"row", "col"} and isinstance(value, str)
                    }
                ),
            )
        return self._scales(drawing)

    def _legends(
        self, target: SubFigure, tables: list[pl.DataFrame], created: Set[Legend]
    ) -> dict[str, Artist]:
        """Replace native combined guides with explicit shared and secondary keys."""
        panel = self.panel
        parent = target.get_figure(root=True)
        assert parent is not None
        native = {
            text.get_text(): handle
            for legend in created
            for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True)
            if handle is not None
        }
        for owner in (target, parent):
            owner.legends[:] = [legend for legend in owner.legends if legend not in created]
        secondary = self._secondary(native)
        if secondary and (
            panel.legend is not None
            or any(key in panel.variables for key in ("marker", "linestyle", "fill"))
        ):
            local = cast("Callable[..., Legend]", target.axes[0].legend)
            local(
                list(secondary.values()),
                [self.style.labels.get(label, label) for label in secondary],
                **(
                    panel.legend if panel.legend is not None else {"loc": "best", "frameon": False}
                ),
            )
        elif panel.legend is not None and native:
            local = cast("Callable[..., Legend]", target.axes[0].legend)
            local(
                list(native.values()),
                [self.style.labels.get(label, label) for label in native],
                **panel.legend,
            )
            return {}
        # Color-only figure legends do not confuse batch markers with engine identity.
        if not self.style.colors:
            return native
        return self._shared(tables)

    def _scales(self, drawing: so.Plot) -> so.Plot:
        """Keep color identity stable across subsets and reuse native nominal ordering."""
        panel = self.panel
        for variable, values in (
            ("color", self.style.colors),
            ("marker", self.style.markers),
            ("linestyle", self.style.linestyles),
        ):
            if values:
                drawing = drawing.scale(
                    **{
                        variable: so.Nominal(
                            values=values, order=list(panel.order.get(variable, values))
                        )
                    }
                )
        for variable, order in panel.order.items():
            if variable not in {"color", "marker", "linestyle"}:
                drawing = drawing.scale(**{variable: so.Nominal(order=list(order))})
        if not self.style.colors:
            drawing = drawing.scale(color=so.Nominal(values=self.style.palette))
        if panel.facet:
            facet = cast("Callable[..., so.Plot]", drawing.facet)
            drawing = facet(**panel.facet)
        return drawing

    def _secondary(self, native: dict[str, Artist]) -> dict[str, Artist]:
        """Draw observed batch/line/fill keys independently of engine colors."""
        panel, frame = self.panel, self.frame
        if not self.style.colors:
            return native
        # A native secondary key retains batch markers/fill while colors remain shared.
        secondary = {
            label: handle
            for label, handle in native.items()
            if label not in self.style.colors and label not in panel.variables.values()
        }
        proxy = cast("Callable[..., Line2D]", Line2D)
        variables = {"marker", "linestyle", "fill"} & panel.variables.keys()
        for variable in sorted(variables):
            for value in frame[panel.variables[variable]].unique(maintain_order=True):
                label = str(value)
                marker = self.style.markers.get(label, "o")
                linestyle = self.style.linestyles.get(label, "none")
                secondary[label] = proxy(
                    [],
                    [],
                    color=mpl.rcParams["text.color"],
                    marker=marker,
                    linestyle=linestyle,
                    markerfacecolor="none"
                    if variable == "fill" and not value
                    else mpl.rcParams["text.color"],
                )
        return secondary

    def _shared(self, tables: list[pl.DataFrame]) -> dict[str, Artist]:
        """Build one engine key from the explicit colors actually present in layers."""
        present: set[str] = set()
        for layer, selected in zip(self.panel.layers, tables, strict=True):
            variables = self.panel.variables | layer.variables
            if "color" in variables:
                present.update(selected[variables["color"]].cast(pl.String).unique())
        proxy = cast("Callable[..., Line2D]", Line2D)
        return {
            name: proxy([], [], color=color, **self.style.legend_marker)
            for name, color in self.style.colors.items()
            if name in present
        }

    def _validate(self, frame: pl.DataFrame, layer: Layer) -> None:
        """Missing data and absent interval bounds are never silently estimated."""
        panel = self.panel
        variables = (
            panel.variables
            | layer.variables
            | {
                key: value
                for key, value in panel.facet.items()
                if key in {"row", "col"} and isinstance(value, str)
            }
        )
        columns = list(dict.fromkeys(variables.values()))
        self._data(frame.select(columns))
        self._bounds(frame, variables)
        if layer.mark in {"Range", "Band"} and not ({"xmin", "ymin"} & variables.keys()):
            raise ValueError("Range/Band requires explicit bounds from SQL, never estimation")
        if layer.mark in {"Bar", "Bars"}:
            self._bars(frame, variables)
        if self.style.colors and "color" in variables:
            absent = (
                set(frame[variables["color"]].cast(pl.String).unique()) - self.style.colors.keys()
            )
            if absent:
                raise ValueError(f"style has no explicit colors for {sorted(absent)}")
