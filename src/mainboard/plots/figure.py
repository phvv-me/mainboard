"""Compose SQL-selected panels using native Seaborn objects and Matplotlib settings."""

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import polars as pl

from .panel import PanelPlot
from .table import Plot

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure
    from matplotlib.legend import Legend

    from ..manifest import FigureSpec
    from ..manifest.schema.plot import PlotStyle


class FigurePlot(Plot):
    """SQL supplies values and bounds; native libraries supply marks and geometry."""

    def __init__(self, style: PlotStyle | None = None) -> None:
        super().__init__(pl.DataFrame(), style)

    def render(
        self,
        specification: FigureSpec,
        query: Callable[[str | Path], pl.DataFrame],
        *paths: Path,
        dpi: int | None = None,
    ) -> tuple[Path, ...]:
        """Render one named figure; all SQL and output paths use the caller's cwd."""
        paths = self._outputs(paths or specification.out, dpi)
        tables = {
            name: query(panel.file or panel.sql) for name, panel in specification.panels.items()
        }
        with mplstyle.context([self.style.theme, self.style.rc]), ExitStack() as cleanup:
            mpl.rcParams["savefig.dpi"] = dpi or self.style.dpi
            create = cast("Callable[..., Figure]", plt.figure)
            canvas = create(**{"figsize": self.style.figsize, **specification.layout})
            cleanup.callback(plt.close, canvas)
            mosaic = specification.mosaic or (tuple(specification.panels),)
            grid = canvas.add_gridspec(len(mosaic), len(mosaic[0]), **specification.gridspec)
            shared = {}
            for name, panel in specification.panels.items():
                positions = [
                    (r, c)
                    for r, row in enumerate(mosaic)
                    for c, value in enumerate(row)
                    if value == name
                ]
                rows, columns = zip(*positions, strict=True)
                target = canvas.add_subfigure(
                    grid[min(rows) : max(rows) + 1, min(columns) : max(columns) + 1]
                )
                layers = [
                    query(layer.file or layer.sql)
                    if layer.file or layer.sql.strip()
                    else tables[name]
                    for layer in panel.layers
                ]
                shared.update(PanelPlot(tables[name], panel, self.style).draw(target, layers))
            if shared:
                ordered = [label for label in self.style.colors if label in shared]
                ordered += [label for label in shared if label not in self.style.colors]
                labels = [self.style.labels.get(label, label) for label in ordered]
                legend = cast("Callable[..., Legend]", canvas.legend)
                legend([shared[label] for label in ordered], labels, **self.style.legend)
            self._publish(canvas, paths)
        return paths
