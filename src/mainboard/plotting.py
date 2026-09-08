"""Render selected result columns with Seaborn and the house figure style."""

import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import paleta
import polars as pl
import seaborn as sns

from .manifest.schema.plot import PlotStyle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Plot:
    """One already selected table; SQL owns filtering, grouping, and aggregation."""

    def __init__(self, frame: pl.DataFrame, style: PlotStyle | None = None) -> None:
        self.frame = frame
        self.style = style if style is not None else PlotStyle()

    def save(
        self,
        *paths: Path,
        x: str,
        y: str,
        hue: str = "",
        kind: Literal["scatter", "line", "bar"] = "scatter",
        dpi: int | None = None,
        title: str = "",
    ) -> tuple[Path, ...]:
        """Save a chart without estimation, error bars, or overwriting existing output.

        x, y, hue: selected column names; omit hue for one series.
        paths: output formats follow their extensions; dpi controls raster resolution.
        """
        dpi = self.style.dpi if dpi is None else dpi
        if not paths or dpi <= 0:
            raise ValueError("plot needs an output path and a positive DPI")
        paths = tuple(path.expanduser().absolute() for path in paths)
        if len(set(paths)) != len(paths):
            raise ValueError("plot output paths must be distinct")
        for path in paths:
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            raise FileExistsError(f"plot output already exists: {path}")
        data = self.frame.select(list(dict.fromkeys([x, y, *([hue] if hue else [])])))
        if data.is_empty() or any(data.null_count().row(0)):
            raise ValueError("plot columns must contain rows without null values")
        if not data[y].dtype.is_numeric():
            raise ValueError("plot y column must be numeric")
        if any(
            not series.cast(pl.Float64).is_finite().all()
            for series in data
            if series.dtype.is_numeric()
        ):
            raise ValueError("plot columns must contain finite values")
        if kind == "bar" and data.n_unique([x, *([hue] if hue else [])]) != data.height:
            raise ValueError("bar groups repeat; aggregate each x/hue group in SQL first")
        paleta.register()
        with mpl.style.context([self.style.theme, self.style.rc]):
            mpl.rcParams["savefig.dpi"] = dpi
            canvas, axis = paleta.figure()
            try:
                if self.style.figsize is not None:
                    canvas.set_size_inches(self.style.figsize)
                self._draw(axis, data, x=x, y=y, hue=hue, kind=kind)
                axis.set_title(title)
                self._publish(canvas, paths)
            finally:
                plt.close(canvas)
        return paths

    def _draw(
        self, axis: Axes, data: pl.DataFrame, *, x: str, y: str, hue: str, kind: str
    ) -> None:
        """Use Seaborn's native axes functions, with SQL order retained for lines."""
        columns = data.to_dict(as_series=False)
        palette = sns.color_palette(self.style.palette)
        levels = data[hue].unique(maintain_order=True).to_list() if hue else []
        if len(levels) > len(palette):
            raise ValueError(
                f"palette has {len(palette)} slots; fold the tail into Other or facet"
            )
        colors = dict(zip(levels, palette[: len(levels)], strict=True)) if hue else None
        color = None if hue else palette[0]
        match kind:
            case "scatter":
                sns.scatterplot(
                    data=columns,
                    x=x,
                    y=y,
                    hue=hue or None,
                    hue_order=levels or None,
                    palette=colors,
                    color=color,
                    ax=axis,
                )
            case "line":
                sns.lineplot(
                    data=columns,
                    x=x,
                    y=y,
                    hue=hue or None,
                    hue_order=levels or None,
                    palette=colors,
                    color=color,
                    ax=axis,
                    estimator=None,
                    errorbar=None,
                    sort=False,
                )
            case "bar":
                # Each checked group has one value, so this sum is the identity.
                sns.barplot(
                    data=columns,
                    x=x,
                    y=y,
                    hue=hue or None,
                    hue_order=levels or None,
                    palette=colors,
                    color=color,
                    ax=axis,
                    estimator="sum",
                    errorbar=None,
                    saturation=1,
                )
            case _:
                raise ValueError("plot kind must be scatter, line, or bar")

    def _publish(self, canvas: Figure, paths: tuple[Path, ...]) -> None:
        """Render every requested format before publishing any complete file."""
        formats = canvas.canvas.get_supported_filetypes()
        if any(path.suffix.casefold().lstrip(".") not in formats for path in paths):
            raise ValueError(f"unsupported plot extension; choose from {', '.join(formats)}")
        with ExitStack() as staging:
            temporary = []
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                directory = staging.enter_context(TemporaryDirectory(dir=path.parent))
                temporary.append(Path(directory) / path.name)
            paleta.save(canvas, *temporary)
            for source, target in zip(temporary, paths, strict=True):
                with source.open("rb+") as complete:
                    os.fsync(complete.fileno())
                os.link(source, target)
