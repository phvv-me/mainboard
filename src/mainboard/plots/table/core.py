"""Render selected result columns with Seaborn and the house figure style."""

import json
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns

from ...manifest.schema.figures.plot import Chart, PlotStyle
from ...render.stamps import Stamp
from .columns import Columns, Value

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Each chart kind's own settings: lines keep SQL order, and a bar group holds one checked value,
# so its sum is the identity; nothing is estimated.
_KINDS: dict[str, tuple[Callable[..., Axes], dict[str, str | int | bool | None]]] = {
    "scatter": (sns.scatterplot, {}),
    "line": (sns.lineplot, {"estimator": None, "errorbar": None, "sort": False}),
    "bar": (sns.barplot, {"estimator": "sum", "errorbar": None, "saturation": 1}),
}


# The metadata field each format keeps a figure's provenance in, and the file beside it.
_METADATA = {
    ".png": "Description",
    ".pdf": "Subject",
    ".svg": "Description",
    ".svgz": "Description",
}
_SIDECAR = ".metadata.json"


class Plot:
    """One already selected table; SQL owns filtering, grouping, and aggregation."""

    @staticmethod
    def _linked(files: Sequence[Path], target: Path) -> None:
        """`files` made durable, then linked at `target` and beside it, the figure first."""
        for file in files:
            with file.open("rb+") as complete:
                os.fsync(complete.fileno())
        for file in files:
            os.link(file, target.with_name(target.name + file.name[len(files[0].name) :]))

    def __init__(
        self,
        frame: Columns,
        style: PlotStyle | None = None,
        *,
        provenance: Sequence[Stamp] = (),
    ) -> None:
        self.frame = frame
        self.style = style if style is not None else PlotStyle()
        self.provenance = provenance

    def _charted(self, chart: Chart) -> Columns:
        """The columns `chart` reads, refused when its y is not numeric or a bar group repeats."""
        data = self.frame.select(chart.columns)
        data.checked()
        if chart.y not in data.numbers:
            raise ValueError("plot y column must be numeric")
        if chart.kind == "bar" and data.distinct(chart.groups) != data.height:
            raise ValueError("bar groups repeat; aggregate each x/hue group in SQL first")
        return data

    def _colors(
        self, levels: Sequence[Value], palette: Sequence[tuple[float, float, float]]
    ) -> dict[Value, str | tuple[float, float, float]]:
        """Each hue level's color: the style's named one, else the palette's in order."""
        if not self.style.colors:
            return dict(zip(levels, palette[: len(levels)], strict=True))
        self._named(str(level) for level in levels)
        return {level: self.style.colors[str(level)] for level in levels}

    def _draw(self, axis: Axes, data: Columns, *, chart: Chart) -> None:
        """Use Seaborn's native axes functions, with SQL order retained for lines."""
        palette = sns.color_palette(self.style.palette)
        hue = chart.hue
        levels = data.unique(hue) if hue else []
        if len(levels) > len(palette) and not self.style.colors:
            raise ValueError(
                f"palette has {len(palette)} slots; fold the tail into Other or facet"
            )
        try:
            draw, options = _KINDS[chart.kind]
        except KeyError:
            raise ValueError("plot kind must be scatter, line, or bar") from None
        draw(
            data=data.plain(),
            x=chart.x,
            y=chart.y,
            hue=hue or None,
            hue_order=levels or None,
            palette=self._colors(levels, palette) if hue else None,
            color=None if hue else palette[0],
            ax=axis,
            **options,
        )

    def _named(self, levels: Iterable[str]) -> None:
        """A named color is an identity, so a level the style leaves unnamed is refused."""
        if absent := set(levels) - self.style.colors.keys():
            raise ValueError(f"style has no explicit colors for {sorted(absent)}")

    def _outputs(self, paths: tuple[Path, ...], dpi: int | None) -> tuple[Path, ...]:
        """Check every destination before querying or creating a figure."""
        if not paths or (dpi is not None and dpi <= 0):
            raise ValueError("plot needs an output path and a positive DPI")
        absolute = tuple(path.expanduser().absolute() for path in paths)
        if len(set(absolute)) != len(absolute):
            raise ValueError("plot output paths must be distinct")
        existing = [
            target
            for path in absolute
            for target in (path, path.with_name(path.name + _SIDECAR))
            if os.path.lexists(target)
        ]
        if existing:
            raise FileExistsError(f"plot output already exists: {existing[0]}")
        return absolute

    def _publish(self, canvas: Figure, paths: tuple[Path, ...]) -> None:
        """Render every requested format before publishing any complete file."""
        formats = canvas.canvas.get_supported_filetypes()
        if any(path.suffix.casefold().lstrip(".") not in formats for path in paths):
            raise ValueError(f"unsupported plot extension; choose from {', '.join(formats)}")
        with ExitStack() as staging:
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
            temporary = [
                Path(staging.enter_context(TemporaryDirectory(dir=path.parent))) / path.name
                for path in paths
            ]
            staged = [self._staged(canvas, path) for path in temporary]
            for files, target in zip(staged, paths, strict=True):
                Plot._linked(files, target)

    def _staged(self, canvas: Figure, path: Path) -> list[Path]:
        """`canvas` rendered at `path`, its provenance in the file's metadata and beside it.

        The files written, the figure first.
        """
        if not self.provenance:
            canvas.savefig(path)
            return [path]
        results = [stamp.model_dump() for stamp in self.provenance]
        document = json.dumps({"results": results}, indent=2) + "\n"
        key = _METADATA.get(path.suffix.casefold())
        canvas.savefig(path, metadata={key: document} if key else None)
        sidecar = path.with_name(path.name + _SIDECAR)
        sidecar.write_text(document, encoding="utf-8")
        return [path, sidecar]

    def save(self, *paths: Path, chart: Chart, dpi: int | None = None) -> tuple[Path, ...]:
        """Save `chart` without estimation, error bars, or overwriting existing output.

        paths: output formats follow their extensions; dpi controls raster resolution.
        """
        dpi = self.style.dpi if dpi is None else dpi
        paths = self._outputs(paths, dpi)
        data = self._charted(chart)
        with mplstyle.context([self.style.theme, self.style.rc]), ExitStack() as cleanup:
            mpl.rcParams["savefig.dpi"] = dpi
            canvas, axis = plt.subplots(figsize=self.style.figsize, layout="constrained")
            cleanup.callback(plt.close, canvas)
            self._draw(axis, data, chart=chart)
            if chart.hue:
                sns.move_legend(axis, "upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
            axis.set_title(chart.title)
            self._publish(canvas, paths)
        return paths


def loose[**P, T](method: Callable[P, T]) -> Callable[..., T]:
    """`method` taking whatever keyword options a manifest supplies.

    A manifest's options are JSON values, which Matplotlib's and Seaborn's typed signatures
    cannot check against; they reach the native call as written and it judges them.
    """
    return method
