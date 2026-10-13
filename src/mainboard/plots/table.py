"""Render selected result columns with Seaborn and the house figure style."""

import json
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns

from ..manifest.schema.plot import PlotStyle
from ..render.query import Stamp
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


class Plot:
    """One already selected table; SQL owns filtering, grouping, and aggregation."""

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
        paths = self._outputs(paths, dpi)
        hues = [hue] if hue else []
        data = self.frame.select([x, y, *hues])
        data.checked()
        if y not in data.numbers:
            raise ValueError("plot y column must be numeric")
        if kind == "bar" and data.distinct([x, *hues]) != data.height:
            raise ValueError("bar groups repeat; aggregate each x/hue group in SQL first")
        with mplstyle.context([self.style.theme, self.style.rc]), ExitStack() as cleanup:
            mpl.rcParams["savefig.dpi"] = dpi
            canvas, axis = plt.subplots(figsize=self.style.figsize, layout="constrained")
            cleanup.callback(plt.close, canvas)
            self._draw(axis, data, x=x, y=y, hue=hue, kind=kind)
            if hue:
                sns.move_legend(axis, "upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
            axis.set_title(title)
            self._publish(canvas, paths)
        return paths

    def _named(self, levels: Iterable[str]) -> None:
        """A named color is an identity, so a level the style leaves unnamed is refused."""
        if absent := set(levels) - self.style.colors.keys():
            raise ValueError(f"style has no explicit colors for {sorted(absent)}")

    def _draw(self, axis: Axes, data: Columns, *, x: str, y: str, hue: str, kind: str) -> None:
        """Use Seaborn's native axes functions, with SQL order retained for lines."""
        palette = sns.color_palette(self.style.palette)
        levels = data.unique(hue) if hue else []
        if len(levels) > len(palette) and not self.style.colors:
            raise ValueError(
                f"palette has {len(palette)} slots; fold the tail into Other or facet"
            )
        colors: dict[Value, str | tuple[float, float, float]] | None = None
        if hue and not self.style.colors:
            colors = dict(zip(levels, palette[: len(levels)], strict=True))
        elif hue:
            self._named(str(level) for level in levels)
            colors = {level: self.style.colors[str(level)] for level in levels}
        try:
            draw, options = _KINDS[kind]
        except KeyError:
            raise ValueError("plot kind must be scatter, line, or bar") from None
        draw(
            data=data.plain(),
            x=x,
            y=y,
            hue=hue or None,
            hue_order=levels or None,
            palette=colors,
            color=None if hue else palette[0],
            ax=axis,
            **options,
        )

    def _outputs(self, paths: tuple[Path, ...], dpi: int | None) -> tuple[Path, ...]:
        """Check every destination before querying or creating a figure."""
        if not paths or (dpi is not None and dpi <= 0):
            raise ValueError("plot needs an output path and a positive DPI")
        absolute = tuple(path.expanduser().absolute() for path in paths)
        if len(set(absolute)) != len(absolute):
            raise ValueError("plot output paths must be distinct")
        for path in absolute:
            for target in (path, path.with_name(path.name + ".metadata.json")):
                if os.path.lexists(target):
                    raise FileExistsError(f"plot output already exists: {target}")
        return absolute

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
            for path in temporary:
                if self.provenance:
                    document = (
                        json.dumps(
                            {"results": [stamp.model_dump() for stamp in self.provenance]},
                            indent=2,
                        )
                        + "\n"
                    )
                    key = {
                        ".png": "Description",
                        ".pdf": "Subject",
                        ".svg": "Description",
                        ".svgz": "Description",
                    }.get(path.suffix.casefold())
                    if key:
                        canvas.savefig(path, metadata={key: document})
                    else:
                        canvas.savefig(path)
                    path.with_name(path.name + ".metadata.json").write_text(
                        document, encoding="utf-8"
                    )
                else:
                    canvas.savefig(path)
            for source, target in zip(temporary, paths, strict=True):
                artifacts = [source]
                if self.provenance:
                    artifacts.append(source.with_name(source.name + ".metadata.json"))
                for artifact in artifacts:
                    with artifact.open("rb+") as complete:
                        os.fsync(complete.fileno())
                os.link(source, target)
                if self.provenance:
                    os.link(
                        source.with_name(source.name + ".metadata.json"),
                        target.with_name(target.name + ".metadata.json"),
                    )
