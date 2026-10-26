"""Native Matplotlib helpers for bespoke figures using manifest-owned styles."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.colors import to_rgb

from ..core import Project
from ..manifest.loading import load_plot_config

_RATIO = 1.618
_SRGB_LINEAR_SEGMENT_DIVISOR = 12.92
_SRGB_LINEAR_SEGMENT_THRESHOLD = 0.04045
_SRGB_GAMMA_OFFSET_SCALE = 1.055
_RED_LUMINANCE_WEIGHT = 0.2126
_GREEN_LUMINANCE_WEIGHT = 0.7152

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from numpy.typing import NDArray

    from ..manifest.schema.figures.plot import PlotStyle


def load_style(config: str | Path) -> PlotStyle:
    """Load a project's paper style over the root manifest's native settings."""
    root = Project().find_root(Path.cwd())
    return load_plot_config(Project().manifest(root), root / config).plots["paper"]


def apply_style(style: PlotStyle) -> None:
    """Apply the resolved native theme and project rcParams to this process."""
    mplstyle.use([style.theme, style.rc, {"savefig.dpi": style.dpi}])


def subplots(
    width: float = 5.5,
    *,
    ratio: float = _RATIO,
    ncols: int = 1,
    nrows: int = 1,
    sharex: bool | Literal["none", "all", "row", "col"] = False,
    sharey: bool | Literal["none", "all", "row", "col"] = False,
    gridspec_kw: Mapping[str, float | Sequence[float]] | None = None,
) -> tuple[Figure, Axes | NDArray[np.object_]]:
    """Create native axes at a manuscript width in inches and per-panel aspect ratio."""
    return plt.subplots(
        nrows,
        ncols,
        figsize=(width, width / ncols / ratio * nrows),
        constrained_layout=True,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=dict(gridspec_kw or {}),
    )


def save_figure(figure: Figure, *paths: Path | str) -> tuple[Path, ...]:
    """Save multiple formats from one settled layout without changing its geometry."""
    figure.canvas.draw()
    figure.set_layout_engine("none")
    written = tuple(Path(path) for path in paths)
    for path in written:
        figure.savefig(path)
    return written


def figure_legend(
    canvas: Figure, source: Axes | None = None, *, ncols: int | None = None
) -> Legend:
    """Place a native legend below the panels, using one panel's handles."""
    handles, labels = (
        source if source is not None else canvas.axes[0]
    ).get_legend_handles_labels()
    return canvas.legend(
        handles, labels, loc="outside lower center", ncols=ncols or len(labels) or 1
    )


def contrasting_text(fill: str, *, ink: str, surface: str) -> str:
    """Choose the configured foreground or background using linear-sRGB luminance."""
    channels = [
        c / _SRGB_LINEAR_SEGMENT_DIVISOR
        if c <= _SRGB_LINEAR_SEGMENT_THRESHOLD
        else ((c + 0.055) / _SRGB_GAMMA_OFFSET_SCALE) ** 2.4
        for c in to_rgb(fill)
    ]
    luminance = sum(
        c * weight
        for c, weight in zip(
            channels, (_RED_LUMINANCE_WEIGHT, _GREEN_LUMINANCE_WEIGHT, 0.0722), strict=True
        )
    )
    return ink if luminance > 0.2 else surface
