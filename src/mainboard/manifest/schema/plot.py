"""Named plot settings; palette and theme names belong to the plotting libraries."""

from patos import FrozenModel
from pydantic import PositiveFloat, PositiveInt


class PlotStyle(FrozenModel):
    """A palette, a Matplotlib theme, and optional native rcParams overrides.

    figsize: width and height in inches; omitted uses paleta's text-column size.
    dpi: raster resolution; vector formats retain vector geometry.
    """

    palette: str = "paleta-shiho"
    theme: str = "paleta-shiho"
    figsize: tuple[PositiveFloat, PositiveFloat] | None = None
    dpi: PositiveInt = 300
    rc: dict[str, bool | float | str | list[str]] = {}
