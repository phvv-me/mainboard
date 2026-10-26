"""Named plot settings; palette and theme names belong to the plotting libraries."""

from typing import Annotated, Literal, Self

from patos import FrozenModel
from pydantic import ConfigDict, Field, JsonValue, PositiveFloat, PositiveInt


class Chart(FrozenModel):
    """What one chart draws from an already selected table: its columns, its marks and title."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    x: str = ""
    """the column on the x axis."""
    y: str = ""
    """the numeric column on the y axis."""
    hue: str = ""
    """the column splitting the series; omit it for one series."""
    kind: Literal["scatter", "line", "bar"] = "scatter"
    """the marks; bar requires one row per x/hue group."""
    title: str = ""
    """the chart title, including the measurement scope when appropriate."""

    @property
    def columns(self) -> list[str]:
        """The columns the chart reads, `hue` only when it splits the series."""
        return [self.x, self.y, *self.groups[1:]]

    @property
    def groups(self) -> list[str]:
        """The columns one series point is keyed by: `x`, and `hue` when it splits the series."""
        return [self.x, *([self.hue] if self.hue else [])]


class PlotStyle(FrozenModel):
    """A palette, a Matplotlib theme, and optional native rcParams overrides.

    figsize: width and height in inches; omitted uses the Matplotlib theme's size.
    dpi: raster resolution; vector formats retain vector geometry.
    """

    palette: Annotated[str | list[str], Field(min_length=1)] = "deep"
    theme: str = "default"
    figsize: tuple[PositiveFloat, PositiveFloat] | None = None
    dpi: PositiveInt = 300
    rc: dict[str, bool | float | str | list[str]] = {}
    colors: dict[str, str] = {}
    labels: dict[str, str] = {}
    markers: dict[str, str] = {}
    linestyles: dict[str, str] = {}
    legend: dict[str, JsonValue] = {"loc": "outside lower center", "frameon": False}
    legend_marker: dict[str, JsonValue] = {"marker": "o", "linestyle": "none"}

    def merged(self, over: Self) -> Self:
        """Overlay explicitly supplied project fields on a shared style.

        Native settings and semantic maps merge by key; lists and scalars replace.
        """
        values = over.model_dump()
        for name, value in self.model_dump(exclude_unset=True).items():
            values[name] = values[name] | value if isinstance(value, dict) else value
        return type(self).model_validate(values)
