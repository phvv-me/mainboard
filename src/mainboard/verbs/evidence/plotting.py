"""`paper plot`: a chart drawn from a local SELECT with Seaborn and Matplotlib.

The CLI loads this module by name only when the verb is typed: importing the plotting stack takes
over a second, which no other verb should pay.
"""

from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ...core.errors import MissionError
from ...core.project import Project
from ...log import logger
from ...manifest.loading import load_plot_config
from ...manifest.schema.figures.plot import Chart, PlotStyle
from ...results import Results
from ..options import query_source
from ..session import Session

if TYPE_CHECKING:
    from collections.abc import Generator

    from ...manifest.schema.figures import figure as figure_schema
    from ...manifest.schema.root import Manifest

# The modules a plot draws with, which the plot extra brings; another missing one is a real fault.
_PLOT_EXTRA = frozenset({"seaborn", "matplotlib", "pandas"})


def _refusal(fault: ModuleNotFoundError) -> Exception:
    """The install line for a missing plot extra, else the fault itself."""
    if fault.name not in _PLOT_EXTRA:
        return fault
    return MissionError(
        "plotting requires the plot extra. From the monorepo root run: "
        "uv tool install --reinstall --python 3.14 "
        "--from './packages/mainboard[plot]' mainboard --force"
    )


@contextmanager
def _plot_extra() -> Generator[None]:
    """Answer a missing plot module with its install line, any other fault unchanged."""
    try:
        yield
    except ModuleNotFoundError as missing:
        raise _refusal(missing) from missing


with _plot_extra():
    from ...plots import figure
    from ...plots.table import columns
    from ...plots.table import core as table


@Parameter(name="*")
class Rendering(FrozenModel):
    """Where a plot's SQL, styles and figures come from, and where it writes."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    file: Path | None = None
    """read a UTF-8 SQL file instead of the positional statement. Both the file and paths inside
    SQL resolve from the caller's current directory."""
    config: Path | None = None
    """overlay project styles and figures from this manifest-format TOML file. It does not select
    an environment or change path resolution."""
    figure: str = ""
    """render a named [figures.NAME] specification; omit SQL, x, and y. Panels use native
    Seaborn marks and Matplotlib settings, without estimation."""
    out: tuple[Path, ...] = ()
    """a new output path; repeat for multiple formats, such as .pdf and .png."""
    project: str = ""
    """restrict scientific rows to this research project."""
    style: str = ""
    """a named [plots.NAME] entry; defaults to paper when declared."""
    dpi: int | None = None
    """raster resolution, overriding the style's DPI when supplied."""

    def _figured(self, manifest: Manifest) -> figure_schema.FigureSpec | None:
        """The declared figure `--figure` names, None without one."""
        if not self.figure:
            return None
        try:
            return manifest.figures[self.figure]
        except KeyError:
            raise MissionError(
                f"no figure {self.figure!r}; declared figures are {sorted(manifest.figures)}"
            ) from None

    def _styled(self, manifest: Manifest, declared: figure_schema.FigureSpec | None) -> PlotStyle:
        """The style asked, else the figure's, else `paper` when declared, else the default."""
        style = self.style or (declared.style if declared else "")
        style = style or ("paper" if "paper" in manifest.plots else "")
        try:
            return manifest.plots[style] if style else PlotStyle()
        except KeyError:
            raise MissionError(
                f"no plot style {style!r}; declared styles are {sorted(manifest.plots)}"
            ) from None

    def draw(
        self, results: Results, manifest: Manifest, *, source: str | Path, chart: Chart
    ) -> tuple[Path, ...]:
        """Draw the declared figure, else `chart` over `source`, and answer the files written."""
        declared = self._figured(manifest)
        style = self._styled(manifest, declared)
        if declared is not None:
            query = partial(results.stamped, project=self.project)
            return figure.FigurePlot(style).render(
                declared, query, *self.out, dpi=self.dpi, provenance=results.provenance
            )
        frame = columns.Columns.of(results.stamped(source, project=self.project))
        drawn = table.Plot(frame, style, provenance=results.provenance)
        return drawn.save(*self.out, chart=chart, dpi=self.dpi)

    def source(self, sql: str | None, chart: Chart) -> str | Path:
        """The SELECT a chart draws, refused when the asks contradict one another.

        A declared figure carries its own SQL and mappings, so it takes none of them; a chart
        needs a statement and both axes.
        """
        source = query_source(sql, self.file)
        if self.figure:
            if source is not None or chart != _CHART:
                raise MissionError(
                    "--figure cannot be combined with SQL or individual chart mappings"
                )
            return ""
        if source is None:
            raise MissionError("plot requires a SQL statement or --file")
        if not chart.x or not chart.y:
            raise MissionError("plot requires --x and --y without --figure")
        return source


# The verb's defaults: a plain chart, the style and outputs the manifest declares.
_CHART = Chart()
_RENDERING = Rendering()


def plot(
    sql: Annotated[
        str | None,
        Parameter(help="define the table, including any filtering, grouping, and ordering."),
    ] = None,
    *,
    chart: Annotated[Chart, Parameter(name="*")] = _CHART,
    rendering: Rendering = _RENDERING,
) -> None:
    """Plot a local SELECT using Seaborn and Matplotlib, without implicit aggregation."""
    source = rendering.source(sql, chart)
    root = Session().workspace_root()
    manifest = load_plot_config(Project().manifest(root), rendering.config)
    results = Results(root)
    for path in rendering.draw(results, manifest, source=source, chart=chart):
        logger.result(path.as_posix())
    for stamp in results.provenance:
        logger.result(stamp.footer)
