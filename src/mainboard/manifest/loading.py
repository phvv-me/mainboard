from typing import TYPE_CHECKING

from .composing import composition
from .schema.figures.plot import PlotStyle
from .schema.project.workspace import Header
from .schema.root import Manifest

if TYPE_CHECKING:
    from pathlib import Path


def load(path: Path) -> Manifest:
    """Parse, interpolate, compose and validate the manifest at `path`, errors naming their spot.

    Members join first (see `manifest.members`). The machines the workspace holds live in its
    lake, which reads this manifest, so `state.holds.holding` lays them over it.
    """
    return composition(path).composed()


def load_plot_config(root: Path, config: Path | None = None) -> Manifest:
    """Layer project plot settings over the shared manifest, without selecting an env.

    root: workspace manifest path; a plain SQL plot also works without a manifest.
    config: optional project style/figure file, resolved from the caller's cwd.
    """
    shared = load(root) if root.is_file() else Manifest(workspace=Header(name="plots"))
    if config is None:
        return shared
    project = load(config)
    styles = {
        name: style.merged(shared.plots.get(name, PlotStyle()))
        for name, style in project.plots.items()
    }
    return project.model_copy(
        update={"plots": shared.plots | styles, "figures": shared.figures | project.figures}
    )
