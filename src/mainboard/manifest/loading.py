from typing import TYPE_CHECKING

from ..core.errors import MissionError
from .held import Holdings
from .members import Composition
from .parsing import rendered, validated
from .schema.plot import PlotStyle
from .schema.root import Manifest
from .schema.workspace import Header

if TYPE_CHECKING:
    from pathlib import Path


def load(path: Path) -> Manifest:
    """Parse, interpolate, compose and validate the manifest at `path`, errors naming their spot.

    Members join first (see `manifest.members`), then held machines join `[hosts]`, so a held
    alias resolves like a declared one.
    """
    composed = composition(path).composed()
    return composed.holding(Holdings(path.parent).profiles())


def composition(path: Path) -> Composition:
    """The manifest at `path` and the members it declares, before they are folded in."""
    try:
        tree = rendered(path)
    except FileNotFoundError:
        raise MissionError(f"no manifest at {path}") from None
    return Composition(path.parent, validated(path, tree))


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
