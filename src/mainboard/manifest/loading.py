import tomllib
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..core.errors import MissionError
from .held import Holdings
from .render.interpolate import Interpolator
from .schema.plot import PlotStyle
from .schema.root import Manifest
from .schema.workspace import Header

if TYPE_CHECKING:
    from pathlib import Path


def load(path: Path) -> Manifest:
    """Parse, interpolate, and validate the manifest at `path`.

    Stdlib tomllib (TOML 1.1 arrives with Python 3.15), then the `{{ }}`
    rendering pass, then schema
    validation, so a template error and a schema error each name their spot.
    The machines the workspace is holding join `[hosts]` last, so a held alias resolves
    wherever a declared one does.

    path: the workspace manifest file.
    """
    try:
        tree = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MissionError(f"no manifest at {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise MissionError(f"{path} is not valid TOML: {error}") from None
    rendered = Interpolator(path.parent).rendered(tree)
    try:
        manifest = Manifest.model_validate(rendered)
    except ValidationError as error:
        raise MissionError(f"{path} failed validation:\n{error}") from None
    return manifest.holding(Holdings(path.parent).profiles())


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
