import tomllib
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..core.errors import MissionError
from .render.interpolate import Interpolator
from .schema.root import Manifest

if TYPE_CHECKING:
    from pathlib import Path

    from .render.interpolate import Json


def rendered(path: Path) -> dict[str, Json]:
    """The manifest at `path` parsed with every template rendered, `config_root` its directory.

    Without `[workspace]` the workspace is named after its directory. A missing file raises
    `FileNotFoundError`, which a workspace refuses and a member reads as having no manifest.
    """
    try:
        tree = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise MissionError(f"{path} is not valid TOML: {error}") from None
    tree = Interpolator(path.parent).rendered(tree)
    tree.setdefault("workspace", {"name": path.resolve().parent.name})
    return tree


def validated(path: Path, tree: dict[str, Json]) -> Manifest:
    """`tree`, read from `path`, as a manifest, each validation error naming its spot."""
    try:
        return Manifest.model_validate(tree)
    except ValidationError as error:
        raise MissionError(f"{path} failed validation:\n{error}") from None
