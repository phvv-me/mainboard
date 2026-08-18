import tomllib
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..core.errors import MissionError
from .render.interpolate import Interpolator
from .schema.root import Manifest

if TYPE_CHECKING:
    from pathlib import Path


def load(path: Path) -> Manifest:
    """Parse, interpolate, and validate the manifest at `path`.

    Stdlib tomllib (TOML 1.1 arrives with Python 3.15), then the `{{ }}`
    rendering pass, then schema
    validation, so a template error and a schema error each name their spot.

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
        return Manifest.model_validate(rendered)
    except ValidationError as error:
        raise MissionError(f"{path} failed validation:\n{error}") from None
