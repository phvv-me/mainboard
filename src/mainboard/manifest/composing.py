"""A workspace manifest and the members it declares, read without the lake.

The lake itself reads its settings from here, so this module never reaches the lake: `load`
lays the held machines over what it composes.
"""

from typing import TYPE_CHECKING

from ..core.errors import MissionError
from .members import Composition
from .parsing import rendered, validated

if TYPE_CHECKING:
    from pathlib import Path


def composition(path: Path) -> Composition:
    """The manifest at `path` and the members it declares, before they are folded in."""
    try:
        tree = rendered(path)
    except FileNotFoundError:
        raise MissionError(f"no manifest at {path}") from None
    return Composition(path.parent, validated(path, tree))
