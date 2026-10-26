"""A named figure combines native panel layouts without combining measurement units."""

from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import JsonValue, model_validator

from .panel import Panel

if TYPE_CHECKING:
    from collections.abc import Sequence


class FigureSpec(FrozenModel):
    """A rectangular mosaic, with repeated names spanning adjacent panel cells."""

    style: str = ""
    out: tuple[Path, ...] = ()
    mosaic: tuple[tuple[str, ...], ...] = ()
    layout: dict[str, JsonValue] = {"layout": "constrained"}
    gridspec: dict[str, JsonValue] = {}
    panels: dict[str, Panel]

    @model_validator(mode="after")
    def placement(self) -> FigureSpec:
        """Refuse missing, duplicated, or nonrectangular panel ownership."""
        if not self.panels:
            raise ValueError("figure needs at least one panel")
        cells = self.mosaic or (tuple(self.panels),)
        if not cells[0] or len({len(row) for row in cells}) != 1:
            raise ValueError("figure mosaic must be rectangular")
        names = {name for row in cells for name in row} - {"."}
        if names != set(self.panels):
            raise ValueError("figure mosaic must contain exactly its named panels")
        ragged = [name for name in names if not self._is_rectangle(cells, name)]
        if ragged:
            raise ValueError(f"mosaic panel {ragged[0]!r} is not a rectangle")
        return self

    @staticmethod
    def _is_rectangle(cells: Sequence[Sequence[str]], name: str) -> bool:
        """Whether the cells named `name` fill their bounding box exactly."""
        positions = [
            (row_index, column_index)
            for row_index, row in enumerate(cells)
            for column_index, value in enumerate(row)
            if value == name
        ]
        rows, columns = zip(*positions, strict=True)
        return len(positions) == (max(rows) - min(rows) + 1) * (max(columns) - min(columns) + 1)
