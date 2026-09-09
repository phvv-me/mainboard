"""A named figure combines native panel layouts without combining measurement units."""

from pathlib import Path

from patos import FrozenModel
from pydantic import JsonValue, model_validator

from .panel import Panel


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
        for name in names:
            positions = [
                (r, c)
                for r, row in enumerate(cells)
                for c, value in enumerate(row)
                if value == name
            ]
            rows, columns = zip(*positions, strict=True)
            if len(positions) != (max(rows) - min(rows) + 1) * (max(columns) - min(columns) + 1):
                raise ValueError(f"mosaic panel {name!r} is not a rectangle")
        return self
