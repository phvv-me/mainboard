"""Native Seaborn marks, with explicit data mappings and no statistical estimation."""

from pathlib import Path
from typing import Literal

from patos import FrozenModel
from pydantic import JsonValue, model_validator


class Layer(FrozenModel):
    """A native mark constructor; variables name SQL columns, kws configure the mark."""

    mark: Literal[
        "Dot",
        "Dots",
        "Line",
        "Lines",
        "Path",
        "Paths",
        "Dash",
        "Range",
        "Bar",
        "Bars",
        "Area",
        "Band",
        "Text",
    ]
    variables: dict[str, str] = {}
    kws: dict[str, JsonValue] = {}
    moves: dict[Literal["Dodge", "Stack"], dict[str, JsonValue]] = {}
    file: Path | None = None
    sql: str = ""

    @model_validator(mode="after")
    def source(self) -> Layer:
        """Omit data to inherit the panel; otherwise supply exactly one SELECT source."""
        if self.file is not None and self.sql.strip():
            raise ValueError("layer accepts a SQL statement or file, not both")
        return self
