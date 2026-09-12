"""One selected table and its native layers, axes, and optional facets."""

from pathlib import Path

from patos import FrozenModel
from pydantic import JsonValue, model_validator

from .layer import Layer


class Panel(FrozenModel):
    """Paths retain caller-cwd semantics; facet, share and axis use native library keywords."""

    file: Path | None = None
    sql: str = ""
    variables: dict[str, str] = {}
    layers: tuple[Layer, ...]
    facet: dict[str, JsonValue] = {}
    share: dict[str, JsonValue] = {}
    order: dict[str, tuple[str, ...]] = {}
    axis: dict[str, JsonValue] = {}
    ticks: dict[str, JsonValue] = {}
    grid: dict[str, JsonValue] = {}
    legend: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def source(self) -> Panel:
        """A panel reads exactly one SELECT source and draws at least one layer."""
        if (self.file is None) == (not self.sql.strip()):
            raise ValueError("panel needs exactly one SQL statement or file")
        if not self.layers:
            raise ValueError("panel needs at least one layer")
        return self
