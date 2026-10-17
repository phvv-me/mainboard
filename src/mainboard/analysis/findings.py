"""What the engine reports and hands to fixes: findings placed by a span, and each whole row."""

from patos import FrozenModel
from pydantic import ConfigDict

PLACE = ("path", "start_line", "start_column")
ENDS = ("end_line", "end_column")


class Span(FrozenModel):
    """A source range, lines one-indexed and columns zero-indexed, as the kernel reports them."""

    path: str
    start_line: int
    start_column: int = 0
    end_line: int | None = None
    end_column: int | None = None


class Finding(FrozenModel):
    """One finding, the unit compared, reported and fixed."""

    rule: str
    span: Span
    message: str


type Cell = str | int | float | bool | None


class Hit(FrozenModel):
    """One finding's whole row, handed to its rule's fix; columns read as `hit.name`."""

    model_config = ConfigDict(extra="allow")

    def __getattr__(self, column: str) -> Cell:
        try:
            return (self.model_extra or {})[column]
        except KeyError:
            raise AttributeError(f"this finding's row has no column {column!r}") from None

    def sequence(self, column: str) -> list[Cell]:
        """The list a column holds, empty when the row has none."""
        return (self.model_extra or {}).get(column) or []

    def records(self, column: str) -> list[dict[str, int | str]]:
        """The structs a list column holds (`members`, `calls`), empty when the row has none."""
        return (self.model_extra or {}).get(column) or []

    @property
    def span(self) -> Span:
        """Where the finding sits."""
        row = self.model_extra or {}
        return Span(**{key: row[key] for key in (*PLACE, *ENDS) if key in row})

    def at(self, prefix: str) -> Span:
        """The span the row's `{prefix}_path`, `{prefix}_start_line`, ... columns hold."""
        row = self.model_extra or {}
        return Span(**{key: row[f"{prefix}_{key}"] for key in (*PLACE, *ENDS)})
