"""A selected result as named columns of plain values, the shape Seaborn is handed."""

import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Self, cast
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..state.relations import Relation

# What DuckDB hands back for one cell.
type Value = (
    str
    | int
    | float
    | bool
    | Decimal
    | date
    | datetime
    | time
    | timedelta
    | UUID
    | bytes
    | None
    | list[Value]
    | dict[str, Value]
)

# The DuckDB type ids a plot reads as numbers.
_NUMERIC = frozenset(
    {
        "tinyint",
        "smallint",
        "integer",
        "bigint",
        "hugeint",
        "utinyint",
        "usmallint",
        "uinteger",
        "ubigint",
        "uhugeint",
        "float",
        "double",
        "decimal",
    }
)


class Columns:
    """Named columns read in whole from one relation, each with whether it holds numbers.

    values: each column's values in row order.
    numbers: the columns whose type is numeric.
    """

    def __init__(
        self,
        values: dict[str, list[Value]] | None = None,
        numbers: frozenset[str] = frozenset(),
    ) -> None:
        self.values = values or {}
        self.numbers = numbers

    @classmethod
    def of(cls, relation: Relation) -> Self:
        rows = relation.fetchall()
        return cls(
            {name: [row[index] for row in rows] for index, name in enumerate(relation.columns)},
            frozenset(
                name
                for name, kind in zip(relation.columns, relation.types, strict=True)
                if kind.id in _NUMERIC
            ),
        )

    @property
    def height(self) -> int:
        return len(next(iter(self.values.values()), []))

    def __getitem__(self, name: str) -> list[Value]:
        return self.values[name]

    def floats(self, name: str) -> list[float]:
        """A numeric column's values as floats; refused for any other column."""
        if name not in self.numbers:
            raise ValueError(f"plot column {name!r} must be numeric")
        return [float(cast("float | Decimal", value)) for value in self.values[name]]

    def select(self, names: Sequence[str]) -> Self:
        """Only `names`, each once, in that order."""
        kept = list(dict.fromkeys(names))
        return type(self)({name: self.values[name] for name in kept}, self.numbers & set(kept))

    def unique(self, name: str) -> list[Value]:
        """`name`'s distinct values in the order they first appear."""
        return list(dict.fromkeys(self.values[name]))

    def distinct(self, names: Sequence[str]) -> int:
        """How many distinct combinations of `names` the rows hold."""
        return len(set(zip(*(self.values[name] for name in dict.fromkeys(names)), strict=True)))

    def plain(self) -> dict[str, list[Value]]:
        """Every column, decimal SQL literals as floats, so a plot reads them without changing the
        selected evidence."""
        return {
            name: [float(value) if isinstance(value, Decimal) else value for value in column]
            for name, column in self.values.items()
        }

    def checked(self) -> None:
        """Refuse missing and nonfinite values before any plotting library sees them."""
        if not self.height or any(None in column for column in self.values.values()):
            raise ValueError("plot columns must contain rows without null values")
        if not all(math.isfinite(value) for name in self.numbers for value in self.floats(name)):
            raise ValueError("plot columns must contain finite values")
