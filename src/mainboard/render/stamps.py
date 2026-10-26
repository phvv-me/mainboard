"""Query results as printed cells, and the multiset stamp that identifies what was printed."""

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import StrEnum, auto
from typing import TYPE_CHECKING
from uuid import UUID

from patos import FrozenModel

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Sequence

type Key = (
    str | int | float | bool | Decimal | date | datetime | time | timedelta | UUID | bytes | None
)
type Value = Key | list[Value] | dict[Key, Value]
type Json = str | int | float | bool | None | list[Json] | dict[str, Json]

_MODULUS = 1 << 256


def normalized(value: Value) -> Json:
    """SQL values in portable display form; floats have twelve significant digits.

    Signed zero becomes zero, every NaN becomes `NaN`, and nested maps have sorted keys.
    Display precision intentionally hides ordinary parallel DOUBLE aggregation roundoff;
    values straddling a rounding boundary can still differ.
    """
    match value:
        case float():
            return _float(value)
        case list():
            return [normalized(item) for item in value]
        case dict():
            return {
                str(key): normalized(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        case _:
            return value if value is None or isinstance(value, str | int) else str(value)


def _float(value: float) -> Json:
    """A float for display: `NaN`, `Infinity`, or twelve significant digits, signed zero zero."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return 0.0 if value == 0 else float(format(value, ".12g"))


def canonical(value: Value) -> str:
    """A printed TSV cell: SQL NULL is bare `NULL`, literal `NULL` is quoted.

    Empty strings stay empty; tabs, newlines, carriage returns and leading quotes are escaped.
    """
    displayed = normalized(value)
    if displayed is None:
        return "NULL"
    text = _text(displayed)
    return (
        json.dumps(text, ensure_ascii=False)
        if text == "NULL" or text.startswith('"') or any(c in text for c in "\t\n\r")
        else text
    )


def _text(displayed: Json) -> str:
    """A displayed value's cell text, before any quoting."""
    match displayed:
        case str():
            return displayed
        case int():
            return str(displayed)
        case list() | dict():
            return json.dumps(displayed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        case _:
            return format(displayed, ".12g")


class Ending(StrEnum):
    """How a scan ended: it read every row, it was stopped early, or it timed out."""

    COMPLETE = auto()
    STOPPED = auto()
    TIMED_OUT = auto()


class Stamp(FrozenModel):
    """Rendered provenance, not an integrity seal; incomplete hashes cover counted prefixes.

    `snapshot_id` pins lake tables only. File-backed views and jobs are not snapshot-pinned.
    """

    rows: int
    sha256: str
    snapshot_id: int
    complete: bool = True
    timed_out: bool = False

    @property
    def footer(self) -> str:
        count = f"{self.rows} rows" if self.complete else f"at least {self.rows} rows"
        stopped = (
            " (query timed out)"
            if self.timed_out
            else ("" if self.complete else " (query stopped early)")
        )
        partial = "" if self.complete else " (partial)"
        return (
            f"# {count}{stopped}; sha256={self.sha256}{partial}; snapshot_id={self.snapshot_id}"
            " (lake.* only; file views/jobs unpinned)"
        )


class ResultHash:
    """Portable multiset SHA-256 over canonical printed cells, including duplicate rows.

    Each row is UTF-8 compact JSON of its ordered TSV cells, SHA-256 hashed, read big-endian,
    then added modulo 2**256. The final SHA-256 covers a version tag, column names, row count
    and that sum. Row order is irrelevant; NULL differs from empty text and literal NULL.
    This identifies the rendered values, never raw floating-point bits or DuckDB hashes.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        if len({name.casefold() for name in columns}) != len(columns):
            raise MissionError(
                "query has duplicate column names; give each column a unique AS alias"
            )
        self.rows = 0
        self._sum = 0

    def add(self, cells: Sequence[str]) -> None:
        encoded = json.dumps(cells, ensure_ascii=False, separators=(",", ":")).encode()
        self._sum = (self._sum + int.from_bytes(hashlib.sha256(encoded).digest())) % _MODULUS
        self.rows += 1

    def stamp(self, snapshot_id: int, *, ending: Ending = Ending.COMPLETE) -> Stamp:
        document = json.dumps(
            ["mainboard-rendered-multiset-v1", self.columns, self.rows, f"{self._sum:064x}"],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return Stamp(
            rows=self.rows,
            sha256=hashlib.sha256(document).hexdigest(),
            snapshot_id=snapshot_id,
            complete=ending is Ending.COMPLETE,
            timed_out=ending is Ending.TIMED_OUT,
        )
