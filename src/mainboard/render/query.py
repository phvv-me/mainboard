import hashlib
import json
import math
import sys
from collections import deque
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

import duckdb
from patos import FrozenModel

from ..core.errors import MissionError
from .human import render_table

if TYPE_CHECKING:
    from collections.abc import Sequence

type Key = (
    str | int | float | bool | Decimal | date | datetime | time | timedelta | UUID | bytes | None
)
type Value = Key | list[Value] | dict[Key, Value]
type Json = str | int | float | bool | None | list[Json] | dict[str, Json]

_CHUNK = 16
_EXTRA_ROWS = 10000
_MODULUS = 1 << 256


def normalized(value: Value) -> Json:
    """SQL values in portable display form; floats have twelve significant digits.

    Signed zero becomes zero, every NaN becomes `NaN`, and nested maps have sorted keys.
    Display precision intentionally hides ordinary parallel DOUBLE aggregation roundoff;
    values straddling a rounding boundary can still differ.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return 0.0 if value == 0 else float(format(value, ".12g"))
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalized(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def canonical(value: Value) -> str:
    """A printed TSV cell: SQL NULL is bare `NULL`, literal `NULL` is quoted.

    Empty strings stay empty; tabs, newlines, carriage returns and leading quotes are escaped.
    """
    displayed = normalized(value)
    if displayed is None:
        return "NULL"
    text = (
        json.dumps(displayed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if isinstance(displayed, list | dict)
        else format(displayed, ".12g")
        if isinstance(displayed, float)
        else str(displayed)
    )
    return (
        json.dumps(text, ensure_ascii=False)
        if text == "NULL" or text.startswith('"') or any(c in text for c in "\t\n\r")
        else text
    )


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

    def stamp(self, snapshot_id: int, *, complete: bool = True, timed_out: bool = False) -> Stamp:
        document = json.dumps(
            ["mainboard-rendered-multiset-v1", self.columns, self.rows, f"{self._sum:064x}"],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return Stamp(
            rows=self.rows,
            sha256=hashlib.sha256(document).hexdigest(),
            snapshot_id=snapshot_id,
            complete=complete,
            timed_out=timed_out,
        )


def execution_timeout(fault: duckdb.Error) -> bool:
    """Local interrupts and Quack's forwarded interrupts/HTTP POST timeout faults."""
    return isinstance(fault, duckdb.InterruptException) or any(
        text in str(fault).casefold()
        for text in ("timeout was reached", "interrupted", "maximum execution time")
    )


class _Rendered(FrozenModel):
    cells: tuple[str, ...]
    text: str

    @property
    def size(self) -> int:
        return len(self.text.encode()) + 1


class _Window:
    """An exact prefix and suffix fitting both caps, with honest omission accounting."""

    def __init__(self, max_rows: int, max_bytes: int) -> None:
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.head: list[_Rendered] = []
        self.tail: deque[_Rendered] = deque()
        self.used = self.tail_bytes = self.truncated_at = 0
        self.tail_limit = min(5, max_rows // 2) if max_rows > 0 else 5
        self.limits: list[str] = []

    def keep(self, row: _Rendered, counted: int) -> bool:
        """Retain this row if it fits; return whether the extra-row budget is exhausted."""
        if not self.truncated_at:
            if self.max_rows > 0 and len(self.head) >= self.max_rows:
                self.limits.append(f"--max-rows {self.max_rows}")
            if self.max_bytes > 0 and self.used + row.size > self.max_bytes:
                self.limits.append(f"--max-bytes {self.max_bytes}")
            if not self.limits:
                self.head.append(row)
                self.used += row.size
                return False
            self.truncated_at = counted
            # Reserve a suffix only when there is room for one, preserving the first row.
            while (
                self.tail_limit
                and len(self.head) > 1
                and (
                    len(self.tail) < self.tail_limit
                    or (self.max_bytes > 0 and self.used > self.max_bytes // 2)
                )
            ):
                kept = self.head.pop()
                self.used -= kept.size
                self.tail.appendleft(kept)
                self.tail_bytes += kept.size
        if self.tail_limit and (self.max_bytes <= 0 or row.size <= self.max_bytes - self.used):
            self.tail.append(row)
            self.tail_bytes += row.size
        else:
            self.tail.clear()
            self.tail_bytes = 0
        while self.tail and (
            len(self.tail) > self.tail_limit
            or (self.max_bytes > 0 and self.used + self.tail_bytes > self.max_bytes)
        ):
            if self.max_bytes > 0 and self.used + self.tail_bytes > self.max_bytes:
                limit = f"--max-bytes {self.max_bytes}"
                if limit not in self.limits:
                    self.limits.append(limit)
            self.tail_bytes -= self.tail.popleft().size
        return counted - self.truncated_at >= _EXTRA_ROWS

    def details(self, stamp: Stamp) -> dict[str, Json]:
        shown = len(self.head) + len(self.tail)
        return {
            "omitted": stamp.rows - shown,
            "displayed": shown,
            "limits": [*self.limits],
            **stamp.model_dump(),
        }

    def marker(self, stamp: Stamp) -> str:
        shown = len(self.head) + len(self.tail)
        bound = "at least " if not stamp.complete else ""
        reasons = [*self.limits]
        if shown <= 1:
            reasons.append(f"{shown} {'row' if shown == 1 else 'rows'} displayed")
        suffix = " (" + "; ".join(reasons) + ")" if reasons else ""
        return f"… {bound}{stamp.rows - shown} rows omitted{suffix} …"


class Query:
    """A live cursor, consumed in bounded chunks with bounded retained output.

    Caps cover encoded data rows, excluding header, omission marker and footer. Uncapped
    TSV/JSON writes each row immediately. After truncation at most 10,000 extra rows are
    counted and hashed; incomplete scans display the prefix only, never a misleading tail.
    JSON remains an array, with a truncation object when needed and its footer on stderr.
    """

    def __init__(
        self,
        cursor: duckdb.DuckDBPyConnection,
        snapshot_id: int,
        *,
        timeout: float = 10,
        explain: bool = False,
    ) -> None:
        self.cursor = cursor
        self.snapshot_id = snapshot_id
        self.columns = tuple(column[0] for column in (cursor.description or ()))
        ResultHash(self.columns)  # Reject ambiguous names before any row is displayed.
        self.timeout = timeout
        self.explain = explain

    @property
    def header(self) -> str:
        return "\t".join(canonical(column) for column in self.columns)

    def render(
        self, *, mode: str | None = None, max_rows: int = 1000, max_bytes: int = 10000
    ) -> Stamp:
        digest = ResultHash(self.columns)
        if not self.columns:
            stamp = digest.stamp(self.snapshot_id)
            if mode == "json":
                sys.stdout.write("[]\n")
            stream = sys.stderr if mode == "json" else sys.stdout
            stream.write(f"ok\n{stamp.footer}; displayed=0; omitted=0\n")
            return stamp
        window = _Window(max_rows, max_bytes)
        complete = True
        timed_out = False
        uncapped = max_rows <= 0 and max_bytes <= 0
        first = True
        if uncapped and mode != "human":
            sys.stdout.write("[" if mode == "json" else self.header + "\n")
        while True:
            try:
                batch = self.cursor.fetchmany(_CHUNK)
            except duckdb.Error as fault:
                if not execution_timeout(fault):
                    raise
                complete = False
                timed_out = True
                break
            if not batch:
                break
            for row in batch:
                cells = tuple(canonical(value) for value in row)
                digest.add(cells)
                displayed = tuple(str(value) for value in row) if self.explain else cells
                text = (
                    json.dumps(
                        dict(zip(self.columns, (normalized(value) for value in row), strict=True)),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if mode == "json"
                    else "\t".join(displayed)
                )
                rendered = _Rendered(cells=displayed, text=text)
                if uncapped:
                    if mode == "human":
                        window.head.append(rendered)
                    else:
                        sys.stdout.write(
                            ("" if first else ",") + text if mode == "json" else text + "\n"
                        )
                    first = False
                    continue
                if window.keep(rendered, digest.rows):
                    self.cursor.interrupt()
                    complete = False
                    break
            if uncapped and mode == "human":
                render_table(
                    [dict(zip(self.columns, row.cells, strict=True)) for row in window.head],
                    fields=self.columns,
                    title="results",
                )
                window.head.clear()
            if not complete:
                break
        stamp = digest.stamp(self.snapshot_id, complete=complete, timed_out=timed_out)
        if uncapped:
            if mode == "human" and not stamp.rows:
                render_table([], fields=self.columns, title="results")
            if mode == "json":
                if not complete:
                    marker = json.dumps({"_truncated": {"omitted": 0, **stamp.model_dump()}})
                    sys.stdout.write(("" if first else ",") + marker)
                sys.stdout.write("]\n")
        else:
            if not complete:
                window.tail.clear()
            self._display(window, stamp, mode=mode)
        stream = sys.stderr if mode == "json" else sys.stdout
        shown = stamp.rows if uncapped else len(window.head) + len(window.tail)
        bound = "=" if complete else ">="
        stream.write(f"{stamp.footer}; displayed={shown}; omitted{bound}{stamp.rows - shown}\n")
        if timed_out:
            raise MissionError(
                f"query timed out (client limit {self.timeout:g}s; Quack uses its server limit)"
            )
        return stamp

    def _display(self, window: _Window, stamp: Stamp, *, mode: str | None) -> None:
        head, tail = window.head, window.tail
        omitted = stamp.rows - len(head) - len(tail)
        marker = window.marker(stamp)
        if mode == "json":
            items = [row.text for row in head]
            if omitted or not stamp.complete:
                items.append(
                    json.dumps({"_truncated": {**window.details(stamp), "columns": self.columns}})
                )
            items.extend(row.text for row in tail)
            sys.stdout.write("[" + ",".join(items) + "]\n")
        elif mode == "human":
            records = [dict(zip(self.columns, row.cells, strict=True)) for row in head]
            if omitted:
                records.append({self.columns[0]: marker})
            records.extend(dict(zip(self.columns, row.cells, strict=True)) for row in tail)
            render_table(records, fields=self.columns, title="results")
        else:
            sys.stdout.write(self.header + "\n")
            for row in head:
                sys.stdout.write(row.text + "\n")
            if omitted:
                sys.stdout.write(marker + "\n")
            for row in tail:
                sys.stdout.write(row.text + "\n")
