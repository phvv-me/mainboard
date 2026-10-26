"""Where a query's rendered rows go: printed as they arrive, or windowed and printed at the end."""

import json
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from patos import FrozenModel

from ...log import logger
from ..human import render_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..stamps import Json, Stamp
    from .core import Query

_EXTRA_ROWS = 10000


class Rendered(FrozenModel):
    cells: tuple[str, ...]
    text: str

    @property
    def size(self) -> int:
        return len(self.text.encode()) + 1


class Window:
    """An exact prefix and suffix fitting both caps, with honest omission accounting."""

    def __init__(self, max_rows: int, *, max_bytes: int) -> None:
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.head: list[Rendered] = []
        self.tail: deque[Rendered] = deque()
        self.used = self.tail_bytes = self.truncated_at = 0
        self.tail_limit = min(5, max_rows // 2) if max_rows > 0 else 5
        self.limits: list[str] = []

    def _overfull(self) -> bool:
        """Whether the head and the suffix together pass the byte cap."""
        return self.max_bytes > 0 and self.used + self.tail_bytes > self.max_bytes

    def _reserved(self) -> None:
        """Move rows from the head's end into the suffix while it wants them, keeping the first.

        The suffix wants rows until it holds its limit, or while the head fills half the bytes.
        """
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

    def _trailed(self, row: Rendered) -> None:
        """Keep `row` as the newest of the suffix, dropping the oldest past either cap."""
        if self.tail_limit and (self.max_bytes <= 0 or row.size <= self.max_bytes - self.used):
            self.tail.append(row)
            self.tail_bytes += row.size
        else:
            self.tail.clear()
            self.tail_bytes = 0
        while self.tail and (len(self.tail) > self.tail_limit or self._overfull()):
            if self._overfull() and (limit := f"--max-bytes {self.max_bytes}") not in self.limits:
                self.limits.append(limit)
            self.tail_bytes -= self.tail.popleft().size

    def _truncated(self, row: Rendered, counted: int) -> bool:
        """Whether `row` no longer fits the head, truncating there the first time it does.

        Truncating names the caps it hit and reserves a suffix from the head when there is room
        for one, keeping the first row.
        """
        if self.truncated_at:
            return True
        if self.max_rows > 0 and len(self.head) >= self.max_rows:
            self.limits.append(f"--max-rows {self.max_rows}")
        if self.max_bytes > 0 and self.used + row.size > self.max_bytes:
            self.limits.append(f"--max-bytes {self.max_bytes}")
        if not self.limits:
            return False
        self.truncated_at = counted
        self._reserved()
        return True

    def details(self, stamp: Stamp) -> dict[str, Json]:
        shown = len(self.head) + len(self.tail)
        return {
            "omitted": stamp.rows - shown,
            "displayed": shown,
            "limits": [*self.limits],
            **stamp.model_dump(),
        }

    def keep(self, row: Rendered, counted: int) -> bool:
        """Retain this row if it fits; return whether the extra-row budget is exhausted."""
        if not self._truncated(row, counted):
            self.head.append(row)
            self.used += row.size
            return False
        self._trailed(row)
        return counted - self.truncated_at >= _EXTRA_ROWS

    def marker(self, stamp: Stamp) -> str:
        shown = len(self.head) + len(self.tail)
        bound = "at least " if not stamp.complete else ""
        reasons = [*self.limits]
        if shown <= 1:
            reasons.append(f"{shown} {'row' if shown == 1 else 'rows'} displayed")
        suffix = " (" + "; ".join(reasons) + ")" if reasons else ""
        return f"… {bound}{stamp.rows - shown} rows omitted{suffix} …"


class Sink(ABC):
    """Where a query's rendered rows go as they are read."""

    @abstractmethod
    def finished(self, stamp: Stamp) -> int:
        """Close the output once the scan ended as `stamp` says; answer the rows displayed."""

    @abstractmethod
    def flushed(self) -> None:
        """Show what one chunk left pending."""

    @abstractmethod
    def took(self, row: Rendered, counted: int) -> bool:
        """Take one row, the `counted`th read; answer whether the scan should stop."""

    def __init__(self, query: Query, *, mode: str | None) -> None:
        self.query, self.mode = query, mode

    def table(self, rows: Sequence[Rendered | dict[str, str]]) -> None:
        """`rows` as one rich table, a marker row given as a mapping."""
        records = [
            row if isinstance(row, dict) else dict(zip(self.query.columns, row.cells, strict=True))
            for row in rows
        ]
        render_table(records, fields=self.query.columns, title="results")


class Streamed(Sink):
    """Every row printed as it arrives: TSV lines, one JSON array, or a rich table per chunk."""

    def __init__(self, query: Query, *, mode: str | None) -> None:
        super().__init__(query, mode=mode)
        self.first = True
        self.pending: list[Rendered] = []
        if mode != "human":
            logger.result("[" if mode == "json" else query.header + "\n", end="")

    def finished(self, stamp: Stamp) -> int:
        if self.mode == "human" and not stamp.rows:
            self.table([])
        if self.mode == "json":
            if not stamp.complete:
                marker = json.dumps({"_truncated": {"omitted": 0, **stamp.model_dump()}})
                logger.result(("" if self.first else ",") + marker, end="")
            logger.result("]\n", end="")
        return stamp.rows

    def flushed(self) -> None:
        if self.mode == "human":
            self.table(self.pending)
            self.pending.clear()

    def took(self, row: Rendered, counted: int) -> bool:
        if self.mode == "human":
            self.pending.append(row)
        else:
            text = (
                ("" if self.first else ",") + row.text if self.mode == "json" else row.text + "\n"
            )
            logger.result(text, end="")
        self.first = False
        return False


class Windowed(Sink):
    """A bounded prefix and suffix of the rows, printed once the scan ends."""

    def __init__(self, query: Query, window: Window, *, mode: str | None) -> None:
        super().__init__(query, mode=mode)
        self.window = window

    def _json(self, stamp: Stamp, *, omitted: int) -> None:
        items = [row.text for row in self.window.head]
        if omitted or not stamp.complete:
            details = {**self.window.details(stamp), "columns": self.query.columns}
            items.append(json.dumps({"_truncated": details}))
        items.extend(row.text for row in self.window.tail)
        logger.result("[" + ",".join(items) + "]\n", end="")

    def _rows(self, stamp: Stamp, *, omitted: int) -> None:
        marker = self.window.marker(stamp)
        if self.mode == "human":
            middle = [{self.query.columns[0]: marker}] if omitted else []
            self.table([*self.window.head, *middle, *self.window.tail])
            return
        logger.result(self.query.header + "\n", end="")
        lines = [row.text for row in self.window.head] + ([marker] if omitted else [])
        for line in [*lines, *(row.text for row in self.window.tail)]:
            logger.result(line + "\n", end="")

    def finished(self, stamp: Stamp) -> int:
        if not stamp.complete:
            self.window.tail.clear()
        shown = len(self.window.head) + len(self.window.tail)
        if self.mode == "json":
            self._json(stamp, omitted=stamp.rows - shown)
        else:
            self._rows(stamp, omitted=stamp.rows - shown)
        return shown

    def flushed(self) -> None:
        """Nothing: the window prints once the scan ends."""

    def took(self, row: Rendered, counted: int) -> bool:
        return self.window.keep(row, counted)
