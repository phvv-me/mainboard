"""A live query cursor, read in bounded chunks into a sink, hashed and stamped."""

import json
import sys
from typing import TYPE_CHECKING

import duckdb

from ...core.errors import MissionError
from ...log import logger
from ..stamps import Ending, ResultHash, Stamp, Value, canonical, normalized
from .sinks import Rendered, Sink, Streamed, Window, Windowed

_MAX_ROWS = 1000
_MAX_BYTES = 10000

if TYPE_CHECKING:
    from collections.abc import Sequence

_CHUNK = 16


def is_execution_timeout(fault: duckdb.Error) -> bool:
    """Local interrupts and Quack's forwarded interrupts/HTTP POST timeout faults."""
    return isinstance(fault, duckdb.InterruptException) or any(
        text in str(fault).casefold()
        for text in ("timeout was reached", "interrupted", "maximum execution time")
    )


class Query:
    """A live cursor, consumed in bounded chunks with bounded retained output.

    Caps cover encoded data rows, excluding header, omission marker and footer. Uncapped
    TSV/JSON writes each row immediately. After truncation at most 10,000 extra rows are
    counted and hashed; incomplete scans display the prefix only, never a misleading tail.
    JSON remains an array, with a truncation object when needed and its footer on stderr.
    """

    @classmethod
    def started(
        cls,
        connection: duckdb.DuckDBPyConnection,
        text: str,
        snapshot_id: int,
        *,
        timeout: float = 10,
        explain: bool = False,
    ) -> Query:
        """`text` run on `connection`.

        A blocking pipeline (an aggregate) runs inside `execute` and can time out there, before any
        row: the query then holds the statement's bound columns and renders as timed out, the same
        as one that timed out between rows.
        """
        try:
            return cls(connection.execute(text), snapshot_id, timeout=timeout, explain=explain)
        except duckdb.Error as fault:
            if not is_execution_timeout(fault):
                raise
        query = cls(connection, snapshot_id, timeout=timeout, explain=explain)
        query.columns = query._bound(connection, text)
        query.expired = True
        return query

    @property
    def header(self) -> str:
        return "\t".join(canonical(column) for column in self.columns)

    @staticmethod
    def _footed(stamp: Stamp, *, shown: int, mode: str | None) -> None:
        """The footer: the stamp, what was displayed and how much was left out."""
        stream = sys.stderr if mode == "json" else sys.stdout
        bound = "=" if stamp.complete else ">="
        stream.write(f"{stamp.footer}; displayed={shown}; omitted{bound}{stamp.rows - shown}\n")

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
        self.expired = False

    def _bound(self, connection: duckdb.DuckDBPyConnection, text: str) -> tuple[str, ...]:
        """The columns `text` binds to after it timed out.

        A Quack server that stops a query at its own limit aborts the transaction before any
        column is known, and binding anew would run the whole query once more, so the timeout is
        then the whole answer.
        """
        try:
            return tuple(connection.sql(text).columns)
        except duckdb.TransactionException:
            raise self._expiry() from None

    def _consumed(self, digest: ResultHash, sink: Sink, *, mode: str | None) -> Ending:
        """Fetch every row chunk by chunk into `sink`, hashing each; how the scan ended."""
        if self.expired:
            return Ending.TIMED_OUT
        while batch := self._fetched():
            if self._fed(batch, digest, sink, mode=mode):
                self.cursor.interrupt()
                return Ending.STOPPED
            sink.flushed()
        return Ending.COMPLETE if batch is not None else Ending.TIMED_OUT

    def _empty(self, digest: ResultHash, *, mode: str | None) -> Stamp:
        """A statement with no columns: `ok`, and a footer counting nothing."""
        stamp = digest.stamp(self.snapshot_id)
        if mode == "json":
            logger.result("[]\n", end="")
        stream = sys.stderr if mode == "json" else sys.stdout
        stream.write(f"ok\n{stamp.footer}; displayed=0; omitted=0\n")
        return stamp

    def _expiry(self) -> MissionError:
        """The refusal of a query that outlived its limit."""
        return MissionError(
            f"query timed out (client limit {self.timeout:g}s; Quack uses its server limit)"
        )

    def _fed(
        self, batch: Sequence[Sequence[Value]], digest: ResultHash, sink: Sink, *, mode: str | None
    ) -> bool:
        """Hash and hand over each row of `batch`; answer whether the sink wants no more."""
        for row in batch:
            cells, rendered = self._rendered(row, mode=mode)
            digest.add(cells)
            if sink.took(rendered, digest.rows):
                return True
        return False

    def _fetched(self) -> list[tuple[Value, ...]] | None:
        """The next chunk of rows, empty once read through, None once the query timed out."""
        try:
            return self.cursor.fetchmany(_CHUNK)
        except duckdb.Error as fault:
            if not is_execution_timeout(fault):
                raise
            return None

    def _rendered(
        self, row: Sequence[Value], *, mode: str | None
    ) -> tuple[tuple[str, ...], Rendered]:
        """One row's canonical cells, which the hash covers, and the row as it is printed."""
        cells = tuple(canonical(value) for value in row)
        displayed = tuple(str(value) for value in row) if self.explain else cells
        if mode != "json":
            return cells, Rendered(cells=displayed, text="\t".join(displayed))
        record = dict(zip(self.columns, (normalized(value) for value in row), strict=True))
        text = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        return cells, Rendered(cells=displayed, text=text)

    def render(
        self, *, mode: str | None = None, max_rows: int = _MAX_ROWS, max_bytes: int = _MAX_BYTES
    ) -> Stamp:
        """Print every row within the caps, then a footer stamping what was read.

        `mode` is TSV, `json` or `human`. A timed-out query refuses after printing.
        """
        digest = ResultHash(self.columns)
        if not self.columns:
            return self._empty(digest, mode=mode)
        sink: Sink = (
            Streamed(self, mode=mode)
            if max_rows <= 0 and max_bytes <= 0
            else Windowed(self, Window(max_rows, max_bytes=max_bytes), mode=mode)
        )
        ending = self._consumed(digest, sink, mode=mode)
        stamp = digest.stamp(self.snapshot_id, ending=ending)
        Query._footed(stamp, shown=sink.finished(stamp), mode=mode)
        if stamp.timed_out:
            raise self._expiry()
        return stamp
