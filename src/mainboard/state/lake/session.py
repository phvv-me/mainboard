"""A lake kept attached for as long as its holder keeps it, and the retry every operation on a
lake rides: SQLite refusing the catalog as locked is outlived by a fresh attach."""

import weakref
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from threading import RLock
from typing import TYPE_CHECKING, Any

import duckdb
from sqlalchemy import ClauseElement, Table
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from ...runtime.tree import FileBudget
from .attachment import Access
from .staging import insert

if TYPE_CHECKING:
    from .core import Lake

# How often a whole operation is retried when SQLite refused it as locked, and the longest pause
# between tries. SQLite answers a transaction that read before another writer committed with an
# immediate "database is locked" its busy timeout never waits out, since waiting cannot help that
# snapshot; a fresh attach reads the new one. Nothing was committed, so a retry cannot double.
_LOCKED_ATTEMPTS = 30
_LOCKED_WAIT_S = 1.0


class _Attachment:
    """The connection a session holds and the stack that detaches it.

    Kept apart from the session so its finalizer can close the attachment without holding the
    session: a finalizer that dropped only the stack left the closed connection on the session,
    and a session cached for the life of the process (the evidence keeper's) still held it when
    the interpreter's extension modules were torn down, which nanobind reports as a leak.
    """

    def __init__(self) -> None:
        self.stack = ExitStack()
        self.connection: duckdb.DuckDBPyConnection | None = None

    def close(self) -> None:
        """Detach and close the connection, and let it go."""
        self.stack.close()
        self.connection = None


class Session:
    """One lake attached read-write for as long as its owner keeps it, attached on first use.

    Attaching costs a tenth of a second, so a registry or a bus keeps one session rather than
    attaching per statement, and never attaches at all when it is never used (a host installing
    an environment builds a board that never touches its registry). A statement the catalog
    refused as locked or stale is run again on a fresh attach; a failed statement committed
    nothing, so running it again cannot duplicate a row.
    """

    def __init__(self, lake: Lake) -> None:
        self.lake = lake
        self._attachment = _Attachment()
        self._turn = RLock()
        # Detaching is the collector's job, not a caller's: otherwise only interpreter exit
        # releases the catalog.
        weakref.finalize(self, self._attachment.close)

    def _attached(self) -> duckdb.DuckDBPyConnection:
        """The connection, attaching first.

        A fresh workspace's lake is created on this use, and an older release's lake brought up to
        this schema.
        """
        attachment = self._attachment
        if attachment.connection is None:
            connection = attachment.stack.enter_context(
                self.lake.ready().attached(access=Access.WRITE)
            )
            self.lake.evolve(connection)
            attachment.connection = connection
        return attachment.connection

    def append(self, table: Table, rows: Iterable[Mapping[str, object]]) -> int:
        """Append `rows` to `table` in one commit, returning how many were appended."""
        staged = list(rows)
        return self.run(lambda connection: insert(connection, table, staged))

    def close(self) -> None:
        """Detach now; the next statement attaches again."""
        with self._turn:
            self._attachment.close()

    def rows(
        self, statement: ClauseElement | str, parameters: Iterable[object] = ()
    ) -> list[tuple[Any, ...]]:
        """What `statement` answers in this session."""
        bound = list(parameters)
        return self.run(
            lambda connection: self.lake.execute(connection, statement, bound).fetchall()
        )

    def run[T](self, statement: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """`statement` over this session's connection, reattached while the lake refuses it.

        One statement at a time: a DuckDB connection is not safe across threads (a sampler
        publishes from its own thread while its owner reads the stream), and a per-thread cursor
        would be closed under its thread by the reattach a stale catalog snapshot needs.
        """
        with self._turn, FileBudget.permitted():
            return patiently(_recoverable, before=self.close)(lambda: statement(self._attached()))


def _locked(fault: BaseException) -> bool:
    """Whether `fault` is SQLite refusing the catalog as locked, which a fresh attach outlives."""
    return isinstance(fault, duckdb.Error) and "database is locked" in str(fault)


def patiently(
    retry: Callable[[BaseException], bool] = _locked, *, before: Callable[[], None] | None = None
) -> Retrying:
    """Attempts of one operation, retried while `retry` accepts the fault, the last one raised.

    before: run between attempts, as a session detaching so the next attempt attaches afresh.
    """
    return Retrying(
        retry=retry_if_exception(retry),
        stop=stop_after_attempt(_LOCKED_ATTEMPTS),
        wait=wait_random_exponential(max=_LOCKED_WAIT_S),
        before_sleep=(lambda _state: before()) if before else None,
        reraise=True,
    )


def _recoverable(fault: BaseException) -> bool:
    """Whether a fresh attach outlives `fault`.

    It does for a locked catalog, or for inlined rows another process's maintenance flushed out
    from under an attached session's cached view of them.
    """
    return _locked(fault) or (
        isinstance(fault, duckdb.Error) and "ducklake_inlined_data" in str(fault)
    )
