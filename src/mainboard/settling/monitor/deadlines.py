"""What a deadline ends: a rented machine at its release deadline, and a settlement pass at its
own."""

import os
import signal
import threading
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from sqlite3 import Error as SQLiteError
from time import monotonic
from typing import TYPE_CHECKING, Final

from ...core.errors import MissionError
from ...dispatch.backends.base import route
from ...dispatch.jobs import vocabulary
from ...log import logger
from .constants import COPIED, QUEUED
from .settlement import failure
from .sweep import uncreated

if TYPE_CHECKING:
    from collections.abc import Generator

    from ...dispatch.backends.base import ProviderBackend
    from ...dispatch.state import Failed, RunRecord
    from .settlement import Settlement

# M34's bridge, not worker supervision: a pass ends before the settler's next one (`job list
# --every` suggests 20m), and a Python watchdog runs only while DuckDB or ssh release the GIL.
# M24 must contain native stalls and supervise the processes a pass starts.
_PASS_SECONDS: Final = 900.0
_PARENT_POLL: Final = 0.25


class Deadlines:
    """The rentals whose release deadline passed, released before a pass probes any host or
    waits for the settlement lock.

    At this boundary the reserved deletion interval has begun, so no unbounded transfer starts.
    Previously copied evidence survives; missing evidence stays explicitly unverified. A provider
    failure keeps the exact handle tracked for retry.
    """

    def __init__(self, settlement: Settlement) -> None:
        self.settlement = settlement

    def _closed(self, current: RunRecord, *, copied: bool, detail: str) -> None:
        """Book a rental released at its deadline: timed out, its evidence as it stands."""
        current = self.settlement.cache.resolve(
            current, vocabulary.Word.TIMEOUT, None, vocabulary.Word.TIMEOUT
        )
        status = "verified" if copied else "unverified"
        self.settlement.evidence(current, (), status=status, detail=detail)
        self.settlement.cache.report(current, current.verdict or vocabulary.Word.TIMEOUT)

    def _expire(self, record: RunRecord) -> Failed | None:
        """Release one rental past its deadline, if it is still the submission that leased it."""
        try:
            leased = self._leased(record)
        except (MissionError, OSError, ValueError, LookupError, SQLiteError) as fault:
            return failure(
                record, f"deadline identity check failed; release remains pending: {fault}"
            )
        if leased is None or leased[1].submitted_at != record.submitted_at:
            return None
        backend, current = leased
        if current.lease is None or not current.lease.expired:
            return None
        return failure(record, self._released(record, current=current, backend=backend))

    def _leased(self, record: RunRecord) -> tuple[type[ProviderBackend], RunRecord] | None:
        """The provider leasing `record`'s run and that run as the cache now holds it.

        None for a run no provider leases.
        """
        backend = route(record.kind)
        if backend == QUEUED:
            return None
        return backend, self.settlement.cache.run(record.handle, record.target)

    def _released(
        self, record: RunRecord, *, current: RunRecord, backend: type[ProviderBackend]
    ) -> str:
        """Cancel a rental at its release deadline, answering what happened to it.

        At this boundary the reserved deletion interval has begun, so no unbounded transfer
        starts. Previously copied evidence survives; missing evidence stays explicitly
        unverified. A provider failure keeps the exact handle tracked for retry.
        """
        copied = current.evidence in COPIED
        detail = "rental release deadline reached"
        status = "copied" if copied else "unverified"
        try:
            self.settlement.evidence(current, (), status=status, detail=detail)
        except (MissionError, OSError, ValueError, SQLiteError) as fault:
            detail += f"; evidence recording failed: {fault}"
            logger.error("{}: {}", record.handle, detail)
        try:
            backend().cancel(record.handle)
        except (MissionError, OSError, ValueError) as fault:
            return f"{detail}; release failed and will be retried: {fault}"
        try:
            self._closed(current, copied=copied, detail=detail)
        except (MissionError, OSError, ValueError, SQLiteError) as fault:
            detail += f"; release confirmed but bookkeeping needs repair: {fault}"
        return detail

    def expired(self) -> list[Failed]:
        """Release every rental whose deadline passed, answering the failed row of each."""
        return [
            failed
            for record in self.settlement.cache.listed("tracked")
            if record.lease is not None and record.lease.expired and not uncreated(record)
            if (failed := self._expire(record)) is not None
        ]


@contextmanager
def bounded_pass() -> Generator[str]:
    """Kill the whole process if its parent session dies or the block outlives `_PASS_SECONDS`.

    Yields the deadline as a timestamp. A handler could wait behind the very call that is stuck,
    so the kill is SIGKILL and the OS frees whatever lock the pass held.
    """
    parent = os.getppid()
    deadline = monotonic() + _PASS_SECONDS
    ended = threading.Event()

    def watch() -> None:
        while not ended.wait(_PARENT_POLL):
            orphaned = os.getppid() != parent
            if orphaned or monotonic() >= deadline:
                logger.error(
                    "settlement pass stopped: {}",
                    "its parent session ended" if orphaned else "it outlived its deadline",
                )
                os.kill(os.getpid(), signal.SIGKILL)

    watcher = threading.Thread(target=watch, name="settlement-watchdog", daemon=True)
    watcher.start()
    try:
        yield (datetime.now(UTC) + timedelta(seconds=_PASS_SECONDS)).isoformat()
    finally:
        ended.set()
        watcher.join()
