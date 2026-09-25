# Command history for a dispatch CLI in the shared state database: one `history` row per
# subcommand invocation, disabled by `MAINBOARD_NO_HISTORY=1`.

import os
import time
import weakref
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..shared import db_file, now
from .storage import connect

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# A CLI subcommand's positional argument, stringified; History only needs the scalar union.
type CommandArg = str | int | float | bool | None


class HistoryEvent(FrozenModel):
    """One recorded dispatch subcommand invocation.

    at: ISO-8601 instant the command finished.
    command: the subcommand name (`ls`, `submit`, ...).
    target: the host alias the command acted on, when applicable.
    handle: the run handle produced or addressed, when applicable.
    outcome: `ok` if the command returned, `error` if it raised.
    detail: a short human note (e.g. the exception summary on error).
    """

    at: str
    command: str
    args: list[str] = []
    target: str | None = None
    handle: str | None = None
    outcome: str
    detail: str | None = None
    duration_ms: int | None = None


class History:
    """The `history` table of the shared state database, a no-op under `MAINBOARD_NO_HISTORY=1`."""

    def __init__(self, path: Path | None = None) -> None:
        disabled = os.environ.get("MAINBOARD_NO_HISTORY") == "1"
        self.connection = None if disabled else connect(path or db_file())
        if self.connection is not None:
            # The collector closes it, as on the cache: a short-lived log has no caller left.
            weakref.finalize(self, self.connection.close)

    def recent(self, limit: int = 20) -> list[HistoryEvent]:
        """The last `limit` recorded events, oldest-to-newest, or [] if none."""
        if self.connection is None:
            return []
        rows = self.connection.execute(
            "SELECT data FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [HistoryEvent.model_validate_json(row["data"]) for row in reversed(rows)]

    def record(
        self,
        command: str,
        args: Sequence[CommandArg],
        started: float,
        outcome: str,
        *,
        handle: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Append one timed invocation; the target is the first string argument."""
        if self.connection is None:
            return
        event = HistoryEvent(
            at=now(),
            command=command,
            args=[str(arg) for arg in args],
            target=next((arg for arg in args if isinstance(arg, str)), None),
            handle=handle,
            outcome=outcome,
            detail=detail,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        self.connection.execute(
            "INSERT INTO history (data) VALUES (?)", (event.model_dump_json(),)
        )
