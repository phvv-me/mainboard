# Durable job history: one WAL-mode SQLite file, mirroring the JSON-blob-row pattern
# `dispatch/state/storage.py` already uses. `patos.sql` (SQLModel) needs its `sql` extra
# (sqlalchemy, sqlmodel) that mainboard does not declare as a dependency, so this is the
# stdlib `sqlite3` fallback the task explicitly allows; see the report for why.

import sqlite3
from typing import TYPE_CHECKING

from .frames import Frame, Kind

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import TracebackType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    exit_code INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    job TEXT NOT NULL,
    offset INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (job, offset)
);
CREATE TABLE IF NOT EXISTS samples (
    job TEXT NOT NULL,
    offset INTEGER NOT NULL,
    at TEXT NOT NULL,
    rss INTEGER,
    PRIMARY KEY (job, offset)
);
"""


class Store:
    """Job history in one WAL-mode SQLite file: `jobs`, `events`, and `samples`.

    `ingest` is idempotent on `(job, offset)`, so replaying an already-recorded batch (a
    channel retry, a resumed follow) never duplicates a row.
    """

    def __init__(self, path: Path) -> None:
        self.connection = self.__connect(path)

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        """Release the underlying SQLite connection."""
        self.connection.close()

    def ingest(self, frames: Sequence[Frame]) -> None:
        """Durably record `frames`, ignoring any offset already stored for its job.

        frames: a batch just fetched, in any order.
        """
        for frame in frames:
            self.connection.execute(
                "INSERT OR IGNORE INTO events (job, offset, data) VALUES (?, ?, ?)",
                (frame.job, frame.offset, frame.model_dump_json()),
            )
            self.__project(frame)

    def tail(self, job: str, since_offset: int) -> list[Frame]:
        """Every frame recorded for `job` after `since_offset`, oldest first."""
        rows = self.connection.execute(
            "SELECT data FROM events WHERE job = ? AND offset > ? ORDER BY offset",
            (job, since_offset),
        ).fetchall()
        return [Frame.model_validate_json(row["data"]) for row in rows]

    @staticmethod
    def __connect(path: Path) -> sqlite3.Connection:
        """Open the observe history database in WAL autocommit mode, creating its schema."""
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10.0, autocommit=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.executescript(_SCHEMA)
        return connection

    def __project(self, frame: Frame) -> None:
        """Fold `frame` into the `jobs` summary row and, for a sample, into `samples` too."""
        if frame.kind is Kind.started:
            self.connection.execute(
                "INSERT INTO jobs (job, state, started_at) VALUES (?, 'running', ?) "
                "ON CONFLICT(job) DO UPDATE SET "
                "state = 'running', started_at = excluded.started_at",
                (frame.job, frame.at.isoformat()),
            )
        elif frame.kind is Kind.ended:
            self.connection.execute(
                "INSERT INTO jobs (job, state, ended_at, exit_code) VALUES (?, 'ended', ?, ?) "
                "ON CONFLICT(job) DO UPDATE SET "
                "state = 'ended', ended_at = excluded.ended_at, exit_code = excluded.exit_code",
                (frame.job, frame.at.isoformat(), frame.payload.get("exit_code")),
            )
        elif frame.kind is Kind.sample:
            self.connection.execute(
                "INSERT OR IGNORE INTO samples (job, offset, at, rss) VALUES (?, ?, ?, ?)",
                (frame.job, frame.offset, frame.at.isoformat(), frame.payload.get("rss")),
            )
