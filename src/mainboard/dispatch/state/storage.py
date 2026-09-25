"""The dispatch state store: one WAL-mode SQLite file (`{STATE_DIR}/db.sqlite`).

Rows are JSON blobs: host facts, the run registry and the history log are all regenerable, so
the schema stays loose.
"""

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hosts (alias TEXT PRIMARY KEY, facts TEXT NOT NULL, probed_at TEXT);
CREATE TABLE IF NOT EXISTS runs (target TEXT NOT NULL, handle TEXT NOT NULL, data TEXT NOT NULL,
    submitted_at TEXT NOT NULL, PRIMARY KEY (target, handle, submitted_at));
CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open the state database, creating the schema on first use.

    WAL lets concurrent commands read without blocking, `busy_timeout` retries a locked write
    rather than failing, and autocommit keeps each upsert a single atomic statement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0, autocommit=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=10000")
    connection.executescript(_SCHEMA)
    return connection
