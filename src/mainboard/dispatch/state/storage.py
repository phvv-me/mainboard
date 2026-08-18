"""The dispatch state store: one WAL-mode SQLite file (`{STATE_DIR}/db.sqlite`).

SQLite in WAL mode is concurrent-safe by construction: readers never block, writes serialize
with a busy timeout, and each upsert is atomic. Rows keep their flexible shape as JSON blobs
(the state, host facts, the run registry, the history log, is all regenerable, so the schema
stays loose).
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
    """Open the state database in WAL autocommit mode, creating the schema on first use.

    WAL lets concurrent dispatch commands read without blocking and serialize writes safely;
    `busy_timeout` retries a locked write rather than failing. Autocommit keeps each
    upsert/insert a single atomic statement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0, autocommit=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=10000")
    connection.executescript(_SCHEMA)
    return connection
