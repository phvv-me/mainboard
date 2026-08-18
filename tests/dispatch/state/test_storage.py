from typing import TYPE_CHECKING

from mainboard.dispatch.state import connect

if TYPE_CHECKING:
    from pathlib import Path


def test_connect_creates_the_schema_and_enables_wal(tmp_path: Path) -> None:
    connection = connect(tmp_path / "sub" / "db.sqlite")
    tables = {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"hosts", "runs", "history"} <= tables
    [(mode,)] = connection.execute("PRAGMA journal_mode").fetchall()
    assert mode.lower() == "wal"


def test_connect_is_idempotent_on_an_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    connect(path)
    connection = connect(path)  # must not raise on a re-open
    connection.execute("INSERT INTO history (data) VALUES ('{}')")
    [(count,)] = connection.execute("SELECT COUNT(*) FROM history").fetchall()
    assert count == 1
