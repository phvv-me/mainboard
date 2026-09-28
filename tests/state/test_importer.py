import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path

import duckdb
import pytest

from mainboard import MissionError
from mainboard.cli import build
from mainboard.state import Importer, Lake, Tally
from mainboard.state import importer as importer_module
from mainboard.state.lake import insert

# The registry schema every release before the lake wrote, which an import must still read.
_REGISTRY = """
CREATE TABLE IF NOT EXISTS hosts (alias TEXT PRIMARY KEY, facts TEXT NOT NULL, probed_at TEXT);
CREATE TABLE IF NOT EXISTS runs (target TEXT NOT NULL, handle TEXT NOT NULL, data TEXT NOT NULL,
    submitted_at TEXT NOT NULL, PRIMARY KEY (target, handle, submitted_at));
CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);
"""


def connect(path: Path) -> sqlite3.Connection:
    """A registry as the releases before the lake kept it: WAL-mode SQLite, autocommit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, autocommit=True)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_REGISTRY)
    return connection


# Each batch log as it sits on disk: CRLF endings, a line that is not UTF-8, a NUL byte and no
# final newline, an empty file, and one nested a level down.
_LOGS = {
    "b1/1.log": b"alpha\r\nbeta\r\n",
    "b1/2.log": b"bad \xff\xfe byte\nfine\n",
    "b1/3.log": b"no newline at end\x00with a nul",
    "b2/sub/4.log": b"",
}

_RECEIPT = {
    "trial_receipt": {
        "run": "r1",
        "trial": "t1",
        "verdict": "ok",
        "host": "gold",
        "at": "2026-09-18T02:42:10+00:00",
    }
}


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")


def _lines(*records: object) -> str:
    return "".join(
        (record if isinstance(record, str) else json.dumps(record)) + "\n" for record in records
    )


def stocked(root: Path) -> Path:
    """A workspace state directory holding one of every source, each with a flaw or two.

    Answers the state directory. Things that are not state (wandb folders, pins, archives,
    recovery, environments) are there too, holding files an import must not read.
    """
    out = Lake.at(root).out
    registry = connect(out / "dispatch" / "db.sqlite")
    registry.executescript(
        """
        INSERT INTO runs VALUES ('gold', '1', '{"verdict": "ok"}', '2026-09-01T00:00:00+00:00');
        INSERT INTO runs VALUES ('gold', '2', 'not json', '2026-09-02T00:00:00+00:00');
        INSERT INTO hosts VALUES ('gold', '{"host": "gold"}', '2026-09-01T00:00:00+00:00');
        INSERT INTO hosts VALUES ('lost', '{', NULL);
        """
    )
    registry.close()
    _write(
        out / "batches/b1/events.ndjson",
        _lines(
            {
                "at": "2026-09-25T03:07:54.892074+00:00",
                "batch": "b1",
                "topic": "job.submitted",
                "job": "j",
                "data": {"handle": "1"},
            },
            {"at": "not a time", "topic": "job.state", "data": {}},
            "",
            "{torn",
            "[1, 2]",
        ),
    )
    _write(
        out / "batches/b1/receipts.ndjson",
        _lines(_RECEIPT, "not json at all", {"trial_receipt": "flat"}),
    )
    _write(
        out / "costs/costs.ndjson",
        _lines({"provider": "vast", "gpu": "5090", "t_submit": 1.0, "t_running": 2}),
    )
    _write(out / "catalog.ndjson", _lines({"provider": "vast", "gpu": "5090", "rate_usd_hr": 0.4}))
    _write(out / "dispatch/holds.json", json.dumps([{"alias": "rented", "gpu": "5090"}]))
    _write(
        out / "studies/s1.jsonl",
        _lines({"at": "2026-09-01T00:00:00+00:00", "kind": "created", "name": "first"}),
    )
    _write(out / "pulse.json", json.dumps({"gold/1": [10, 1790322512.6], "gold/2": "bad"}))
    _write(
        out / "dispatch/digests.json",
        json.dumps({"a.txt": [3, 2**63 + 5, 10, 11, "aa"], "b.txt": [1, 2, "bb"]}),
    )
    _write(
        out / "collection.digests.json",
        json.dumps(
            {
                "c.bin": [5, 12, "cc"],
                "d.bin": [5, 1, 2, 3, "dd"],
                "e.bin": "flat",
                "f.bin": [],
                "g.bin": [1, 2],
                "h.bin": [1, "2", "x"],
                "i.bin": [1, 2, 3, "x"],
            }
        ),
    )
    _write(out / "dispatch/jobs/a1.sh", b"#!/bin/bash\r\necho \xff\n")
    _write(
        out / "dispatch/jobs/closure-abc.tsv",
        "src/a.py\tblob1\tclean\n\nsrc/b.py\tblob2\n",
    )
    for name, content in _LOGS.items():
        _write(out / "batches" / name, content)
    for ignored in (
        "batches/b1/wandb/run-1/logs/debug.log",
        "pins/x.log",
        "recovery/x.log",
        "envs/default/x.log",
        "source-archives/x.log",
    ):
        _write(out / ignored, b"not state\n")
    with suppress(OSError):
        os.symlink(out / "batches/b1/1.log", out / "batches/b1/latest.log")
    return out


def _snapshot(out: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}


def _tallied(tallies: Iterable[Tally]) -> dict[str, tuple[int, int, int, bool]]:
    return {
        tally.source: (tally.expected, tally.imported, tally.strays, tally.ok) for tally in tallies
    }


def test_an_import_lands_every_source_and_proves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every record lands once, every flaw lands as a stray, and the logs rebuild exactly.

    Logs are staged a few bytes at a time here, so every file is its own append.
    """
    monkeypatch.setattr(importer_module, "_CHUNK_BYTES", 8)
    out = stocked(tmp_path)
    before = _snapshot(out)
    lake = Lake.at(tmp_path)
    tallies = Importer(lake).run()
    assert all(tally.ok for tally in tallies), tallies
    assert _tallied(tallies) == {
        "dispatch/db.sqlite runs": (2, 1, 1, True),
        "dispatch/db.sqlite hosts": (2, 1, 1, True),
        "batches/*/events.ndjson": (4, 2, 2, True),
        "batches/*/receipts.ndjson": (3, 3, 0, True),
        "costs/*.ndjson": (1, 1, 0, True),
        "catalog.ndjson": (1, 1, 0, True),
        "dispatch/holds.json": (1, 1, 0, True),
        "studies/*.jsonl": (1, 1, 0, True),
        "pulse.json": (2, 1, 1, True),
        "dispatch/digests.json": (2, 2, 0, True),
        "collection.digests.json": (7, 2, 5, True),
        "dispatch/jobs/*.sh": (1, 1, 0, True),
        "dispatch/jobs/closure-*.tsv": (2, 1, 1, True),
        "batches/**/*.log": (8, 8, 0, True),
        "batches/**/*.log files": (4, 4, 0, True),
        "batches/**/*.log lossy lines": (1, 1, 0, True),
        "batches/**/*.log bytes": (3, 3, 0, True),
    }
    after = _snapshot(out)
    assert {path: after[path] for path in before} == before
    assert all(
        path.is_relative_to(lake.data)
        or path.is_relative_to(lake.lock.parent)
        or path.name.startswith("lake.sqlite")
        for path in after.keys() - before.keys()
    )
    for name, content in _LOGS.items():
        rows = lake.query(
            "SELECT line, lossy FROM lake.log_lines WHERE file = ? ORDER BY n", [name]
        ).rows()
        rebuilt = "\n".join(line for line, _ in rows).encode()
        lossy = any(damaged for _, damaged in rows)
        assert rebuilt == (content.decode("utf-8", "replace").encode() if lossy else content)
        assert lossy == (name == "b1/2.log")
    assert lake.query("SELECT DISTINCT batch FROM lake.log_lines ORDER BY 1")[
        "batch"
    ].to_list() == [
        "b1",
        "b2",
    ]
    assert lake.query("SELECT ts, batch FROM lake.events ORDER BY ts NULLS LAST").rows()[1] == (
        None,
        "b1",
    )
    assert lake.query("SELECT inode, mtime_ns FROM lake.digests WHERE path = 'a.txt'").rows() == [
        (2**63 + 5, 10)
    ]
    assert lake.query("SELECT handle FROM lake.runs").rows() == [("1",)]
    assert lake.query("SELECT alias FROM lake.holds").rows() == [("rented",)]
    assert lake.query("SELECT run FROM lake.receipts WHERE run IS NOT NULL").rows() == [("r1",)]
    strays = lake.query("SELECT source, destination, n FROM lake.strays ORDER BY ALL").rows()
    assert ("batches/b1/events.ndjson", "events", 4) in strays
    assert ("dispatch/jobs/closure-abc.tsv", "closures", 3) in strays


def test_an_import_happens_once_unless_asked_again(tmp_path: Path) -> None:
    """A second import refuses; `again` sets the first lake aside, whole, and starts fresh."""
    stocked(tmp_path)
    lake = Lake.at(tmp_path)
    lake.create()
    first = Importer(lake).run()
    with pytest.raises(MissionError, match="--again"):
        Importer(lake).run()
    again = Importer(lake).run(again=True)
    assert _tallied(again) == _tallied(first)
    (aside,) = (lake.out / "lake.aside").iterdir()
    assert (aside / "lake.sqlite").is_file()
    assert (aside / "lake").is_dir()
    assert lake.query("SELECT count(*) FROM lake.imports").item() == len(first) - 3


def test_an_empty_state_directory_imports_to_nothing(tmp_path: Path) -> None:
    """No source is required; every one missing tallies zero and still verifies."""
    tallies = Importer(Lake.at(tmp_path)).run()
    assert all(tally.ok and tally.expected == tally.imported == 0 for tally in tallies)


@pytest.mark.parametrize(
    ("name", "content", "source"),
    [
        ("dispatch/holds.json", "not json", "dispatch/holds.json"),
        ("dispatch/holds.json", "[1]", "dispatch/holds.json"),
        ("pulse.json", "[1]", "pulse.json"),
        ("dispatch/digests.json", "{", "dispatch/digests.json"),
    ],
    ids=["holds unreadable", "holds not machines", "pulse not a map", "digests unreadable"],
)
def test_an_unreadable_document_is_kept_as_one_stray(
    tmp_path: Path, name: str, content: str, source: str
) -> None:
    _write(Lake.at(tmp_path).out / name, content)
    tallies = {tally.source: tally for tally in Importer(Lake.at(tmp_path)).run()}
    assert _tallied([tallies[source]])[source] == (1, 0, 1, True)
    stray = Lake.at(tmp_path).query("SELECT n, line FROM lake.strays").rows()
    assert stray == [(0, content)]


def test_the_registry_wal_is_read_and_the_registry_never_touched(tmp_path: Path) -> None:
    """Commits still in the registry's WAL are imported, and no sidecar file appears.

    A read-only SQLite open of a WAL database creates `-wal` and `-shm`, so the import reads
    a copy; a writer holding the registry open keeps its commits in the WAL meanwhile.
    """
    path = Lake.at(tmp_path).out / "dispatch" / "db.sqlite"
    holder = connect(path)
    holder.execute("PRAGMA wal_autocheckpoint = 0")
    holder.execute("INSERT INTO runs VALUES ('gold', '9', '{}', 's')")
    wal = Path(f"{path}-wal")
    assert wal.stat().st_size
    before = _snapshot(path.parent)
    tallies = _tallied(Importer(Lake.at(tmp_path)).run())
    assert tallies["dispatch/db.sqlite runs"] == (1, 1, 0, True)
    assert _snapshot(path.parent) == before
    holder.close()


def _losing(table: str, change: str) -> object:
    """An `insert` that damages one row of `table` on its way in: drops it, or alters a line."""

    def damaged(
        connection: duckdb.DuckDBPyConnection, into: str, rows: Iterable[Mapping[str, object]]
    ) -> int:
        staged = [dict(row) for row in rows]
        if into == table and staged:
            if change == "drop":
                staged.pop()
            else:
                staged[0]["line"] = f"{staged[0]['line']}!"
        return insert(connection, into, staged)

    return damaged


@pytest.mark.parametrize(
    ("table", "change", "source"),
    [
        ("events", "drop", "batches/*/events.ndjson"),
        ("log_lines", "alter", "batches/**/*.log bytes"),
    ],
    ids=["a row lost", "a log byte changed"],
)
def test_any_difference_fails_the_verb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    table: str,
    change: str,
    source: str,
) -> None:
    """The verification reads the lake back, so a damaged import is caught and exits 1."""
    stocked(tmp_path / "clean")
    with pytest.raises(SystemExit, match="0"):
        build(tmp_path / "clean")(["center", "migrate-state", "--agent"])
    assert "batches/**/*.log bytes" in capsys.readouterr().out
    stocked(tmp_path / "damaged")
    monkeypatch.setattr(importer_module, "insert", _losing(table, change))
    with pytest.raises(SystemExit, match="1"):
        build(tmp_path / "damaged")(["center", "migrate-state", "--json"])
    printed = {row["source"]: row for row in json.loads(capsys.readouterr().out)}
    assert not printed[source]["ok"]
    assert sum(not row["ok"] for row in printed.values()) == 1


def test_the_registry_opens_as_sqlite_and_nothing_else(tmp_path: Path) -> None:
    """A registry that is not a database fails loudly rather than importing as empty."""
    _write(Lake.at(tmp_path).out / "dispatch" / "db.sqlite", b"garbage" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        Importer(Lake.at(tmp_path)).run()
