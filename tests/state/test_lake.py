import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from filelock import FileLock

from mainboard import MissionError
from mainboard.state import Finding, Lake
from mainboard.state.lake import Session, cache_home, insert, locked, recoverable
from mainboard.state.schema import TABLES, VIEWS

from .conftest import finished, writers

# A process that commits to the catalog and dies before closing it, the state a command killed
# between DuckLake's catalog commit and its close leaves: the commit lives only in the WAL, since
# SQLite copies it back into the database at close. The write is a table DuckLake never reads (a
# value rewritten with itself writes no frame), so the lake means the same either way.
_CRASH = """
import os, sqlite3, sys

catalog = sqlite3.connect(sys.argv[1])
catalog.execute("PRAGMA wal_autocheckpoint = 0")
catalog.execute("CREATE TABLE crash_probe (x)")
catalog.commit()
os._exit(0)
"""


def _files(lake: Lake) -> list[Path]:
    return sorted(lake.data.rglob("*.parquet"))


@pytest.mark.parametrize(
    ("platform", "environ", "expected"),
    [
        ("win32", {"LOCALAPPDATA": "C:/Users/me/AppData/Local"}, "C:/Users/me/AppData/Local"),
        ("win32", {}, "/home/me/.cache"),
        ("darwin", {}, "/home/me/Library/Caches"),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, "/xdg"),
        ("linux", {}, "/home/me/.cache"),
    ],
    ids=["windows", "windows without its variable", "macos", "xdg", "linux"],
)
def test_cache_home_follows_each_platform_convention(
    platform: str, environ: dict[str, str], expected: str
) -> None:
    """Extensions live under the user's cache wherever the OS puts it, never in a workspace."""
    assert cache_home(platform, environ, Path("/home/me")) == Path(expected)


def test_a_lake_is_only_ever_created_on_purpose(tmp_path: Path) -> None:
    """A missing catalog is an error and stays missing; `create` alone makes one, once.

    An ATTACH would quietly create an empty lake, and even one told not to leaves an empty
    SQLite file behind, so the refusal must come before any attach.
    """
    lake = Lake.at(tmp_path)
    assert lake.extensions.is_relative_to(cache_home())
    with pytest.raises(MissionError, match="no state lake"):
        lake.append("costs", [{"provider": "vast"}])
    with pytest.raises(MissionError, match="no state lake"):
        lake.upgrade()
    assert not lake.catalog.exists()
    assert not lake.data.exists()
    spec = lake.create()
    assert spec and lake.spec() == spec
    assert lake.catalog.parent == lake.data.parent
    options = dict(lake.query("SELECT option_name, value FROM lake.options()").rows())
    assert options["parquet_compression"] == "zstd"
    assert options["parquet_compression_level"] == "3"
    assert options["expire_older_than"] == "30 days"
    assert options["delete_older_than"] == "7 days"
    assert options["version"] == spec
    assert lake.query("SELECT version, spec FROM lake.schema_log").rows() == [(1, spec)]
    with pytest.raises(MissionError, match="already exists"):
        lake.create()
    with lake.open(write=True) as connection:
        assert insert(connection, "costs", []) == 0
    with lake.open() as connection, pytest.raises(duckdb.Error, match="read-only"):
        connection.execute("INSERT INTO lake.costs (provider) VALUES ('vast')")


def test_no_schema_name_is_a_duckdb_keyword() -> None:
    """Every table, view and column is usable bare: none is a keyword outside the unreserved.

    DuckDB 2.0 made `at` a type-function keyword, which would force quoting in every query.
    """
    keywords = dict(
        duckdb.connect()
        .execute("SELECT keyword_name, keyword_category FROM duckdb_keywords()")
        .fetchall()
    )
    assert keywords["at"] != "unreserved"
    names = [
        *(table.name for table in TABLES),
        *(view.name for view in VIEWS),
        *(column for table in TABLES for column in table.names),
    ]
    assert [name for name in names if keywords.get(name, "unreserved") != "unreserved"] == []


def test_current_state_is_the_last_append_per_key(lake: Lake) -> None:
    """Nothing is updated in place: a view reads the last record per key, and a drop hides it."""
    lake.append(
        "runs_log",
        [
            {"target": "gold", "handle": "1", "submitted_at": "s", "record": {"verdict": None}},
            {"target": "gold", "handle": "2", "submitted_at": "s", "record": "{}"},
        ],
    )
    lake.append(
        "runs_log",
        [
            {"target": "gold", "handle": "1", "submitted_at": "s", "record": {"verdict": "ok"}},
            {"target": "gold", "handle": "2", "submitted_at": "s", "dropped": True},
        ],
    )
    assert lake.query("SELECT handle, record->>'verdict' FROM lake.runs").rows() == [("1", "ok")]
    lake.append("host_facts", [{"alias": "a", "facts": "{}"}, {"alias": "b", "facts": "{}"}])
    lake.append("host_facts", [{"alias": "a", "facts": {"v": 2}}, {"alias": "b", "dropped": 1}])
    assert lake.query("SELECT alias, facts FROM lake.hosts").rows() == [("a", '{"v":2}')]
    lake.append("holds_log", [{"alias": "h", "held": {"gpu": "5090"}}])
    lake.append("holds_log", [{"alias": "h", "dropped": True}])
    assert lake.query("SELECT * FROM lake.holds").is_empty()
    lake.append("quotes", [{"ts": "2026-09-01T00:00:00Z", "gpu": "a"}, {"gpu": "b"}])
    lake.append("quotes", [{"ts": "2026-09-02T00:00:00Z", "gpu": "c"}])
    assert lake.query("SELECT gpu FROM lake.offers").rows() == [("c",)]
    assert lake.query("SELECT count(*) FROM lake.runs_log").item() == 4


def test_a_moved_workspace_still_opens_its_lake(tmp_path: Path) -> None:
    """The catalog records an absolute data path; every attach overrides it from the root."""
    lake = Lake.at(tmp_path / "here")
    lake.create()
    lake.append("costs", [{"provider": "vast", "t_submit": float(n)} for n in range(3000)])
    assert _files(lake)
    shutil.move(tmp_path / "here", tmp_path / "there")
    moved = Lake.at(tmp_path / "there")
    assert moved.query("SELECT sum(t_submit) FROM lake.costs").item() == sum(range(3000))
    assert moved.check().ok
    listed = moved.query("SELECT data_file FROM ducklake_list_files('lake', 'costs')")
    assert all(Path(path).is_relative_to(moved.data) for path in listed["data_file"])


def test_check_finds_a_deleted_data_file_that_count_hides(lake: Lake) -> None:
    """A lost Parquet file breaks only its own table, and `count(*)` still answers."""
    lake.append("costs", [{"provider": "vast"} for _ in range(3000)])
    lake.append("pulse", [{"key": str(n)} for n in range(3000)])
    assert lake.check().ok
    (lost,) = (path for path in _files(lake) if "costs" in path.parts)
    lost.unlink()
    health = lake.check()
    assert not health.ok
    assert health.findings == (Finding(table="costs", kind="missing", detail=str(lost)),)
    assert lake.query("SELECT count(*) FROM lake.costs").item() == 3000
    with pytest.raises(duckdb.Error):
        lake.query("SELECT list(provider) FROM lake.costs")
    assert lake.query("SELECT count(key) FROM lake.pulse").item() == 3000


def test_check_names_a_missing_or_unreadable_catalog(tmp_path: Path) -> None:
    lake = Lake.at(tmp_path)
    assert [finding.kind for finding in lake.check().findings] == ["catalog"]
    lake.catalog.parent.mkdir(parents=True)
    lake.catalog.write_bytes(b"this is not a database" * 100)
    (finding,) = lake.check().findings
    assert finding.kind == "catalog"
    assert finding.table == ""


def test_check_reports_a_lost_wal_only_when_its_index_proves_unsaved_commits(
    lake: Lake,
) -> None:
    """A crash leaves committed frames in the catalog's WAL; losing that file loses them.

    The WAL index in `-shm` says how many frames were committed and how many were copied back,
    so only a WAL that held uncopied frames is reported, never a clean close or a stray index.
    """
    crashed = subprocess.run([sys.executable, "-c", _CRASH, str(lake.catalog)], check=False)
    assert crashed.returncode == 0
    wal, index = Path(f"{lake.catalog}-wal"), Path(f"{lake.catalog}-shm")
    assert wal.stat().st_size and index.is_file()
    assert lake.check().ok
    wal.unlink()
    assert "wal" in {finding.kind for finding in lake.check().findings}
    for header in (bytes(136), bytes(10)):
        wal.unlink(missing_ok=True)
        index.write_bytes(header)
        assert "wal" not in {finding.kind for finding in lake.check().findings}


def test_two_processes_appending_at_once_lose_nothing(lake: Lake) -> None:
    """Each append attaches, commits and detaches; a lost race is retried, never dropped."""
    assert finished(writers(lake.root, ("left", "right"), 15)) == []
    counted = lake.query("SELECT batch, count(*) FROM lake.events GROUP BY batch ORDER BY 1")
    assert counted.rows() == [("left", 15), ("right", 15)]


def test_maintenance_takes_its_lock_and_appends_carry_on_beside_it(lake: Lake) -> None:
    """One maintainer at a time, a second one stands down, and writers never wait on it."""
    for _ in range(4):
        lake.append("costs", [{"provider": "vast"} for _ in range(1500)])
    lake.lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lake.lock):
        assert lake.maintain() is False
        lake.append("costs", [{"provider": "held"}])
    referenced = "SELECT count(*) FROM ducklake_list_files('lake', 'costs')"
    before = lake.query(referenced).item()
    running = writers(lake.root, ("left", "right"), 10)
    maintained = [lake.maintain() for _ in range(3)]
    assert finished(running) == []
    assert all(maintained)
    assert lake.maintain()
    assert lake.query(referenced).item() < before
    assert lake.query("SELECT count(*) FROM lake.costs").item() == 6001
    assert lake.query("SELECT count(*) FROM lake.events").item() == 20
    assert lake.check().ok


def test_only_a_locked_catalog_is_worth_another_attach() -> None:
    """SQLite's immediate refusal of a stale snapshot is retried; any other fault is not."""
    assert locked(duckdb.Error("Failed to perform CHECKPOINT: database is locked"))
    assert not locked(duckdb.Error("Catalog Error: Table with name nothing does not exist"))
    assert not locked(MissionError("database is locked"))


def test_opening_never_migrates_and_upgrade_does_once(tmp_path: Path) -> None:
    """A catalog at an older spec keeps it until `upgrade`, which records the move once."""
    lake = Lake.at(tmp_path)
    lake.data.mkdir(parents=True)
    connection = duckdb.connect()
    lake._load(connection)
    connection.execute(
        f"ATTACH 'ducklake:sqlite:{lake.catalog.as_posix()}' AS lake "
        f"(DATA_PATH '{lake.data.as_posix()}/', DUCKLAKE_VERSION '1.0')"
    )
    connection.execute("USE lake")
    for table in TABLES:
        connection.execute(table.ddl)
    connection.close()
    assert lake.spec() == "1.0"
    assert lake.spec() == "1.0"
    latest = lake.upgrade()
    assert latest != "1.0"
    assert lake.spec() == latest
    assert lake.upgrade() == latest
    assert lake.query("SELECT spec FROM lake.schema_log").rows() == [(latest,)]


def test_extensions_install_once_and_load_offline_after(lake: Lake, tmp_path: Path) -> None:
    """Loading comes first, so a machine that installed once opens a lake with no network.

    The populated extension folder doubles as a local repository, standing in for the network.
    """
    repository = lake.extensions.as_posix()
    nowhere = (tmp_path / "nowhere").as_posix()
    fresh = tmp_path / "extensions"
    installed = Lake(root=lake.root, extensions=fresh, repository=repository)
    assert installed.query("SELECT count(*) FROM lake.costs").item() == 0
    assert any(fresh.rglob("ducklake.duckdb_extension"))
    offline = Lake(root=lake.root, extensions=fresh, repository=nowhere)
    assert offline.spec() == lake.spec()
    stranded = Lake(root=lake.root, extensions=tmp_path / "empty", repository=nowhere)
    with pytest.raises(MissionError, match="could not be installed"):
        stranded.spec()


def test_a_failed_operation_still_closes_its_connection(lake: Lake) -> None:
    """No detach on failure, but the connection closes, so nothing holds the catalog open."""
    with pytest.raises(duckdb.Error):
        lake.query("SELECT * FROM lake.nothing_here")
    os.replace(lake.catalog, lake.catalog.with_suffix(".moved"))
    os.replace(lake.catalog.with_suffix(".moved"), lake.catalog)
    assert lake.check().ok


def test_a_session_attaches_once_and_again_only_after_a_refusal(
    lake: Lake, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked or stale catalog costs one fresh attach; the statement then runs again."""
    monkeypatch.setattr("mainboard.state.lake._LOCKED_WAIT_S", 0)
    session = Session(lake)
    seen: list[duckdb.DuckDBPyConnection] = []

    def flaky(connection: duckdb.DuckDBPyConnection) -> int:
        seen.append(connection)
        if len(seen) == 1:
            raise duckdb.IOException("database is locked")
        if len(seen) == 2:
            raise duckdb.CatalogException("Table ducklake_inlined_data_1_1 does not exist")
        return 7

    assert session.run(flaky) == 7
    assert len({id(connection) for connection in seen}) == 3
    assert session.append("pulse", [{"key": "gold/1", "size": 3}]) == 1
    assert session.rows("SELECT key FROM lake.pulse") == [("gold/1",)]


def test_a_session_raises_what_another_attach_would_not_fix(lake: Lake) -> None:
    session = Session(lake)
    with pytest.raises(duckdb.CatalogException, match="nonexistent"):
        session.rows("SELECT * FROM lake.nonexistent")
    assert not recoverable(duckdb.CatalogException("Table nonexistent does not exist"))


def test_a_lake_another_process_created_first_is_shared_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raced = Lake.at(tmp_path)
    creating = Lake.create

    def beaten(self: Lake) -> str:
        creating(self)
        raise MissionError("a state lake already exists")

    monkeypatch.setattr(Lake, "create", beaten)
    assert raced.ready().exists()


def test_a_failed_creation_is_not_mistaken_for_a_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(self: Lake) -> str:
        raise MissionError("disk full")

    monkeypatch.setattr(Lake, "create", refused)
    with pytest.raises(MissionError, match="disk full"):
        Lake.at(tmp_path).ready()
