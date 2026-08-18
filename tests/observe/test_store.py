from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mainboard.observe import Frame, Kind, Store

if TYPE_CHECKING:
    from pathlib import Path

_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "history.sqlite")


def test_ingest_and_tail_round_trip_frames_in_offset_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    second = Frame(job="job1", kind=Kind.line, offset=10, at=_AT, payload={"text": "b"})
    first = Frame(job="job1", kind=Kind.line, offset=0, at=_AT, payload={"text": "a"})
    store.ingest([second, first])
    tailed = store.tail("job1", since_offset=-1)
    store.close()
    assert [frame.offset for frame in tailed] == [0, 10]


def test_ingest_is_idempotent_on_job_and_offset(tmp_path: Path) -> None:
    store = _store(tmp_path)
    frame = Frame(job="job1", kind=Kind.line, offset=0, at=_AT, payload={"text": "a"})
    store.ingest([frame])
    store.ingest([frame])
    tailed = store.tail("job1", since_offset=-1)
    store.close()
    assert len(tailed) == 1


def test_tail_excludes_frames_at_or_before_since_offset(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ingest(
        [
            Frame(job="job1", kind=Kind.line, offset=0, at=_AT, payload={}),
            Frame(job="job1", kind=Kind.line, offset=10, at=_AT, payload={}),
        ]
    )
    tailed = store.tail("job1", since_offset=0)
    store.close()
    assert [frame.offset for frame in tailed] == [10]


def test_ingest_projects_started_and_ended_into_the_jobs_summary(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ingest([Frame(job="job1", kind=Kind.started, offset=0, at=_AT, payload={})])
    row = store.connection.execute("SELECT * FROM jobs WHERE job = 'job1'").fetchone()
    assert row["state"] == "running"
    assert row["started_at"] == _AT.isoformat()
    store.ingest([Frame(job="job1", kind=Kind.ended, offset=10, at=_AT, payload={"exit_code": 3})])
    row = store.connection.execute("SELECT * FROM jobs WHERE job = 'job1'").fetchone()
    store.close()
    assert row["state"] == "ended"
    assert row["exit_code"] == 3


def test_ingest_projects_a_sample_into_the_samples_table(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.ingest([Frame(job="job1", kind=Kind.sample, offset=0, at=_AT, payload={"rss": 4096})])
    row = store.connection.execute("SELECT * FROM samples WHERE job = 'job1'").fetchone()
    store.close()
    assert row["rss"] == 4096


def test_store_is_a_context_manager(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        store.ingest([Frame(job="job1", kind=Kind.line, offset=0, at=_AT, payload={})])
        tailed = store.tail("job1", since_offset=-1)
    assert len(tailed) == 1
