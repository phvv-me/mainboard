from typing import TYPE_CHECKING

from mainboard.observe import Frame, Kind

from .support import AT

if TYPE_CHECKING:
    from mainboard.observe import Store


def test_ingest_and_tail_recover_every_frame_in_offset_order_exactly_once(store: Store) -> None:
    """A channel retry replays a batch, and a replayed offset is a row that is already there."""
    first = Frame(job="ordered", kind=Kind.line, offset=0, at=AT, payload={"text": "a"})
    second = Frame(job="ordered", kind=Kind.line, offset=10, at=AT, payload={"text": "b"})
    store.ingest([second, first])
    store.ingest([first])
    assert [frame.offset for frame in store.tail("ordered", since_offset=-1)] == [0, 10]
    assert [frame.offset for frame in store.tail("ordered", since_offset=0)] == [10]


def test_ingest_projects_started_ended_and_a_sample_into_their_own_tables(store: Store) -> None:
    """The `jobs` row is the summary a listing reads, and a sample lands in its own series."""
    store.ingest([Frame(job="projected", kind=Kind.started, offset=0, at=AT, payload={})])
    running = store.connection.execute("SELECT * FROM jobs WHERE job = 'projected'").fetchone()
    assert running["state"] == "running"
    assert running["started_at"] == AT.isoformat()
    store.ingest(
        [
            Frame(job="projected", kind=Kind.sample, offset=10, at=AT, payload={"rss": 4096}),
            Frame(job="projected", kind=Kind.ended, offset=20, at=AT, payload={"exit_code": 3}),
        ]
    )
    settled = store.connection.execute("SELECT * FROM jobs WHERE job = 'projected'").fetchone()
    assert settled["state"] == "ended"
    assert settled["ended_at"] == AT.isoformat()
    assert settled["exit_code"] == 3
    sample = store.connection.execute("SELECT * FROM samples WHERE job = 'projected'").fetchone()
    assert sample["rss"] == 4096
