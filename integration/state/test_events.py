"""The settling pass's event stream: each moved job once, named from the dispatch cache, read by
a follower that never rereads a line nor reads half of one."""

from pathlib import Path

from mainboard.dispatch import now
from mainboard.dispatch.state import (
    Cache,
    Failed,
    Finished,
    Held,
    MonitorReport,
    Resumed,
    RunRecord,
)
from mainboard.settling.events import Events, Kind
from mainboard.state import Lake


def _cache(workspace: Path) -> Cache:
    """The workspace's dispatch cache, knowing one run."""
    cache = Cache(Lake.at(workspace))
    cache.record(
        RunRecord(
            handle="7",
            target="gold",
            kind="ssh",
            script="job.sh",
            args="",
            submitted_at=now(),
            name="cutok-v26",
            node="tokenizers",
        )
    )
    return cache


def test_a_sweep_appends_each_moved_job_and_a_follower_reads_it_once(workspace: Path) -> None:
    events = Events(workspace)
    cache = _cache(workspace)
    report = MonitorReport(
        resumed=[Resumed(handle="9", target="miyabi-g", name="wave-2")],
        held=[Held(handle="h1", target="miyabi-g", reason="queue full")],
        finished=[Finished(handle="7", target="gold", pulled_path="evidence/run7")],
        failed=[Failed(handle="8", target="purple", reason="exit 1")],
    )

    recorded = events.record(cache, report)

    assert [(event.kind, event.handle) for event in recorded] == [
        (Kind.DISPATCHED, "9"),
        (Kind.HELD, "h1"),
        (Kind.OK, "7"),
        (Kind.FAILED, "8"),
    ]
    assert (recorded[2].name, recorded[2].node, recorded[2].detail) == (
        "cutok-v26",
        "tokenizers",
        "evidence/run7",
    )
    read, cursor = events.since()
    assert read == recorded
    with events.path.open("a", encoding="utf-8") as stream:
        stream.write('{"at": "half a line')
    assert events.since(cursor) == ([], cursor)
    assert events.record(cache, MonitorReport()) == []
