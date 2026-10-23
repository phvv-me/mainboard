"""The settling pass's event stream: each moved job once, named from the dispatch cache, read by
a follower that never rereads a line nor reads half of one."""

from pathlib import Path
from types import SimpleNamespace

from mainboard.dispatch.state.monitor import Failed, Finished, Held, MonitorReport, Resumed
from mainboard.events import Events, Kind


class _Cache:
    """The dispatch cache, knowing one run."""

    def run(self, handle: str, host: str) -> SimpleNamespace:
        if (handle, host) != ("7", "gold"):
            raise LookupError(handle)
        return SimpleNamespace(label="cutok-v26", node="tokenizers")


def test_a_sweep_appends_each_moved_job_and_a_follower_reads_it_once(tmp_path: Path) -> None:
    events = Events(tmp_path)
    board = SimpleNamespace(dispatcher=SimpleNamespace(cache=_Cache()))
    report = MonitorReport(
        resumed=[Resumed(handle="9", target="miyabi-g", name="wave-2")],
        held=[Held(handle="h1", target="miyabi-g", reason="queue full")],
        finished=[Finished(handle="7", target="gold", pulled_path="evidence/run7")],
        failed=[Failed(handle="8", target="purple", reason="exit 1")],
    )

    recorded = events.record(board, report)

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
    assert events.record(board, MonitorReport()) == []
