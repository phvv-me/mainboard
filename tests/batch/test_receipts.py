from pathlib import Path

from mainboard.batch import Receipts, Topic, TransferSet
from mainboard.batch.receipts import latest, payload, publish


def test_a_published_line_carries_the_envelope_and_reads_back_as_itself(tmp_path: Path) -> None:
    """The file transport is one NDJSON line per event, written whole and replayed whole."""
    bus = Receipts(tmp_path / "deep" / "events.ndjson")
    assert bus.replay() == []
    event = publish(bus, "smoke-1", Topic.SUBMITTED, job="a", data={"handle": "7"})
    assert (event.batch, event.topic, event.job, event.data) == (
        "smoke-1",
        Topic.SUBMITTED,
        "a",
        {"handle": "7"},
    )
    assert event.at
    assert bus.replay() == [event]
    assert bus.path.read_text().count("\n") == 1


def test_a_blank_line_in_the_log_replays_as_nothing_rather_than_as_a_failure(
    tmp_path: Path,
) -> None:
    bus = Receipts(tmp_path / "events.ndjson")
    publish(bus, "smoke-1", Topic.OPENED, data={"name": "smoke"})
    bus.path.write_text(bus.path.read_text() + "\n\n")
    assert [event.topic for event in bus.replay()] == [Topic.OPENED]


def test_a_record_becomes_a_payload_the_same_way_a_broker_would_carry_it() -> None:
    """The payload is the model's own JSON shape, so nothing depends on python types surviving."""
    measured = TransferSet(job="a", target="gold", paths=("packages",), files=2, raw_bytes=9)
    assert payload(measured) == {
        "job": "a",
        "target": "gold",
        "paths": ["packages"],
        "files": 2,
        "raw_bytes": 9,
        "wire_bytes": 0,
        "since": "",
    }
    assert TransferSet.model_validate(payload(measured)) == measured


def test_the_cursor_a_resumed_pass_reads_is_the_last_line_per_job(tmp_path: Path) -> None:
    """A pass reads its cursor out of the log, which is what makes the log the state."""
    bus = Receipts(tmp_path / "events.ndjson")
    publish(bus, "b", Topic.STATE, job="a", data={"verdict": "running"})
    publish(bus, "b", Topic.STATE, job="a", data={"verdict": "ok"})
    publish(bus, "b", Topic.STATE, job="z", data={"verdict": "running"})
    events = bus.replay()
    assert {job: event.data["verdict"] for job, event in latest(events, Topic.STATE).items()} == {
        "a": "ok",
        "z": "running",
    }
    assert latest(events, Topic.SETTLED) == {}
