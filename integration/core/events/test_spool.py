"""The event spool: OTLP log records as JSON lines, one writer a file, read back by byte offset."""

import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.core.events import Event, EventSpool, Identity

_NOON = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
_IDENTITY = Identity(host="gold", workspace="life", session="claude:0b6e1f", job="gold/7")
_SETTLED = Event(
    name="mb.job.settled",
    at=_NOON,
    body="ok",
    attributes={
        **{key.value: value for key, value in _IDENTITY.attributes().items()},
        "cost_usd": 0.0123,
        "n": 5,
        "ok": True,
        "tags": ["a", "b"],
    },
)

_SCALAR = (
    st.text(max_size=12)
    | st.booleans()
    | st.integers(-(2**63), 2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
)
_EVENT = st.builds(
    Event,
    name=st.text(min_size=1, max_size=20),
    at=st.datetimes(
        min_value=datetime(1970, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
    ),
    body=st.text(max_size=20),
    severity=st.integers(0, 24),
    attributes=st.dictionaries(
        st.text(min_size=1, max_size=8), _SCALAR | st.lists(_SCALAR, max_size=3)
    ),
)


def _spool(tmp_path: Path, *events: Event) -> Path:
    """A spool file holding `events`, its writer closed."""
    path = tmp_path / "spool" / "settle.jsonl"
    with EventSpool(path) as spool:
        for event in events:
            spool.append(event)
    return path


@settings(max_examples=200)
@given(event=_EVENT)
@example(event=_SETTLED)
@example(event=Event(name="mb.edge", at=_NOON, attributes={"big": 2**63 - 1, "": 0, "e": []}))
def test_an_event_survives_its_line(event: Event) -> None:
    assert Event.parse(event.line()) == event
    assert "\n" not in event.line()


def test_a_spool_line_is_one_otlp_request_the_sink_takes_as_it_stands() -> None:
    line = json.loads(_SETTLED.line())
    record = line["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert line["resourceLogs"][0]["scopeLogs"][0]["scope"] == {"name": "mb.spool", "version": "1"}
    assert (record["eventName"], record["timeUnixNano"], record["severityNumber"]) == (
        "mb.job.settled",
        str(int(_NOON.timestamp()) * 10**9),
        9,
    )
    assert {"key": "n", "value": {"intValue": "5"}} in record["attributes"]
    assert {"key": "ok", "value": {"boolValue": True}} in record["attributes"]


def test_a_read_resumes_at_the_offset_the_last_one_ended_on(tmp_path: Path) -> None:
    later = _SETTLED.model_copy(update={"at": _NOON + timedelta(seconds=1)})
    path = _spool(tmp_path, _SETTLED)
    first = EventSpool.read(path)
    with EventSpool(path) as spool:
        spool.append(later)
    second = EventSpool.read(path, first.offset)
    assert (first.events, second.events) == ((_SETTLED,), (later,))
    assert second.offset == path.stat().st_size


def test_a_line_still_being_written_waits_for_the_next_read(tmp_path: Path) -> None:
    path = _spool(tmp_path, _SETTLED)
    complete = path.stat().st_size
    path.write_bytes(path.read_bytes() + _SETTLED.line().encode()[:40])
    page = EventSpool.read(path)
    assert (page.events, page.offset) == ((_SETTLED,), complete)


def test_a_writer_cuts_the_half_line_a_crash_left_before_appending(tmp_path: Path) -> None:
    path = _spool(tmp_path, _SETTLED)
    path.write_bytes(path.read_bytes() + b'{"resourceLogs":[{"res')
    with EventSpool(path) as spool:
        spool.append(_SETTLED)
    assert EventSpool.read(path).events == (_SETTLED, _SETTLED)


def test_a_second_writer_is_refused_and_the_file_stays_as_it_was(tmp_path: Path) -> None:
    path = _spool(tmp_path, _SETTLED)
    before = path.read_bytes()
    with EventSpool(path), pytest.raises(MissionError, match="already has a writer"):
        EventSpool(path)
    assert path.read_bytes() == before


def test_a_closed_writer_releases_the_file_and_it_is_private(tmp_path: Path) -> None:
    path = _spool(tmp_path, _SETTLED)
    with EventSpool(path):
        pass
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        pytest.param(
            _SETTLED.line().replace('"version":"1"', '"version":"2"'),
            "reads 'mb.spool' '1'",
            id="a-newer-schema",
        ),
        pytest.param('{"resourceLogs":[]}', "not enough values", id="a-request-with-no-records"),
        pytest.param("not json", "Invalid JSON", id="text-that-is-not-json"),
    ],
)
def test_a_line_of_another_schema_or_shape_is_refused_with_its_offset(
    tmp_path: Path, line: str, reason: str
) -> None:
    path = _spool(tmp_path, _SETTLED)
    start = path.stat().st_size
    path.write_bytes(path.read_bytes() + f"{line}\n".encode())
    with pytest.raises(ValueError, match=f"(?s)at byte {start}.*{reason}"):
        EventSpool.read(path)


def test_the_mainboard_duckdb_reads_the_spool_lines_with_read_json(tmp_path: Path) -> None:
    path = _spool(
        tmp_path, _SETTLED, _SETTLED.model_copy(update={"name": "mb.job.failed", "severity": 17})
    )
    rows = (
        duckdb.connect()
        .execute(
            f"""
        SELECT r.eventName, r.severityNumber, r.body.stringValue,
               epoch_ms(r.timeUnixNano::BIGINT // 1000000),
               list_extract([a.value.stringValue FOR a IN r.attributes IF a.key = 'session.id'], 1)
        FROM (SELECT unnest(rl.scopeLogs[1].logRecords) AS r
              FROM (SELECT unnest(resourceLogs) AS rl
                    FROM read_json('{path}', format = 'newline_delimited')))
        """
        )
        .fetchall()
    )
    assert rows == [
        ("mb.job.settled", 9, "ok", _NOON.replace(tzinfo=None), "claude:0b6e1f"),
        ("mb.job.failed", 17, "ok", _NOON.replace(tzinfo=None), "claude:0b6e1f"),
    ]
